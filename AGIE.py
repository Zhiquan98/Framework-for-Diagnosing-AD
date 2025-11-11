import torch.nn as nn
import torch
import einops


class DropPath(nn.Module):
    def __init__(self, prob):
        """
        对于网络中的一些路径 Path 。比如跳连结构。以 prob 的概率在一次前向传播过程中，舍弃该路径
        具体方式为，对于 路径上的 数据 【batch, C, any】，以 prob 的概率舍弃其中某些 batch 的数据
        从而实现某些数据样本的前向传播过程中，弃用路径
        所以该结构使用时不能用在 主路上  x = self.Droppath(x)
        而是应该在 侧路径上 x = x + self.Droppath(  侧路结构（x）  )
        :param prob:
        """
        super(DropPath, self).__init__()
        self.keep_prob = 1 - prob

    def forward(self, x):
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # [batch, 1, ...]  除了 batch 维度不变，其余变成1

        random_tensor = self.keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        # 生成 [batch,1,...] 的矩阵，里面是 0-1 的随机数，加保存概率，变成 保存概率-1+保存概率

        random_tensor.floor_()  # 向下取整，小于1的变成0，大于1的变成1。即 [batch,1,...] 的矩阵，里面是 0 或 1
        output = x.div(self.keep_prob) * random_tensor  # 丢弃部分 batch 数据，剩余数据除保存概率，按比例放大
        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads, attention_dropout=0.1, projection_dropout=0.1):
        """

        :param dim:
        :param num_heads:
        :param attention_dropout:
        :param projection_dropout:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, q_token, kv_token):
        """
        q 和 kv 如果送进来同一个参数 [batch, Num_token, dim]
        那就是普通的 SA 结构，得到同样 shape 的自注意力结果[batch, num_token, dim]
        :param q:
        :param kv:
        :return:
        """
        B, N, C = q_token.shape

        # 映射一下并把 channel 切分多头
        Q = einops.rearrange(self.q(q_token), 'b token (nums head_dim) -> b nums token head_dim', nums=self.num_heads, head_dim=C // self.num_heads)
        KV = einops.rearrange(self.kv(kv_token), 'b token (N nums head_dim) -> N b nums token head_dim', nums=self.num_heads, head_dim=C // self.num_heads)
        K, V = KV[0], KV[1]

        # 对每个头做 SA 计算，获得对 V 加权求和 的权重
        attn = torch.einsum('b N q c, b N k c -> b N q k', Q, K) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 利用获得的权重对 V 进行加权求和
        x = torch.einsum('b N q k, b N v c -> b N q c', attn, V) * self.scale
        # 把多头重新拼接，并用全连接层 proj 融合一下
        x = einops.rearrange(x, 'b nums token head -> b token (nums head)')
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, dim, num_head, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1, out_dim=None):
        super(TransformerEncoderLayer, self).__init__()

        if out_dim is None:
            out_dim = dim

        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.self_attn = Attention(dim=dim, num_heads=num_head,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.norm = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, out_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, src_q: torch.Tensor, src_kv, *args, **kwargs) -> torch.Tensor:
        """
        如果传进来的 src_q 和 src_kv 是同一个参数 [batch, num_token, dim]
        那么先进行自注意力得到 [batch, num_token, dim] 并加上输入（残差）

        然后经过 MLP , 即两次全连接，获得[batch, num_token, out_dim]，一般 out_dim == dim
        所以 shape 不会改变
        :param src_q:
        :param src_kv:
        :param args:
        :param kwargs:
        :return:
        """


        attn = self.self_attn(self.q_norm(src_q), self.kv_norm(src_kv))
        src = src_q + self.drop_path(attn)

        src = self.norm(src)
        src2 = self.MLP(src)
        src = src + self.drop_path(self.dropout(src2))
        return src


class Geometry_information_Extractor(nn.Module):
    def __init__(self, dim, num_head, prompt_length):
        super().__init__()
        self.prompt_length = prompt_length
        prompt_shape = (1, prompt_length) + (dim, )
        self.segmentation_prompt = nn.Parameter(torch.zeros(prompt_shape), requires_grad=True)
        self.classification_prompt = nn.Parameter(torch.zeros(prompt_shape), requires_grad=True)

        self.TransEncoder_1 = TransformerEncoderLayer(dim=dim, num_head=num_head)
        self.TransEncoder_2 = TransformerEncoderLayer(dim=dim, num_head=num_head)
        self.TransEncoder_share = TransformerEncoderLayer(dim=dim, num_head=num_head)
        self.norm = nn.LayerNorm(dim)  # 对 dim 维度归一化

    def forward(self, seg_feature_map, C_feature_map):
        """
        输入两个特征图 [B, C, H, W, D]
        首先转化为 [B, Num_token, dim]，然后各自 self-attn 编码
        编码后的特征加入各自 Prompt 指示特征来源，随后互相索引特征
        返回 [B, DHW, C]
        :param seg_feature_map:
        :param C_feature_map:
        :return:
        """
        seg_token = einops.rearrange(seg_feature_map, 'b c h w d -> b (h w d) c')
        C_token = einops.rearrange(C_feature_map, 'b c h w d -> b (h w d) c')
        B, _, _ = seg_token.shape
        segmentation_prompt = self.segmentation_prompt.expand(B, -1, -1)
        classification_prompt = self.classification_prompt.expand(B, -1, -1)

        seg_embed = self.TransEncoder_1(seg_token, seg_token)
        C_embed = self.TransEncoder_2(C_token, C_token)

        seg_embed = torch.concat((segmentation_prompt, seg_embed), dim=1)
        C_embed = torch.concat((classification_prompt, C_embed), dim=1)

        feature_from_seg = self.norm(self.TransEncoder_share(C_embed, seg_embed))[:, self.prompt_length:, :]
        feature_from_C = self.norm(self.TransEncoder_share(seg_embed, C_embed))[:, self.prompt_length:, :]

        return feature_from_seg, feature_from_C


if __name__ == "__main__":
    a = torch.ones((1, 64, 4, 4, 8))
    b = torch.ones((1, 64, 4, 4, 8))
    model = Geometry_information_Extractor(dim=64, num_head=32, prompt_length=10)
    y1, y2 = model(a, b)
    print(y1.shape)  # 全变0或全变2
    print(y2.shape)
