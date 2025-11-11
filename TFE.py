import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LayerNorm_C(nn.Module):

    def __init__(self, dim):
        """
        输入 [batch, channel , D, H, W]，将 channel 方向进行 标准化
        :param dim:
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b d h w c -> b c d h w')


class Deformable_Attention_3D(nn.Module):

    def __init__(
            self, Q_dim, KV_dim, n_heads, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, kernel_size
    ):
        """

        :param dim:
        :param n_heads:
        :param n_groups:
        :param attn_drop:
        :param proj_drop:
        :param stride:
        :param offset_range_factor:
        :param kernel_size: 每个像素点偏移量的生成感受野
        :param no_off:
        """
        super().__init__()

        self.Q_dim = Q_dim
        self.KV_dim = KV_dim

        self.n_heads = n_heads
        self.n_head_channels = KV_dim // n_heads
        assert KV_dim % n_heads == 0, "KV的 channel 无法被 头数 整除"
        self.scale = self.n_head_channels ** -0.5

        self.n_groups = n_groups
        self.n_group_channels = KV_dim // n_groups
        assert KV_dim % n_groups == 0, "KV的 channel 无法被 偏移量组数 整除"
        self.offset_range_factor = offset_range_factor


        self.conv_offset = nn.Sequential(
            # 这个卷积会将特征图下采样一些，即多个像素得到一个偏移量
            nn.Conv3d(self.n_group_channels, self.n_group_channels, kernel_size, stride, kernel_size // 2, groups=self.n_group_channels),
            LayerNorm_C(self.n_group_channels),
            nn.GELU(),
            # 这个卷积就是把 channel 变为 3, 因为每个像素点有 x y z 三个坐标
            nn.Conv3d(self.n_group_channels, 3, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv3d(self.Q_dim, self.KV_dim, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv3d(self.KV_dim, self.KV_dim, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv3d(self.KV_dim, self.KV_dim, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv3d(self.KV_dim, self.Q_dim, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)



    @torch.no_grad()
    def _get_ref_points(self, Dw, Hw, Ww, B, dtype, device):
        # 生成坐标系，坐标轴的取值范围是 特征图 （0.5，Hw-0.5）。
        # 如果宽高是 3 ，则 坐标轴为 [0.5, 1.5, 2.5]
        # ref_xyz.shape = [WHD]
        ref_z, ref_y, ref_x = torch.meshgrid(  # 第一个返回值对应第一个坐标轴
            torch.linspace(0.5, Dw - 0.5, Dw, dtype=dtype, device=device),
            torch.linspace(0.5, Hw - 0.5, Hw, dtype=dtype, device=device),
            torch.linspace(0.5, Ww - 0.5, Ww, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_z, ref_y, ref_x), -1)  # shape = [Dw, Hw, Ww， 3]
        # ref[..., 0] 索引对应第一个坐标轴
        # 各个轴坐标 除以宽高（缩放到【0， 1】），然后乘以2减1（缩放到 [-1, 1]）

        ref[..., 0].div_(Dw).mul_(2).sub_(1)
        ref[..., 1].div_(Hw).mul_(2).sub_(1)
        ref[..., 2].div_(Ww).mul_(2).sub_(1)


        # ref[None, ...] 表示在最前面插入一个维度 = [1, ， 2]
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1, -1)  # [batch*n_group，Hk, Wk, Dk， 3]

        return ref

    def forward(self, Q_feature, KV_feature):

        B, Q_C, Q_D, Q_H, Q_W = Q_feature.size()
        B, KV_C, KV_D, KV_H, KV_W = KV_feature.size()
        dtype, device = Q_feature.dtype, Q_feature.device

        Q = self.proj_q(Q_feature)  # 把特征图每个像素的特征向量都映射一下 [B, C, D, H, W] -> [B, C, D, H, W]

        # 分组求偏移量，DC中每个像素求得偏移后， channel 维度所有信息都按照同一个偏移。现学习 n_groups 数来把 channel 均分，每一组都学习自己的偏移
        q_off = einops.rearrange(Q, 'b (g c) d h w -> (b g) c d h w', g=self.n_groups, c=self.n_group_channels)

        offset = self.conv_offset(q_off)
        # [B*n_group, c, d, h, w]  ->  [B*n_group, 3, d/ks, h/ks, w/ks]
        # 与 DC 时相比，因为DC是卷积核采集点的偏移，所以 offset 中的 channel 是 KS*KS*KS 用以表示一个卷积核所有元素的偏移
        # 现在因为只需对 KV 特征图进行偏移，所以只需要知道像素点本身的偏移，所以 channel 是 3 即可让每个像素都偏移
        # 对于 尺寸，完全取决于 stride，尺寸会下降 stride 倍数

        Dk, Hk, Wk = offset.size(2), offset.size(3), offset.size(4)  # 获取偏移坐标图的宽高
        n_sample = Dk * Hk * Wk  # 获取总采样点个数

        if self.offset_range_factor > 0:  # 源码中，这里取1， 2， 3， 4
            # 指的是偏移倍数。在低级特征图中，一般偏移不会太大
            # 在高级特征图中，偏移可以很大很大
            # 求标准坐标轴中，两个轴各自 相邻两个坐标点之间的 差值
            offset_range = torch.tensor([1.0/ Dk, 1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 3, 1, 1, 1)  # [1, 3, 1, 1, 1]
            # tanh() == 把偏移量约束在 [-1, 1]。 因为后续用到的标准坐标轴左上角为（-1， -1），右下角为（1， 1）
            # mul(self.offset_range_factor) == 坐标轴放大 factor 倍，左上角为（-factor, -factor），右下角为（factor, factor）
            # mul(offset_range) == 除以特征图尺寸 Hk， 左上角为（-factor/Hk, -factor/Wk），右下角为（factor/Hk, factor/Wk）
            """
            关于为什么要除以特征图的尺寸
            因为偏移量是看作每一个像素点的偏移量
            假设某个像素点学习到的偏移量是（1， 1），并不是意味着这个像素点后续的取值就是原图的右下角
            而是以当前像素点为原点，其相邻元素的右下角。
            所以当某个像素点学习到的偏移量为(1,1)，除以尺寸后就变成了(像素距离，像素距离)
            即 offset_range 是限制了每一个像素最多便宜到相邻多少个像素
            这个值加上标准坐标系上的坐标，就得到了其右下角一个像素距离 的像素
            所以如果没有不对偏移量进行放大，则每个像素点的最大偏移也就只能取到相邻的一个像素
            放大倍数是多少，就代表每个像素最大能偏移到相邻的几个像素
            """

            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p d h w -> b d h w p')  # [B*n_group, d, h, w, 3]把 3 放到最后，方便调取 X，Y， Z 轴
        reference = self._get_ref_points(Dk, Hk, Wk, B, dtype, device)  # 标准坐标轴，左上角为（-1， -1），右下角为（1， 1）
        # 获得 [batch*n_group, Dk, Hk, Wk, 2]。 和 offset 同一个 shape
        # 其中宽高坐标轴缩放在了 [-1, 1]


        if self.offset_range_factor >= 0:
            pos = offset + reference  # 在标准坐标系中偏移
        else:
            pos = (offset + reference).tanh()

        # 利用坐标 [batch*n_group, Dk , Hk , Wk， 3] 对特征图 [batch, channel, D, H, W] 进行分组采样采
        # 得到新的特征图 [batch*n_group, group_channel, Dk , Hk , Wk]
        """
        F.grid_sample(input, grid)
        input = [B, channel, Hin, Win]
        grid = [B, Hout, Wout, 2]
        对于 input 会自动生成坐标系，左上角为（-1， -1），右下角为（1， 1）
        grid 就是一个坐标位置表格，每一个格子内的二维坐标，都代表了对应输出格子应该去原图的哪里进行采样
        所以 grid 的元素必须都在 [-1, 1]
        例如 grid 左上角的元素为(1, 1)，就代表了 输出图像左上角的 元素来自原图右下角
        如果坐标位置不恰好落在原图像素，则采用插值求得
        """
        x_sampled = F.grid_sample(  # shape '[3, 2, 20, 20, 20]' is invalid for input of size 24000
            input=KV_feature.reshape(B * self.n_groups, self.n_group_channels, KV_D, KV_H, KV_W),
            grid=pos[..., (2, 1, 0)],  # xyz轴的顺序，在 tensor 中对应的是 WHD, 所以这里要稍微调换一下
            mode='bilinear', align_corners=True)  # [batch*n_group, group_channel, Dk , Hk , Wk]

        # 把分组学习的 偏移特征图 还原回[batch, channel, Dk, Hk，Wk]
        x_sampled = x_sampled.reshape(B, KV_C, Dk, Hk, Wk)  # [batch, channel, 1, DHW]

        # 多头注意力，把Q分成多个头 [batch, channel, D, H, W] →→ [B*n_head, head_channel, D*H*W]
        Q = Q.reshape(B * self.n_heads, self.n_head_channels, Q_D*Q_H*Q_W)

        # 特征图偏移后映射成 KV ，并分成多个头 [B*n_head, head_channel, Hk*Wk]
        K = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        V = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        # SA注意力公式 [batch, channel, HW]@[batch, channel, Hk*Wk]   ->  [batch, HW, Hk*Wk]
        attn = torch.einsum('b c m, b c n -> b m n', Q, K)  # B * h, HWk, Ns
        attn = attn.mul(self.scale)

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        # 计算得到SA输出
        out = torch.einsum('b m n, b c n -> b c m', attn, V)
        out = out.reshape(B, KV_C, Q_D, Q_H, Q_W)  # Q_D, Q_H, Q_W

        y = self.proj_drop(self.proj_out(out))

        return y


class Deformable_Attention_block(nn.Module):
    def __init__(self, Q_dim, KV_dim, n_heads, n_groups, attn_drop, proj_drop, path_drop, stride, offset_range_factor, kernel_size, mlp_drop):
        """"""
        super().__init__()
        self.Q_LN1 = LayerNorm_C(Q_dim)
        self.KV_LN1 = LayerNorm_C(KV_dim)
        self.De_MSA = Deformable_Attention_3D(Q_dim=Q_dim, KV_dim=KV_dim, n_heads=n_heads, n_groups=n_groups,
                                              attn_drop=attn_drop, proj_drop=proj_drop, stride=stride,
                                              offset_range_factor=offset_range_factor, kernel_size=kernel_size)
        self.Droppath1 = DropPath(path_drop)

        self.LN2 = LayerNorm_C(Q_dim)
        self.MLP = TransformerMLP_3D(channels=Q_dim, expansion=4, drop=mlp_drop)
        self.Droppath2 = DropPath(path_drop)

        """这里删了相对位置 bias"""

    def forward(self, Q, KV):
        x = self.Droppath1(self.De_MSA(self.Q_LN1(Q), self.KV_LN1(KV))) + Q

        x = self.Droppath2(self.MLP(self.LN2(x))) + x

        return x


class TransformerMLP_3D(nn.Module):

    def __init__(self, channels, expansion, drop):
        """
        输入 [batch, channel , D, H, W]，将 channel 方向进行 全连接-激活-DP-全连接-DP 操作
        本质上就是对 channel 维度进行全连接编码

        :param channels:
        :param expansion:
        :param drop:
        """
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop))

    def forward(self, x):
        _, _, D, H, W = x.size()
        x = einops.rearrange(x, 'b c d h w -> b (d h w) c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b (d h w) c -> b c d h w', h=H, w=W)
        return x


if __name__ == "__main__":
    x = torch.ones((1, 12, 10, 10, 10))
    b = torch.ones((1, 6, 20, 20, 20))
    model = Deformable_Attention_block(Q_dim=12, KV_dim=6, n_heads=2, n_groups=3, attn_drop=0., proj_drop=0., stride=2,
                                       offset_range_factor=1, kernel_size=2, mlp_drop=0.1, path_drop=0.1)
    y = model(x, b)
    print(y.shape)