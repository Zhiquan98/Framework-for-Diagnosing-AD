import torch.nn as nn
import torch
from Model.TFE import Deformable_Attention_block


class Residual_block_2(nn.Module):  # 对应 18 层和 34 层 ResNet 的残差结构 residual
    expansion = 1  # 表示一个 residual 中最后的卷积核 out_channel 是前面的几倍。对于18，34层这个倍数是1
    # 用这个变量来标记 residual ，后面用以识别当前 residual 是两层卷积还是三层卷积

    def __init__(self, in_channel, out_channel, stride, downsample=None):
        """

        :param in_channel: residual 模块接收的特征图的 channel
        :param out_channel: residual 模块第一层卷积层的 输出 channel
        :param stride: residual 第一层卷积层的 步长。 对于第 2 3 4 个 Block 内的第一个 residual，一般设置为 2 用以下采样
        :param downsample: 下采样，在第 2 3 4 个 Block 的第一层 residual 用来下采样图片
        """

        super(Residual_block_2, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # residual 的第一个卷积
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)  # residual 的第二个卷积
        self.bn2 = nn.BatchNorm3d(out_channel)


    def forward(self, x):
        identity = x  # 把输入保存到跳连结构

        if self.downsample is not None:
            identity = self.downsample(x)  # 如果当前卷积层是block第一层，就要下采样

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Encoder(nn.Module):

    def __init__(self,
                 img_channel,
                 block,  # 根据不同的ResNet深度，传入上面定义的两个不同的 residual 中的一个
                 blocks_num,  # 所使用残差结构residual的数目。应该是一个列表参数【int1, int2, int3】 分别表示各个block有几个残差结构
                 blocks_channel,
                 ):
        super().__init__()

        self.in_channel = 64  # 获取每个 residual 结构接收到的深度。这里以经过7*7卷积后的深度为初始值

        self.conv1 = nn.Conv3d(img_channel, self.in_channel, kernel_size=7, stride=2,  # 第一个卷积层
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 开始用_make_layer函数生成四个block
        self.layer1 = self._make_layer(block, blocks_channel[0], blocks_num[0], stride=1)
        self.layer2 = self._make_layer(block, blocks_channel[1], blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, blocks_channel[2], blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, blocks_channel[3], blocks_num[3], stride=2)
        self.D_MSA = Deformable_Attention_block(Q_dim=512, KV_dim=256, n_heads=32, n_groups=8, attn_drop=0.1,
                                                proj_drop=0.1, path_drop=0.05, stride=2, offset_range_factor=4,
                                                kernel_size=2, mlp_drop=0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride):  # 默认stride=1，即默认当前是第一个block，要考虑3层residual的block1的特殊性
        """
        定义一个 layer ，首先要明确同一个 layer 内，不同的 residual 主线上是完全一样的
        不同点在于 跳连结构处。共有三种
        第一：直接跳连  （ 所有 非 第一个 residual  和  18， 34层 ResNet 的第一个 Layer 的第一个 residual）
        第二：调整 channel 为 4 倍并下采样 （ 所有第 2， 3， 4 个 Layer 的第一个 residual）
        第三：只调整 channel  ( 50， 101， 152 层 ResNet 的第一个 Layer 的第一个 residual )


        :param block: 传入两种残差结构 residual 之一
        :param channel: 第一个 residual 的卷积核个数
        :param block_num: 当前block包含了多少个残差结构
        :param stride: 第一个 residual 的第一层卷积层的 步长。 对于第 2 3 4 个 Block 内的第一个 residual，一般设置为 2 用以下采样
        :return:
        """

        """==========================获取第一个 residual 所采用的跳连结构==============================="""
        downsample = None  # 默认关闭下采样。即第一种跳连

        # 第二种跳连
        if stride == 2:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, channel * block.expansion, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm3d(channel * block.expansion)
            )

        # 第三种跳连，利用之前标记 residual 的类属性 expansion
        elif channel * block.expansion != self.in_channel:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channel, channel * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(channel * block.expansion)
            )
        """==========================获取第一个 residual 所采用的跳连结构==============================="""

        layers = []
        layers.append(block(self.in_channel,  # 添加第一个 residual。需要放入下采样跳连结构
                            channel,
                            downsample=downsample,
                            stride=stride))

        self.in_channel = channel * block.expansion  # 更新下一个residual所需要接收的数据深度

        for _ in range(1, block_num):  # 配置好第一个 residual 后，当前 block 剩下的 residual 都是实线直接跳连
            layers.append(block(self.in_channel,
                                channel,
                                downsample=None,
                                stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)

        feature4 = self.D_MSA(feature4, feature3)


        return feature4


class ResNet10(nn.Module):

    def __init__(self,
                 img_channel,
                 block,  # 根据不同的ResNet深度，传入上面定义的两个不同的 residual 中的一个
                 blocks_num,  # 所使用残差结构residual的数目。应该是一个列表参数【int1, int2, int3】 分别表示各个block有几个残差结构
                 num_classes,
                 blocks_channel):
        super().__init__()
        self.C_Encoder = ResNet_Encoder(img_channel, block, blocks_num , blocks_channel)
        self.Avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # output size = (1, 1)

        self.Classifier = nn.Linear(blocks_channel[-1] * block.expansion, num_classes)

    def forward(self, x):
        feature = self.C_Encoder(x)  # 4 4 8
        vetor = self.Avgpool(feature)
        final_vector = vetor.flatten(1)  # 【batch, channel】
        x = self.Classifier(final_vector)

        return x, final_vector


def resnet10(in_channels, num_classes, block_channels):
    return ResNet10(img_channel=in_channels, block=Residual_block_2, blocks_num=[1, 1, 1, 1], num_classes=num_classes, blocks_channel=block_channels)


if __name__ == "__main__":
    model = resnet10(in_channels=1, num_classes=2)
    x = torch.ones((1, 1, 128, 128, 256))
    y = model(x)
    print(y.shape)
