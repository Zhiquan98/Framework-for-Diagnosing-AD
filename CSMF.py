from Model.ResNet_with_TFE import resnet10
from Model.Unet_3D import UNet
from Model.AGIE import Geometry_information_Extractor
import torch.nn as nn
import torch
import torch.nn.functional as F


class CSMF(nn.Module):
    def __init__(self, in_channels, block_channels, num_classes, base_c):
        super(CSMF, self).__init__()
        self.seg_model = UNet(in_channel=in_channels, num_class=num_classes, base_c=base_c)
        self.ResNet10 = resnet10(in_channels=in_channels, num_classes=num_classes, block_channels=block_channels)
        self.Cross_Atten = Geometry_information_Extractor(dim=512, num_head=32, prompt_length=10)
        self.Attention_pool = nn.Linear(512, 1)

    def forward(self, Unet_input, Resnet_input):
        seg_final_feature_map, x5, x4, x3, x2, x1 = self.seg_model.encoder(Unet_input)
        C_final_feature_map = self.ResNet10.C_Encoder(Resnet_input)

        """两个特征图交换特征，得到 [B, DHW, C]"""
        feature_from_seg, feature_from_C = self.Cross_Atten(seg_final_feature_map, C_final_feature_map)


        """ 从 C 中获取到的特征值，重塑回 seg 特征图的 shape 再加回去"""
        # [B, DHW, C] -> [B, C, DHW] -> [B, C, D, H, W]
        feature_from_C = feature_from_C.transpose(-1, -2).reshape(seg_final_feature_map.shape)
        pre_Mask = self.seg_model.decoder(seg_final_feature_map + feature_from_C, x5, x4, x3, x2, x1)


        """ 从 seg 中获取到的特征值，利用 Sequential Pooling 从 [batch, num_token, dim] 变成 [batch, dim]，加到分类特征向量"""
        # Attention_pool: [B, DHW, C] -> [B, DHW, 1]  每个 token 获得自己的 softmax 权重
        Attn = F.softmax(self.Attention_pool(feature_from_seg), dim=1)
        sum = torch.einsum('b n c, b n e -> b c e', feature_from_seg, Attn).squeeze(-1)  # [batch, 512]
        C_feature_Vector = torch.flatten(self.ResNet10.Avgpool(C_final_feature_map), 1)   # [batch, 512, 4, 4, 8] → [batch, 512]
        final_C_Vector = C_feature_Vector + sum
        C_predict = self.ResNet10.Classifier(final_C_Vector)


        return pre_Mask, C_predict, final_C_Vector


if __name__ == "__main__":
    x = torch.ones((1, 1, 128, 128, 256)).to("cuda:0")
    model = CSMF(in_channels=1, num_classes=2, block_channels=(64, 128, 256, 512), base_c=32).to("cuda:0")
    pre_Mask, C_predict = model(x)
    print(pre_Mask.shape,  C_predict.shape)  # 全变0或全变2
