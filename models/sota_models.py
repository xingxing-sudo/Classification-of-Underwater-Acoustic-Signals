# sota_models.py - 自动生成
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel


class ASTModel(BaseModel):
    """Audio Spectrogram Transformer (AST) 模型，音频分类SOTA模型之一"""

    def __init__(self, num_classes=10, in_channels=1, img_size=128, patch_size=16, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., dropout=0.1):
        super(ASTModel, self).__init__(num_classes)

        # 图像/频谱图尺寸
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        # 计算patch数量
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        # 初始化pos_embed
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # 初始化cls_token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # 初始化conv和linear层
        self.apply(self._init_linear_conv_weights)

    def _init_linear_conv_weights(self, m):
        """初始化线性和卷积层权重"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播
        :param x: 输入频谱图 [batch, in_channels, height, width]
        :return: 分类结果 [batch, num_classes]
        """
        batch_size = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)  # [batch, embed_dim, h_patch, w_patch]
        x = x.flatten(2)  # [batch, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch, num_patches, embed_dim]

        # 添加Class Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch, num_patches+1, embed_dim]

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer编码
        x = self.encoder(x)

        # 分类
        x = self.norm(x)
        x = x[:, 0]  # 取class token的输出
        x = self.head(x)  # [batch, num_classes]

        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder层"""

    def __init__(self, embed_dim, depth, num_heads, mlp_ratio, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block（包含Multi-Head Attention和MLP）"""

    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super(TransformerBlock, self).__init__()
        # Self-Attention层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # MLP层
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-Attention残差连接
        x_attn, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout1(x_attn)

        # MLP残差连接
        x_mlp = self.mlp(self.norm2(x))
        x = x + x_mlp

        return x