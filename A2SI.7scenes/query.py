from typing import List, Optional, Tuple
import sys
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from vision3d.layers import FourierEmbedding, TransformerLayer, AttentionLayer, ConfidenceAwareTransformerLayer

class CrossQueryFusionModule(nn.Module):
    def __init__(
        self,
        img_input_dim: int,
        pcd_input_dim: int,
        output_dim: int,
        query_hidden_dim: int = 256,  # ✅ 统一为 512
        num_heads: int = 8,
        blocks: List[str] = [],
        query_blocks: List[str] = [],
        dropout: Optional[float] = None,
        activation_fn: str = "ReLU",
        use_embedding: bool = True,
        embedding_dim: int = 10,
        num_queries: int = 12,
    ):
        super().__init__()
        self.num_queries = num_queries
        
        self.layernorm = nn.LayerNorm(256)

        self.query_embed = nn.Embedding(num_queries, query_hidden_dim)

        self.query_blocks = query_blocks
        query_layers = []
        layers = []
        for query_block in self.query_blocks:
            assert query_block in ["image", "point cloud"]
            query_layers.append(TransformerLayer(query_hidden_dim, num_heads, dropout=dropout, act_cfg=activation_fn))
            layers.append(TransformerLayer(query_hidden_dim, num_heads, dropout=dropout, act_cfg=activation_fn))

        query_layers.append(TransformerLayer(query_hidden_dim, num_heads, dropout=dropout, act_cfg=activation_fn))
        query_layers.append(TransformerLayer(query_hidden_dim, num_heads, dropout=dropout, act_cfg=activation_fn))
        layers.append(TransformerLayer(query_hidden_dim, num_heads, dropout=dropout, act_cfg=activation_fn))

        self.query_transformer = nn.ModuleList(query_layers)
        self.transformer = nn.ModuleList(layers)

        self.cross_attention1 = nn.ModuleList()
        self.cross_attention1.append(ConfidenceAwareTransformerLayer(query_hidden_dim, num_heads, dropout=dropout, act_cfg=activation_fn))  # 原版
        self.cross_attention1.append(ConfidenceAwareTransformerLayer(query_hidden_dim, num_heads, dropout=dropout, act_cfg=activation_fn))  # ✅ 加入 confidence 的版本


    def create_2d_embedding(self, pixels):
        embeddings = self.embedding(pixels)  # (1, HxW, L)
        embeddings = self.img_emb_proj(embeddings)  # (1, HxW, C)
        return embeddings

    def create_3d_embedding(self, points, masks):
        points_norm = torch.zeros_like(points).cuda()
        for i in range(points.shape[0]):
            points_norm[i] = points[i] - points[i][~masks[i]].mean(dim=0) 
        embeddings = self.embedding(points_norm)
        embeddings = self.pcd_emb_proj(embeddings)
        return embeddings

    def forward(
        self,
        img_feats: Tensor,
        pcd_feats: Tensor,
        pcd_masks: Optional[Tensor] = None,
        confidence: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Cross-Modal Feature Fusion Module.

        Args:
            img_feats (tensor): the features of the image in the shape of (B, HxW, Ci).
            img_pixels (tensor): the coordinates of the image in the shape of (B, HxW, 2).
            pcd_feats (tensor): the features of the point cloud in the shape of (B, N, Cp).
            pcd_points (tensor): the coordinates of the point cloud in the shape of (B, N, 3).
            img_masks (tensor, optional): the masks of the image in the shape of (B, H, W).
            pcd_masks (tensor, optional): the masks of the point cloud in the shape of (B, N).

        Returns:
            A tensor of the fused features of the image in the shape of (B, Co, H, W).
            A tensor of the fused features of the point cloud in the shape of (N, Co).
        """
        #img_tokens = self.img_in_proj(img_feats)  # (B, HxW, Ci) -> (B, HxW, C)
        #pcd_tokens = self.pcd_in_proj(pcd_feats)  # (B, N, Cp) -> (B, N, C)
        img_tokens = self.layernorm(img_feats)
        pcd_tokens = self.layernorm(pcd_feats)
        
        query_feats = self.query_embed.weight.unsqueeze(0).repeat(img_feats.shape[0], 1, 1)

        for i, query_block in enumerate(self.query_blocks):
            if query_block == "image":
                query_feats = self.transformer[i](query_feats, query_feats, query_feats)
                query_feats = self.query_transformer[i](query_feats, img_tokens, img_tokens)
            else:
                query_feats = self.transformer[i](query_feats, query_feats, query_feats)
                query_feats = self.query_transformer[i](query_feats, pcd_tokens, pcd_tokens, k_masks = pcd_masks)
                
        query_feats = self.transformer[-1](query_feats, query_feats, query_feats)

        query_img_feats = self.query_transformer[-2](query_feats, img_tokens, img_tokens)
        query_pcd_feats = self.query_transformer[-1](query_feats, pcd_tokens, pcd_tokens, k_masks = pcd_masks)

        return query_img_feats, query_pcd_feats, img_tokens, pcd_tokens