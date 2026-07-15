from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from vision3d.layers import FourierEmbedding

from typing import Optional, Tuple, Union
from query import CrossQueryFusionModule
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from vision3d.ops import pairwise_cosine_similarity
from basic_layers import build_act_layer, build_dropout_layer

class CrossModalFusionModule(nn.Module):
    def __init__(
        self,
        cfg,
        img_input_dim: int,
        pcd_input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_heads: int,
        blocks: List[str],
        dropout: Optional[float] = None,
        activation_fn: str = "ReLU",
        use_embedding: bool = True,
        embedding_dim: int = 10,
        query_agg_fn: Optional[callable] = None  # ✅ 添加这行
    ):
        super().__init__()

        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embedding = FourierEmbedding(embedding_dim, use_pi=False, use_input=True)
            self.img_emb_proj = nn.Linear(embedding_dim * 4 + 2, hidden_dim)
            self.pcd_emb_proj = nn.Linear(embedding_dim * 6 + 3, hidden_dim)
        else:
            self.embedding = None
            self.img_emb_proj = None
            self.pcd_emb_proj = None

        self.img_in_proj = nn.Linear(img_input_dim, hidden_dim)
        self.pcd_in_proj = nn.Linear(pcd_input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.query_agg_fn = query_agg_fn or self.compute_query_attention_aggregate  # ✅ 正确赋值
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            assert block in ["self", "cross"]
            layers.append(TransformerLayer(hidden_dim, num_heads, dropout=dropout, act_cfg=activation_fn, query_agg_fn=self.query_agg_fn))
        self.transformer = nn.ModuleList(layers)
        self.queryformer = CrossQueryFusionModule(
            cfg.model.transformer.img_input_dim,
            cfg.model.transformer.pcd_input_dim,
            cfg.model.transformer.output_dim,
            cfg.model.transformer.hidden_dim,
            cfg.model.transformer.num_heads,
            cfg.model.transformer.blocks,
            cfg.model.transformer.query_blocks,
            use_embedding=cfg.model.transformer.use_embedding,
            num_queries=cfg.model.transformer.num_queries,
        )
        self.query_scores = nn.Parameter(torch.zeros(cfg.model.transformer.num_queries))

        


    def create_2d_embedding(self, pixels):
        embeddings = self.embedding(pixels)  # (1, HxW, L)
        embeddings = self.img_emb_proj(embeddings)  # (1, HxW, C)
        return embeddings

    def create_3d_embedding(self, points):
        points = points - points.mean(dim=1)
        embeddings = self.embedding(points)
        embeddings = self.pcd_emb_proj(embeddings)
        return embeddings
    
    def compute_selected_tokens(
        self,
        img_tokens: Tensor,
        pcd_tokens: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        query_img_feats, query_pcd_feats, _, _ = self.queryformer(img_tokens, pcd_tokens)

        img_sim_mat = pairwise_cosine_similarity(query_img_feats, img_tokens, normalized=True)
        img_topk_indices = torch.topk(img_sim_mat, k=1, dim=-1).indices.squeeze(0).squeeze(-1)
        img_selected_tokens = img_tokens.squeeze(0)[img_topk_indices]

        pcd_sim_mat = pairwise_cosine_similarity(query_pcd_feats, pcd_tokens, normalized=True)
        pcd_topk_indices = torch.topk(pcd_sim_mat, k=1, dim=-1).indices.squeeze(0).squeeze(-1)
        pcd_selected_tokens = pcd_tokens.squeeze(0)[pcd_topk_indices]

        return (
            img_selected_tokens.unsqueeze(0),
            pcd_selected_tokens.unsqueeze(0),
            img_topk_indices,
            pcd_topk_indices,
        )


    def update_query_scores_with_reinforce(self, reward, actions, baseline=0.0):
        probs = torch.sigmoid(self.query_scores)  # [Q]
        log_probs = actions.float() * torch.log(probs + 1e-6) + \
                    (1.0 - actions.float()) * torch.log(1 - probs + 1e-6)

        if baseline is None:
            baseline = reward.mean().detach()

        # REINFORCE 主损失
        reinforce_loss = - ((reward - baseline) * log_probs).sum()

        # 🔹 添加熵奖励 Entropy Bonus
        entropy = - (probs * torch.log(probs + 1e-6) + (1 - probs) * torch.log(1 - probs + 1e-6))
        entropy_loss = -0.01 * entropy.sum()  # 负号：鼓励更多熵，越大越好

        # 总 loss = policy loss + 熵奖励
        total_loss = reinforce_loss + entropy_loss
        total_loss.backward()



    def compute_query_attention_aggregate(
            self,
            img_selected_tokens: Tensor,
            pcd_selected_tokens: Tensor,
            img_tokens: Tensor,
            pcd_tokens: Tensor,) -> Tensor:
            # normalize
        img_tokens = F.normalize(img_tokens, dim=-1)        # [B, NI, C]
        pcd_tokens = F.normalize(pcd_tokens, dim=-1)        # [B, NP, C]
        img_selected = F.normalize(img_selected_tokens, dim=-1)  # [B, Q, C]
        pcd_selected = F.normalize(pcd_selected_tokens, dim=-1)  # [B, Q, C]

        # Step 1: img → query attn, [B, NI, Q]
        img_to_query = torch.matmul(img_tokens, img_selected.transpose(1, 2))  # (B, NI, Q)
        img_to_query = F.softmax(img_to_query, dim=-1)

        # Step 2: query → pcd attn, [B, Q, NP]
        query_to_pcd = torch.matmul(pcd_selected, pcd_tokens.transpose(1, 2))  # (B, Q, NP)
        query_to_pcd = F.softmax(query_to_pcd, dim=-1)

        # Step 3: fused attention map [B, NI, NP]
        fused_map = torch.matmul(img_to_query, query_to_pcd)  # [B, NI, NP]

        return fused_map


    def forward(
        self,
        img_feats: Tensor,
        img_pixels: Tensor,
        pcd_feats: Tensor,
        pcd_points: Tensor,
        img_masks: Optional[Tensor] = None,
        pcd_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, dict]:
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
        img_tokens = self.img_in_proj(img_feats)  # (B, HxW, Ci) -> (B, HxW, C)
        pcd_tokens = self.pcd_in_proj(pcd_feats)  # (B, N, Cp) -> (B, N, C)

        probs = torch.sigmoid(self.query_scores)

        if self.training and getattr(self, "rl_mode", False):
            actions = torch.bernoulli(probs).bool()
            self.last_query_actions = actions.detach()
        else:
            topk = 12
            _, topk_indices = torch.topk(probs, k=topk, dim=0)
            actions = torch.zeros_like(probs, dtype=torch.bool)
            actions[topk_indices] = True

        # Mask query feats（Soft mask 建议）
        alpha = 0.3
        soft_mask = alpha + (1 - alpha) * actions.float()  # [Q]
        self.last_query_mask = soft_mask.detach()          # 用于外部计算 reward

        query_img_feats, query_pcd_feats, _, _ = self.queryformer(img_tokens, pcd_tokens)
        query_img_feats = query_img_feats.squeeze(0) * soft_mask.unsqueeze(-1)
        query_pcd_feats = query_pcd_feats.squeeze(0) * soft_mask.unsqueeze(-1)

        img_selected_tokens, pcd_selected_tokens, img_topk_indices, pcd_topk_indices = self.compute_selected_tokens(query_img_feats.unsqueeze(0), query_pcd_feats.unsqueeze(0))


        
        if self.use_embedding:
            img_embeddings = self.create_2d_embedding(img_pixels)  # (B, HxW, C)
            img_tokens = img_tokens + img_embeddings  # (B, HxW, C)
            pcd_embeddings = self.create_3d_embedding(pcd_points)  # (B, N, C)
            pcd_tokens = pcd_tokens + pcd_embeddings  # (B, N, C)

        for i, block in enumerate(self.blocks):
            if block == "self":
                img_tokens, *_ = self.transformer[i](img_tokens, img_tokens, img_tokens, img_selected_tokens, pcd_selected_tokens, k_masks=img_masks)
                pcd_tokens, *_ = self.transformer[i](pcd_tokens, pcd_tokens, pcd_tokens, img_selected_tokens, pcd_selected_tokens, k_masks=pcd_masks)
            else:
                img_tokens, *_ = self.transformer[i](img_tokens, pcd_tokens, pcd_tokens, img_selected_tokens, pcd_selected_tokens, k_masks=pcd_masks)
                pcd_tokens, *_ = self.transformer[i](pcd_tokens, img_tokens, img_tokens, img_selected_tokens, pcd_selected_tokens, k_masks=img_masks)

        img_feats = self.out_proj(img_tokens)  # (B, HxW, C)
        pcd_feats = self.out_proj(pcd_tokens)  # (B, N, C)

        # ✅ 计算 query → token 的 top1 cosine 相似度，作为强化学习 reward 向量
        with torch.no_grad():
            img_token = img_tokens.squeeze(0)
            pcd_token = pcd_tokens.squeeze(0)
            img_top1 = img_token[img_topk_indices]
            pcd_top1 = pcd_token[pcd_topk_indices]
            sim_img = F.cosine_similarity(query_img_feats, img_top1, dim=-1)
            sim_pcd = F.cosine_similarity(query_pcd_feats, pcd_top1, dim=-1)
            query_rewards = (sim_img + sim_pcd) / 2



        return img_feats, pcd_feats, {
            "query_img_feats": query_img_feats,
            "query_pcd_feats": query_pcd_feats,
            "img_selected_tokens": img_selected_tokens,
            "pcd_selected_tokens": pcd_selected_tokens,
            "query_mask": soft_mask.detach(),
            "query_actions": actions.detach(),
            "img_token": img_tokens.detach(),
            "pcd_token": pcd_tokens.detach(),
            "query_scores": self.query_scores,
            "query_rewards": query_rewards, 
            "img_topk_indices": img_topk_indices.detach(),
            "pcd_topk_indices": pcd_topk_indices.detach(),
        }


        





class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_embed_proj: bool = False,
        k_embed_proj: bool = False,
        v_embed_proj: bool = False,
        qk_embed_proj: bool = False,
        qv_embed_proj: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0, f"'d_model={d_model}' is not divisible by 'num_heads={num_heads}'."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.q_token_layer = nn.Linear(self.d_model, self.d_model)
        self.k_token_layer = nn.Linear(self.d_model, self.d_model)
        self.v_token_layer = nn.Linear(self.d_model, self.d_model)

        self.has_q_embed_proj = q_embed_proj
        if self.has_q_embed_proj:
            self.q_embed_layer = nn.Linear(self.d_model, self.d_model)

        self.has_k_embed_proj = k_embed_proj
        if self.has_k_embed_proj:
            self.k_embed_layer = nn.Linear(self.d_model, self.d_model)

        self.has_v_embed_proj = v_embed_proj
        if self.has_v_embed_proj:
            self.v_embed_layer = nn.Linear(self.d_model, self.d_model)

        self.has_qk_embed_proj = qk_embed_proj
        if self.has_qk_embed_proj:
            self.qk_embed_layer = nn.Linear(self.d_model, self.d_model)

        self.has_qv_embed_proj = qv_embed_proj
        if self.has_qv_embed_proj:
            self.qv_embed_layer = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    def forward(
        self,
        q_tokens: Tensor,
        k_tokens: Tensor,
        v_tokens: Tensor,
        q_embeds: Optional[Tensor] = None,
        k_embeds: Optional[Tensor] = None,
        v_embeds: Optional[Tensor] = None,
        img_selected_tokens: Optional[Tensor] = None,     # ✅ 新加
        pcd_selected_tokens: Optional[Tensor] = None,     # ✅ 新加
        qk_embeds: Optional[Tensor] = None,
        qv_embeds: Optional[Tensor] = None,
        k_weights: Optional[Tensor] = None,
        k_masks: Optional[Tensor] = None,
        qk_weights: Optional[Tensor] = None,
        qk_masks: Optional[Tensor] = None,
        override_attention: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Multi-Head Attention forward propagation.

        Args:
            q_tokens (Tensor): query tokens (B, N, C)
            k_tokens (Tensor): key tokens (B, M, C)
            v_tokens (Tensor): value tokens (B, M, C)
            q_embeds (Tensor): query embeddings (B, N, C)
            k_embeds (Tensor): key embeddings (B, M, C)
            v_embeds (Tensor): value embeddings (B, M, C)
            qk_embeds (Tensor): query-key embeddings (B, N, M, C)
            qv_embeds (Tensor): query-value embeddings (B, N, M, C)
            k_weights (Tensor): key weights (B, M)
            k_masks (BoolTensor): key masks. If True, ignored. (B, M)
            qk_weights (Tensor): query-key weights (B, N, M)
            qk_masks (BoolTensor): query-key masks. If True, ignored. (B, N, M)

        Returns:
            hidden_tokens (Tensor): output tokens (B, C, N)
            attention_scores (Tensor): attention scores (after dropout) (B, H, N, M)
        """
        if override_attention is not None:
    # 替换为外部提供的 attention map：形状 [B, NI, NP]
    # 要扩展为多头：[B, H, NI, NP]
            attention_scores = override_attention.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        


        # input check
        if self.has_q_embed_proj:
            assert q_embeds is not None, "No 'q_embeds' but 'q_embed_proj' is set."
        if self.has_k_embed_proj:
            assert k_embeds is not None, "No 'k_embeds' but 'k_embed_proj' is set."
        if self.has_v_embed_proj:
            assert v_embeds is not None, "No 'v_embeds' but 'v_embed_proj' is set."
        if self.has_qk_embed_proj:
            assert qk_embeds is not None, "No 'qk_embeds' but 'qk_embed_proj' is set."
        if self.has_qv_embed_proj:
            assert qv_embeds is not None, "No 'qv_embeds' but 'qv_embed_proj' is set."

        # compute query and key tokens
        q_tokens = self.q_token_layer(q_tokens)
        if q_embeds is not None:
            if self.has_q_embed_proj:
                q_embeds = self.q_embed_layer(q_embeds)
            q_tokens = q_tokens + q_embeds
        q_tokens = rearrange(q_tokens, "b n (h c) -> b h n c", h=self.num_heads)

        k_tokens = self.k_token_layer(k_tokens)
        if k_embeds is not None:
            if self.has_k_embed_proj:
                k_embeds = self.k_embed_layer(k_embeds)
            k_tokens = k_tokens + k_embeds
        k_tokens = rearrange(k_tokens, "b m (h c) -> b h m c", h=self.num_heads)

        # compute attention scores
        if qk_embeds is not None:
            if self.has_qk_embed_proj:
                qk_embeds = self.qk_embed_layer(qk_embeds)
            qk_embeds = rearrange(qk_embeds, "b n m (h c) -> b h n m c", h=self.num_heads)
            attention_scores = torch.einsum("bhnc,bhnmc->bhnm", q_tokens, k_tokens.unsqueeze(2) + qk_embeds)
        else:
            attention_scores = torch.einsum("bhnc,bhmc->bhnm", q_tokens, k_tokens)
        attention_scores = attention_scores / self.d_model_per_head ** 0.5
        if qk_weights is not None:
            attention_scores = attention_scores * qk_weights.unsqueeze(1)
        if k_weights is not None:
            attention_scores = attention_scores * k_weights.unsqueeze(1).unsqueeze(1)
        if k_masks is not None:
            attention_scores = attention_scores.masked_fill(k_masks.unsqueeze(1).unsqueeze(1), float("-inf"))
        if qk_masks is not None:
            attention_scores = attention_scores.masked_fill(qk_masks.unsqueeze(1), float("-1e5"))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        # compute output tokens
        v_tokens = self.v_token_layer(v_tokens)
        if v_embeds is not None:
            if self.has_v_embed_proj:
                v_embeds = self.v_embed_layer(v_embeds)
            v_tokens = v_tokens + v_embeds
        v_tokens = rearrange(v_tokens, "b m (h c) -> b h m c", h=self.num_heads)

        if qv_embeds is not None:
            if self.has_qv_embed_proj:
                qv_embeds = self.qv_embed_layer(qv_embeds)
            qv_embeds = rearrange(qv_embeds, "b n m (h c) -> b h n m c", h=self.num_heads)
            hidden_tokens = torch.einsum("bhnm,bhnmc->bhnc", attention_scores, v_tokens.unsqueeze(2) + qv_embeds)
        else:
            hidden_tokens = torch.matmul(attention_scores, v_tokens)

        hidden_tokens = rearrange(hidden_tokens, "b h n c -> b n (h c)")

        return hidden_tokens, attention_scores


class AttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_embed_proj: bool = False,
        k_embed_proj: bool = False,
        v_embed_proj: bool = False,
        qk_embed_proj: bool = False,
        qv_embed_proj: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model,
            num_heads,
            q_embed_proj=q_embed_proj,
            k_embed_proj=k_embed_proj,
            v_embed_proj=v_embed_proj,
            qk_embed_proj=qk_embed_proj,
            qv_embed_proj=qv_embed_proj,
            dropout=dropout,
        )
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        q_tokens: Tensor,
        k_tokens: Tensor,
        v_tokens: Tensor,
        q_embeds: Optional[Tensor] = None,
        k_embeds: Optional[Tensor] = None,
        v_embeds: Optional[Tensor] = None,
        img_selected_tokens: Optional[Tensor] = None,     # ✅ 新加
        pcd_selected_tokens: Optional[Tensor] = None,     # ✅ 新加
        qk_embeds: Optional[Tensor] = None,
        qv_embeds: Optional[Tensor] = None,
        k_weights: Optional[Tensor] = None,
        k_masks: Optional[Tensor] = None,
        qk_weights: Optional[Tensor] = None,
        qk_masks: Optional[Tensor] = None,
        override_attention: Optional[Tensor] = None, 
    ) -> Tuple[Tensor, Tensor]:
        hidden_tokens, attention_scores = self.attention(
            q_tokens,
            k_tokens,
            v_tokens,
            q_embeds=q_embeds,
            k_embeds=k_embeds,
            v_embeds=v_embeds,
            img_selected_tokens=img_selected_tokens, 
            pcd_selected_tokens=pcd_selected_tokens,
            qk_embeds=qk_embeds,
            qv_embeds=qv_embeds,
            k_weights=k_weights,
            k_masks=k_masks,
            qk_weights=qk_weights,
            qk_masks=qk_masks,
            override_attention=override_attention,
        )
        hidden_tokens = self.linear(hidden_tokens)
        hidden_tokens = self.dropout(hidden_tokens)
        output_tokens = self.norm(hidden_tokens + q_tokens)
        return output_tokens, attention_scores


class AttentionOutput(nn.Module):
    def __init__(self, d_model: int, dropout: Optional[float] = None, act_cfg: Union[str, dict] = "ReLU"):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * 2)
        self.activation = build_act_layer(act_cfg)
        self.squeeze = nn.Linear(d_model * 2, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_tokens: Tensor) -> Tensor:
        hidden_tokens = self.expand(input_tokens)
        hidden_tokens = self.activation(hidden_tokens)
        hidden_tokens = self.squeeze(hidden_tokens)
        hidden_tokens = self.dropout(hidden_tokens)
        output_tokens = self.norm(input_tokens + hidden_tokens)
        return output_tokens


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_embed_proj: bool = False,
        k_embed_proj: bool = False,
        v_embed_proj: bool = False,
        qk_embed_proj: bool = False,
        qv_embed_proj: bool = False,
        dropout: Optional[float] = None,
        act_cfg: Union[str, dict] = "ReLU",
        query_agg_fn: Optional[callable] = None,  # ✅ 新增：绑定外部 attention 聚合函数
    ):
        super().__init__()
        self.attention = AttentionLayer(
            d_model,
            num_heads,
            q_embed_proj=q_embed_proj,
            k_embed_proj=k_embed_proj,
            v_embed_proj=v_embed_proj,
            qk_embed_proj=qk_embed_proj,
            qv_embed_proj=qv_embed_proj,
            dropout=dropout,
        )
        self.output = AttentionOutput(d_model, dropout=dropout, act_cfg=act_cfg)
        self.query_agg_fn = query_agg_fn  # ✅ 存储函数引用

    def forward(
        self,
        q_tokens: Tensor,
        k_tokens: Tensor,
        v_tokens: Tensor,
        img_selected_tokens: Tensor,
        pcd_selected_tokens: Tensor,
        q_embeds: Optional[Tensor] = None,
        k_embeds: Optional[Tensor] = None,
        v_embeds: Optional[Tensor] = None,
        qk_embeds: Optional[Tensor] = None,
        qv_embeds: Optional[Tensor] = None,
        k_weights: Optional[Tensor] = None,
        k_masks: Optional[Tensor] = None,
        qk_weights: Optional[Tensor] = None,
        qk_masks: Optional[Tensor] = None,
        return_attention_score: bool = False,
        use_query_agg: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:

        # ✅ 若启用 query attention 聚合，则先生成替代 attention map
        query_agg_attn = None
        if use_query_agg:
            assert self.query_agg_fn is not None, "Missing query attention aggregation function."
            query_agg_attn = self.query_agg_fn(
                img_selected_tokens=img_selected_tokens,
                pcd_selected_tokens=pcd_selected_tokens,
                img_tokens=q_tokens,
                pcd_tokens=k_tokens,
            )

        # ✅ 主 attention 调用，注意：无论是否聚合，都执行正常 attention 层
        hidden_tokens, attention_scores = self.attention(
            q_tokens,
            k_tokens,
            v_tokens,
            q_embeds=q_embeds,
            k_embeds=k_embeds,
            v_embeds=v_embeds,
            img_selected_tokens=img_selected_tokens,
            pcd_selected_tokens=pcd_selected_tokens,
            qk_embeds=qk_embeds,
            qv_embeds=qv_embeds,
            k_weights=k_weights,
            k_masks=k_masks,
            qk_weights=qk_weights,
            qk_masks=qk_masks,
            override_attention=query_agg_attn,  # ✅ 若非 None，会覆盖默认 attention
        )

        output_tokens = self.output(hidden_tokens)

        if return_attention_score or use_query_agg:
            return output_tokens, attention_scores, query_agg_attn
        else:
            return output_tokens





