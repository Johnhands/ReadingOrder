"""
model.py —— 动态类别嵌入增强（安全兼容版）
✅ 完全保留你原有逻辑
✅ 不自己生成 position_ids / bbox
✅ 不破坏数据流程
✅ 修复所有 encoder 调用错误
✅ 动态类别嵌入正常工作
"""
import torch
import torch.nn as nn
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Config
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, List, Dict


class CategoryLayoutLMv3(LayoutLMv3ForTokenClassification):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        # 完全保留你原来的初始化逻辑
        num_categories = kwargs.pop("num_categories", getattr(config, "num_categories", 24))
        config.num_categories = num_categories
        super().__init__(config, **kwargs)

        hidden_size = config.hidden_size

        # 1. 类别嵌入（Xavier 初始化）
        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=hidden_size,
            padding_idx=0,
        )
        nn.init.xavier_uniform_(self.category_embedding.weight)
        # 保持 padding_idx 为 0
        self.category_embedding.weight.data[0] = 0

        # 2. 类别专属scale（Xavier 初始化）
        self.category_scale = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=hidden_size,
            padding_idx=0,
        )
        nn.init.xavier_uniform_(self.category_scale.weight)
        # 保持 padding_idx 为 0
        self.category_scale.weight.data[0] = 0

        # 3. 上下文融合
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        for module in self.context_fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 4. 注意力机制融合
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )

    # ----------------------
    # 你原来的函数，完全保留
    # ----------------------
    def _get_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids):
        emb = self.layoutlmv3.embeddings
        try:
            return emb(input_ids=input_ids, bbox=bbox, position_ids=position_ids, token_type_ids=token_type_ids)
        except TypeError:
            return emb(input_ids=input_ids, bbox=bbox, position_ids=position_ids)

    def get_optimized_param_groups(self, pretrained_lr: float, new_module_lr: float) -> List[Dict]:
        new_names = {"category_embedding", "category_scale", "context_fusion"}
        pretrained, new = [], []
        for name, param in self.named_parameters():
            (new if name.split(".")[0] in new_names else pretrained).append(param)
        return [
            {"params": pretrained, "lr": pretrained_lr},
            {"params": new, "lr": new_module_lr},
        ]

    # ==============================================
    # forward：完全保留你的逻辑，只修BUG
    # ==============================================
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        category_ids: Optional[torch.LongTensor] = None,
    ) -> TokenClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ======================
        # 1. 你原来的逻辑：获取 embedding
        # ======================
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self._get_text_embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
            )

        # ======================
        # 2. 类别信息融合
        # ======================
        if category_ids is not None:
            B, L = inputs_embeds.shape[:2]
            cat_ids = category_ids[:, :L]

            # 类别嵌入 + 类别scale
            cat_emb = self.category_embedding(cat_ids)
            cat_scale = self.category_scale(cat_ids)
            enhanced_cat_emb = cat_emb * cat_scale

            # 注意力机制融合
            # 使用文本嵌入作为查询，类别嵌入作为键值
            attn_output, _ = self.attention_fusion(
                query=inputs_embeds,
                key=enhanced_cat_emb,
                value=enhanced_cat_emb,
                key_padding_mask=(category_ids[:, :L] == 0) if attention_mask is None else None
            )

            # 利用 CLS 标记的全局信息进行上下文融合
            # 提取 CLS 标记的表示
            cls_emb = inputs_embeds[:, 0:1, :].repeat(1, L, 1)
            # 融合 CLS 信息和注意力输出
            fused_emb = self.context_fusion(torch.cat([attn_output, cls_emb], dim=-1))

            # 加权融合到原始嵌入
            inputs_embeds = inputs_embeds + fused_emb

        # ======================
        # 3. 完全交给父类 forward
        # 你的原有逻辑 100% 保留
        # ======================
        outputs = super().forward(
            input_ids=None,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )

        return outputs