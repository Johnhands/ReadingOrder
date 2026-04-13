"""
helpers.py  ——  支持元素类别嵌入（Category Embedding）的版本
改动说明：
  1. DataCollator 新增对 source_categories 字段的处理，
     生成 category_ids 张量（形状同 input_ids），CLS/EOS/PAD 位置填 0。
  2. boxes2inputs 同步新增 category_ids 返回，方便推理单样本时使用。
  3. prepare_inputs 无需修改，已能透传任意 key。
  4. parse_logits / check_duplicate 不变。
"""

from collections import defaultdict
from typing import List, Dict, Optional
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3ImageProcessor
from PIL import Image

# ──────────────────────────────────────────────
# 全局常量
# ──────────────────────────────────────────────
MAX_LEN = 510
CLS_TOKEN_ID = 0
UNK_TOKEN_ID = 3
EOS_TOKEN_ID = 2

# 类别定义：可根据实际数据集修改
# 0 保留给 CLS / EOS / PAD 等特殊位置
CATEGORY_PAD_ID = 0
CATEGORY_MAP = {
    "text":  1,
    "paragraph_title":      2,
    "number":      3,
    "table":     4,
    "header":    5,
    "footer":     6,
    "figure_title":     7,
    "image":       8,
    "vision_footnote":   9,
    "header_image":      10,
    "doc_title":  11,
    "reference_content":      12,
    "content":      13,
    "footer_image":     14,
    "display_formula":    15,
    "chart":     16,
    "abstract":     17,
    "aside_text":       18,
    "vertical_text":   19,
    "algorithm":      20,
    "seal":  21,
    "footnote":      22,
    "formula_number":      23,
}
NUM_CATEGORIES = len(CATEGORY_MAP) + 1   # +1 是 PAD/特殊位置


# ──────────────────────────────────────────────
# DataCollator
# ──────────────────────────────────────────────
class DataCollator:
    """
    在原有字段基础上，新增对 source_categories 的处理。

    数据集中每条样本应包含字段（新增）：
        source_categories: List[str | int]
            与 source_boxes 等长，每个元素是该 box 的类别名称（str）
            或直接是类别 ID（int）。
            若数据集暂时没有该字段，则自动全部填充为 CATEGORY_PAD_ID，
            模型依然可以正常训练（类别嵌入退化为全零偏置）。
    """

    def __init__(self, image_dir: str = ""):
        self.image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        self.image_dir = image_dir

    # ------------------------------------------------------------------
    # 内部工具：将原始类别列表转为 int id 列表
    # ------------------------------------------------------------------
    @staticmethod
    def _to_category_ids(raw_categories: Optional[List], length: int) -> List[int]:
        """
        raw_categories: 原始类别列表，元素可以是 str 或 int，允许为 None。
        length:         对应 source_boxes 截断后的长度。
        返回长度为 length 的 int 列表。
        """
        if raw_categories is None:
            return [CATEGORY_PAD_ID] * length

        ids = []
        for c in raw_categories[:length]:
            if isinstance(c, int):
                ids.append(c)
            elif isinstance(c, str):
                ids.append(CATEGORY_MAP.get(c.lower(), CATEGORY_PAD_ID))
            else:
                ids.append(CATEGORY_PAD_ID)

        # 如果截断后还不够 length（理论上不会），补齐
        if len(ids) < length:
            ids += [CATEGORY_PAD_ID] * (length - len(ids))
        return ids

    # ------------------------------------------------------------------
    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        bbox          = []
        labels        = []
        input_ids     = []
        attention_mask = []
        category_ids  = []   # ← 新增
        images        = []

        # ── 读取图像 ──────────────────────────────────────────────────
        for feature in features:
            img = Image.open(feature["image_path"]).convert("RGB")
            images.append(img)
        pixel_values = self.image_processor(images, return_tensors="pt").pixel_values

        # ── 构造每条样本的序列 ────────────────────────────────────────
        for feature in features:
            # bbox
            _bbox = feature["source_boxes"]
            if len(_bbox) > MAX_LEN:
                _bbox = _bbox[:MAX_LEN]
            seq_len = len(_bbox)   # 截断后的有效长度

            # labels
            _labels = feature["target_index"]
            if len(_labels) > MAX_LEN:
                _labels = _labels[:MAX_LEN]

            # input_ids & attention_mask（原逻辑不变）
            _input_ids     = [UNK_TOKEN_ID] * seq_len
            _attention_mask = [1] * seq_len

            # category_ids ← 新增
            _cat_ids = self._to_category_ids(
                feature.get("source_categories", None), seq_len
            )

            assert len(_bbox) == len(_labels) == len(_input_ids) == len(_attention_mask) == len(_cat_ids)

            bbox.append(_bbox)
            labels.append(_labels)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            category_ids.append(_cat_ids)

        # ── 添加 CLS 和 EOS ───────────────────────────────────────────
        for i in range(len(bbox)):
            bbox[i]           = [[0, 0, 0, 0]] + bbox[i]          + [[0, 0, 0, 0]]
            labels[i]         = [-100]          + labels[i]        + [-100]
            input_ids[i]      = [CLS_TOKEN_ID]  + input_ids[i]     + [EOS_TOKEN_ID]
            attention_mask[i] = [1]             + attention_mask[i] + [1]
            category_ids[i]   = [CATEGORY_PAD_ID] + category_ids[i] + [CATEGORY_PAD_ID]  # ← 新增

        # ── Padding 到批次最大长度 ────────────────────────────────────
        max_len = max(len(x) for x in bbox)
        for i in range(len(bbox)):
            pad = max_len - len(bbox[i])
            bbox[i]           += [[0, 0, 0, 0]] * pad
            labels[i]         += [-100]          * pad
            input_ids[i]      += [EOS_TOKEN_ID]  * pad
            attention_mask[i] += [0]             * pad
            category_ids[i]   += [CATEGORY_PAD_ID] * pad   # ← 新增

        ret = {
            "bbox":           torch.tensor(bbox),
            "attention_mask": torch.tensor(attention_mask),
            "labels":         torch.tensor(labels),
            "input_ids":      torch.tensor(input_ids),
            "pixel_values":   pixel_values,
            "category_ids":   torch.tensor(category_ids, dtype=torch.long),  # ← 新增
        }
        # label 后处理（原逻辑不变）
        ret["labels"][ret["labels"] > MAX_LEN] = -100
        ret["labels"][ret["labels"] > 0] -= 1
        return ret


# ──────────────────────────────────────────────
# boxes2inputs（推理单样本辅助函数，同步新增 category_ids）
# ──────────────────────────────────────────────
def boxes2inputs(
    boxes: List[List[int]],
    categories: Optional[List] = None,
) -> Dict[str, torch.Tensor]:
    """
    Args:
        boxes:      source_boxes 列表
        categories: source_categories 列表（可选），与 boxes 等长
    """
    cat_ids = DataCollator._to_category_ids(categories, len(boxes))

    bbox           = [[0, 0, 0, 0]] + boxes     + [[0, 0, 0, 0]]
    input_ids      = [CLS_TOKEN_ID] + [UNK_TOKEN_ID] * len(boxes) + [EOS_TOKEN_ID]
    attention_mask = [1]            + [1]        * len(boxes)      + [1]
    cat_ids        = [CATEGORY_PAD_ID] + cat_ids + [CATEGORY_PAD_ID]

    return {
        "bbox":           torch.tensor([bbox]),
        "attention_mask": torch.tensor([attention_mask]),
        "input_ids":      torch.tensor([input_ids]),
        "category_ids":   torch.tensor([cat_ids], dtype=torch.long),
    }


# ──────────────────────────────────────────────
# prepare_inputs（不变，透传所有 key 到 model.device）
# ──────────────────────────────────────────────
def prepare_inputs(
    inputs: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    ret = {}
    for k, v in inputs.items():
        v = v.to(model.device)
        if torch.is_floating_point(v):
            v = v.to(model.dtype)
        ret[k] = v
    return ret


# ──────────────────────────────────────────────
# parse_logits / check_duplicate（不变）
# ──────────────────────────────────────────────
def parse_logits(logits: torch.Tensor, length: int) -> List[int]:
    logits = logits[1: length + 1, :length]
    orders = logits.argsort(descending=False).tolist()
    ret = [o.pop() for o in orders]
    while True:
        order_to_idxes = defaultdict(list)
        for idx, order in enumerate(ret):
            order_to_idxes[order].append(idx)
        order_to_idxes = {k: v for k, v in order_to_idxes.items() if len(v) > 1}
        if not order_to_idxes:
            break
        for order, idxes in order_to_idxes.items():
            idxes_to_logit = {idx: logits[idx, order] for idx in idxes}
            idxes_to_logit = sorted(idxes_to_logit.items(), key=lambda x: x[1], reverse=True)
            for idx, _ in idxes_to_logit[1:]:
                ret[idx] = orders[idx].pop()
    return ret


def check_duplicate(a: List[int]) -> bool:
    return len(a) != len(set(a))
