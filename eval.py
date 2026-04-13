"""
eval.py  ——  修复版 v3

修复：
  1. 去掉 ignore_mismatched_sizes=True，
     num_categories 由已保存的 config.json 自动还原，尺寸天然匹配。
  2. 不再需要显式传 num_categories，from_pretrained 从 config 读取。
"""

import gzip
import json

import torch
import typer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm

from helpers import (
    DataCollator,
    check_duplicate,
    MAX_LEN,
    parse_logits,
    prepare_inputs,
)
from model import CategoryLayoutLMv3

app = typer.Typer()
chen_cherry = SmoothingFunction()


@app.command()
def main(
    input_file: str = typer.Argument(..., help="input .jsonl.gz file"),
    model_path: str = typer.Argument(..., help="path to saved checkpoint"),
    batch_size: int = typer.Option(16, help="batch size for inference"),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # num_categories 从 config.json 自动还原，无需手动传入
    # 不传 ignore_mismatched_sizes，任何尺寸不匹配都会报错而不是静默跳过
    model = (
        CategoryLayoutLMv3.from_pretrained(
            model_path,
            num_labels=MAX_LEN,
            visual_embed=True,
        )
        .bfloat16()
        .to(device)
        .eval()
    )

    data_collator = DataCollator()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    datasets = []
    with gzip.open(input_file, "rt") as f:
        for line in tqdm(f):
            datasets.append(json.loads(line))
    datasets.sort(key=lambda x: len(x["source_boxes"]), reverse=True)

    total = total_out_idx = total_out_token = 0

    for i in tqdm(range(0, len(datasets), batch_size)):
        batch        = datasets[i: i + batch_size]
        model_inputs = data_collator(batch)
        model_inputs = prepare_inputs(model_inputs, model)

        with torch.no_grad():
            model_outputs = model(**model_inputs)

        logits = model_outputs.logits.cpu()

        for data, logit in zip(batch, logits):
            target_index = data["target_index"][:MAX_LEN]
            pred_index   = parse_logits(logit, len(target_index))

            assert len(pred_index) == len(target_index)
            assert not check_duplicate(pred_index)

            target_texts = data["target_texts"][:MAX_LEN]
            source_texts = data["source_texts"][:MAX_LEN]
            pred_texts   = [source_texts[idx] for idx in pred_index]

            total += 1
            total_out_idx += sentence_bleu(
                [target_index],
                [i + 1 for i in pred_index],
                smoothing_function=chen_cherry.method2,
            )
            total_out_token += sentence_bleu(
                [" ".join(target_texts).split()],
                " ".join(pred_texts).split(),
                smoothing_function=chen_cherry.method2,
            )

    print("total:     ", total)
    print("out_idx:   ", round(100 * total_out_idx   / total, 1))
    print("out_token: ", round(100 * total_out_token / total, 1))


if __name__ == "__main__":
    app()
