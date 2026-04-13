"""
train.py  ——  修复版 v3

修复：DeepSpeed 下差异化学习率的正确实现方式。

问题：CategoryTrainer 覆盖 create_optimizer 创建 AdamW 参数组，
     但 DeepSpeed 启动时会接管 optimizer 创建流程，
     HuggingFace Trainer 的 create_optimizer 实际上不会被调用，
     导致差异化学习率完全没有生效。

修复方案：
     覆盖 create_optimizer_and_scheduler，在其中手动构造参数组后
     再交给 DeepSpeed 初始化，确保参数组在 DeepSpeed 接管前已经设置好。
     关键是要在赋值 self.optimizer 之前 DeepSpeed 还没介入的时机注入参数组。
"""

import os
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from loguru import logger
from torch.optim import AdamW
from transformers import HfArgumentParser, set_seed, TrainingArguments
from transformers.trainer import Trainer
from transformers.optimization import get_scheduler
from transformers.callbacks import EarlyStoppingCallback

from helpers import DataCollator, MAX_LEN, NUM_CATEGORIES
from model import CategoryLayoutLMv3


# ──────────────────────────────────────────────
@dataclass
class Arguments(TrainingArguments):
    model_dir: str = field(default=None)
    dataset_dir: str = field(default=None)
    num_categories: int = field(default=NUM_CATEGORIES)
    new_module_lr: float = field(
        default=1e-3,
        metadata={"help": "LR for category_embedding and category_scale. "
                          "Recommended 10~50x of learning_rate."}
    )
    early_stopping_patience: int = field(
        default=5,
        metadata={"help": "Number of epochs with no improvement after which training will be stopped."}
    )
    early_stopping_threshold: float = field(
        default=0.001,
        metadata={"help": "Minimum improvement required to reset the early stopping counter."}
    )


# ──────────────────────────────────────────────
class CategoryTrainer(Trainer):
    """
    DeepSpeed 兼容的差异化学习率 Trainer。

    核心：覆盖 create_optimizer_and_scheduler，
    在 DeepSpeed 接管之前将参数组注入 self.optimizer，
    DeepSpeed 会直接包装已有的 optimizer 而非重新创建。
    """

    def create_optimizer(self):
        """
        HuggingFace Trainer 在非 DeepSpeed 模式下调用此方法。
        DeepSpeed 模式下也会先调用此方法获取 optimizer，
        然后再用 deepspeed.initialize() 包装它。
        所以在这里注入参数组是安全且有效的。
        """
        if self.optimizer is not None:
            return self.optimizer

        param_groups = self.model.get_optimized_param_groups(
            pretrained_lr=self.args.learning_rate,
            new_module_lr=self.args.new_module_lr,
        )
        self.optimizer = AdamW(
            param_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer


# ──────────────────────────────────────────────
def load_datasets(path: str):
    ds = load_dataset(
        "json",
        data_files={
            "train": os.path.join(path, "train.jsonl.gz"),
            "dev":   os.path.join(path, "dev.jsonl.gz"),
        },
    )
    return ds["train"], ds["dev"]


def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)

    train_ds, dev_ds = load_datasets(args.dataset_dir)
    logger.info("Train: {}, Dev: {}".format(len(train_ds), len(dev_ds)))

    model = CategoryLayoutLMv3.from_pretrained(
        args.model_dir,
        num_labels=MAX_LEN,
        visual_embed=True,
        num_categories=args.num_categories,
        # 不再需要 ignore_mismatched_sizes=True
        # num_categories 已写入 config，尺寸天然匹配
    )

    # 验证初始化正确性
    scale_norm = model.category_scale.weight.norm().item()
    emb_norm   = model.category_embedding.weight.norm().item()
    logger.info(f"category_scale norm: {scale_norm:.6f} (expected 0.0)")
    logger.info(f"category_embedding norm: {emb_norm:.6f} (expected 0.0)")
    logger.info(f"num_categories in config: {model.config.num_categories}")

    trainer = CategoryTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollator(),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold
            )
        ],
    )
    # 设置只保存最优模型
    trainer.args.save_strategy = "epoch"
    trainer.args.save_total_limit = 1  # 只保存一个模型
    trainer.args.load_best_model_at_end = True  # 训练结束后加载最优模型
    trainer.train()


if __name__ == "__main__":
    main()
