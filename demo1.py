import argparse
import json
import os
import random
import ssl
import warnings
from pathlib import Path

import numpy as np
import swanlab
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    import certifi
except ImportError:
    certifi = None


_original_create_default_context = ssl.create_default_context


def create_certifi_context(*args, **kwargs):
    if certifi is not None:
        kwargs.setdefault("cafile", certifi.where())
    return _original_create_default_context(*args, **kwargs)


ssl.create_default_context = create_certifi_context
warnings.filterwarnings("ignore")


# 类别映射：根据数据集文档，分类 code 对应的类别名称
CODE2LABEL = {
    100: "民生", 101: "文化", 102: "娱乐", 103: "体育", 104: "财经",
    106: "房产", 107: "汽车", 108: "教育", 109: "科技", 110: "军事",
    112: "旅游", 113: "国际", 114: "股票", 115: "农业", 116: "电竞",
}

CODE2ID = {code: i for i, code in enumerate(CODE2LABEL.keys())}
ID2LABEL = {i: name for i, name in enumerate(CODE2LABEL.values())}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}


CONFIG_KEYS = [
    "experiment_name",
    "output_dir",
    "train_data_path",
    "dev_data_path",
    "test_data_path",
    "model_name",
    "num_labels",
    "max_length",
    "batch_size",
    "learning_rate",
    "num_epochs",
    "warmup_ratio",
    "weight_decay",
    "logging_steps",
    "save_total_limit",
    "seed",
    "use_swanlab",
    "swanlab_project",
    "report_to",
]


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    missing_keys = [key for key in CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f"配置文件缺少必要参数：{missing_keys}")

    config["model_name"] = os.getenv("MODEL_NAME", config["model_name"])
    return config


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def parse_line(line: str):
    parts = line.strip().split("_!_")
    if len(parts) < 4:
        return None

    _news_id, code_str, _category_name, title = parts[0], parts[1], parts[2], parts[3]
    try:
        label = CODE2ID[int(code_str)]
    except (KeyError, ValueError):
        return None

    text = title.strip()
    if not text:
        return None
    return text, label


def load_data(file_path: str):
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            result = parse_line(line)
            if result:
                text, label = result
                samples.append({"text": text, "label": label})
    return samples


def build_dataset(samples, tokenizer, max_length: int):
    dataset = HFDataset.from_list(samples)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }

    recalls = recall_score(
        labels,
        predictions,
        labels=list(ID2LABEL.keys()),
        average=None,
        zero_division=0,
    )
    for label_id, recall in zip(ID2LABEL.keys(), recalls):
        metrics[f"recall_{label_id}_{ID2LABEL[label_id]}"] = recall

    return metrics


def setup_swanlab(config: dict):
    if not config["use_swanlab"]:
        return False

    swanlab_api_key = os.getenv("SWANLAB_API_KEY")
    if swanlab_api_key:
        swanlab.login(api_key=swanlab_api_key)

    swanlab.init(
        project=config["swanlab_project"],
        experiment_name=config["experiment_name"],
        config=config,
    )
    swanlab.sync_tensorboard_torch()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="Path to config JSON file.")
    args = parser.parse_args()

    config = load_config(args.config)
    if config["num_labels"] != len(ID2LABEL):
        raise ValueError("num_labels must match the number of labels in CODE2LABEL.")

    set_global_seed(config["seed"])

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    print("-------加载数据中-------")
    train_samples = load_data(config["train_data_path"])
    dev_samples = load_data(config["dev_data_path"])
    test_samples = load_data(config["test_data_path"])
    print(
        f"训练集：{len(train_samples)} 条，"
        f"验证集：{len(dev_samples)} 条，"
        f"测试集：{len(test_samples)} 条"
    )

    print("-------加载分词器与模型-------")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_dataset = build_dataset(train_samples, tokenizer, config["max_length"])
    dev_dataset = build_dataset(dev_samples, tokenizer, config["max_length"])
    test_dataset = build_dataset(test_samples, tokenizer, config["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    swanlab_started = False
    try:
        swanlab_started = setup_swanlab(config)

        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "checkpoints"),
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            warmup_ratio=config["warmup_ratio"],
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=config["logging_steps"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=config["save_total_limit"],
            report_to=config["report_to"],
            seed=config["seed"],
            data_seed=config["seed"],
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print("开始训练！")
        trainer.train()

        save_path = os.path.join(output_dir, "best_model")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"验证集 Macro-F1 最优模型已保存至：{save_path}！")

        print("\n测试集评估：")
        test_predictions = trainer.predict(test_dataset, metric_key_prefix="test")
        print(test_predictions.metrics)

        pred_labels = np.argmax(test_predictions.predictions, axis=1)
        true_labels = test_predictions.label_ids
        print("\n详细分类报告：")
        print(
            classification_report(
                true_labels,
                pred_labels,
                labels=list(ID2LABEL.keys()),
                target_names=list(ID2LABEL.values()),
                zero_division=0,
            )
        )
    finally:
        if swanlab_started:
            swanlab.finish()

    print("训练完成！")


if __name__ == "__main__":
    main()
