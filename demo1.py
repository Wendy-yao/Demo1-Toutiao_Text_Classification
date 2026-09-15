import ssl
import certifi

_original_create_default_context = ssl.create_default_context

def create_certifi_context(*args, **kwargs):
    kwargs.setdefault("cafile", certifi.where())
    return _original_create_default_context(*args, **kwargs)

ssl.create_default_context = create_certifi_context


import os
import json
import swanlab
import warnings
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EvalPrediction, set_seed
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

warnings.filterwarnings("ignore")
CONFIG_PATH = "config.json"


# 类别映射：根据数据集文档，分类 code 对应的中文类别名称
CODE2LABEL = {
    100: "民生", 101: "文化", 102: "娱乐", 103: "体育", 104: "财经",
    106: "房产", 107: "汽车", 108: "教育", 109: "科技", 110: "军事",
    112: "旅游", 113: "国际", 114: "股票", 115: "农业", 116: "电竞",
}

CODE2ID = {code: i for i, code in enumerate(CODE2LABEL.keys())}
ID2LABEL = {i: name for i, name in enumerate(CODE2LABEL.values())}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}


# 读取 JSON 配置文件
def load_config(config_path: str = CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["model_name"] = os.getenv("MODEL_NAME", config["model_name"])
    return config


# 解析数据集中的单行数据
def parse_line(line: str):
    parts = line.strip().split("_!_")
    if len(parts) < 4:
        return None

    _news_id, code_str, _category_name, title = parts[0], parts[1], parts[2], parts[3]
    try:
        label = CODE2ID[int(code_str)]
    except (KeyError, ValueError):
        return None

    # 只保留标题作为文本，关键词可选拼接
    text = title.strip()
    if not text:
        return None
    return text, label


# 从文件加载数据，返回样本列表
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


# 将样本列表转换为 HuggingFace Dataset，并完成分词
def build_dataset(samples, tokenizer, max_length: int):
    dataset = HFDataset.from_list(samples)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


# 定义评估指标，计算准确率、两种 F1分数和召回率
def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0)
    }

    recalls = recall_score(
        labels,
        predictions,
        labels=list(ID2LABEL.keys()),
        average=None,
        zero_division=0
    )
    for label_id, recall in zip(ID2LABEL.keys(), recalls):
        metrics[f"recall_{label_id}_{ID2LABEL[label_id]}"] = recall

    return metrics


# 配置 SwanLab
def setup_swanlab(config: dict):
    if not config["use_swanlab"]:
        return False

    swanlab_api_key = os.getenv("SWANLAB_API_KEY")
    if swanlab_api_key:
        swanlab.login(api_key=swanlab_api_key)

    swanlab.init(
        project=config["swanlab_project"],
        experiment_name=config["experiment_name"],
        config=config
    )
    swanlab.sync_tensorboard_torch()
    return True


def main():
    # 读取配置
    config = load_config(CONFIG_PATH)

    # 在创建模型之前固定随机种子
    set_seed(config["seed"])

    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "logs"), exist_ok=True)

    # 加载并处理数据
    print("-------加载数据中-------")
    train_samples = load_data(config["train_data_path"])
    dev_samples = load_data(config["dev_data_path"])
    test_samples = load_data(config["test_data_path"])
    print(f"训练集：{len(train_samples)} 条，验证集：{len(dev_samples)} 条，测试集：{len(test_samples)} 条")

    # 加载分词器与模型
    print("-------加载分词器与模型-------")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_dataset = build_dataset(train_samples, tokenizer, config["max_length"])
    dev_dataset = build_dataset(dev_samples, tokenizer, config["max_length"])
    test_dataset = build_dataset(test_samples, tokenizer, config["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    swanlab_started = False
    try:
        swanlab_started = setup_swanlab(config)

        training_args = TrainingArguments(
            output_dir=os.path.join(config["output_dir"], "checkpoints"),
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            warmup_ratio=config["warmup_ratio"],
            logging_dir=os.path.join(config["output_dir"], "logs"),
            logging_steps=config["logging_steps"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",  # 根据验证集 Macro-F1 选择最优模型
            greater_is_better=True,
            save_total_limit=config["save_total_limit"],
            report_to=config["report_to"],
            seed=config["seed"]
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        print("开始训练！")
        trainer.train()

        # 保存验证集 Macro-F1 最优模型
        save_path = os.path.join(config["output_dir"], "best_model")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"验证集 Macro-F1 最优模型已保存至：{save_path}！")

        # 在测试集上预测并输出分类报告
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
                zero_division=0
            )
        )
    finally:
        # 关闭 SwanLab
        if swanlab_started:
            swanlab.finish()

    print("训练完成！")


if __name__ == "__main__":
    main()
