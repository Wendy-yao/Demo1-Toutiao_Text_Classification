import os
import json
from datasets import Dataset as HFDataset


# 类别映射
CODE2LABEL = {
    100: "民生", 101: "文化", 102: "娱乐", 103: "体育", 104: "财经",
    106: "房产", 107: "汽车", 108: "教育", 109: "科技", 110: "军事",
    112: "旅游", 113: "国际", 114: "股票", 115: "农业", 116: "电竞",
}
CODE2ID = {code: i for i, code in enumerate(CODE2LABEL.keys())}
ID2LABEL = {i: name for i, name in enumerate(CODE2LABEL.values())}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}


# 读取 JSON 配置文件
def load_config(config_path: str = "config.json"):
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


# 将样本列表转换为 HuggingFace Dataset，并完成分词。
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
