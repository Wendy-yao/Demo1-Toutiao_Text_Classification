import ssl
import certifi

_original_create_default_context = ssl.create_default_context

def create_certifi_context(*args, **kwargs):
    kwargs.setdefault("cafile", certifi.where())
    return _original_create_default_context(*args, **kwargs)

ssl.create_default_context = create_certifi_context


import os
import torch
import swanlab
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EvalPrediction
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings

warnings.filterwarnings("ignore")

swanlab_api_key = os.getenv("SWANLAB_API_KEY")
if swanlab_api_key:
    swanlab.login(api_key=swanlab_api_key)


# 参数配置
class Config:
    # 实验标识
    experiment_name = "batchSize_16"  # 实验名
    output_dir = f"./outputs/{experiment_name}"  # 每个实验独立目录

    # 数据路径
    train_data_path = "data_demo1/train_3k.txt"  # 训练集
    dev_data_path = "data_demo1/dev_1k.txt"  # 验证集
    test_data_path = "data_demo1/test_1k.txt"  # 测试集

    # 模型参数
    model_name = os.getenv("MODEL_NAME", "bert-base-chinese")
    num_labels = 15
    max_length = 128

    # 训练参数
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 3
    warmup_ratio = 0.1
    weight_decay = 0.01

    # SwanLab配置
    swanlab_project = "toutiao-text-classification"
    swanlab_config = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "max_length": max_length,
        "model": model_name
    }

os.makedirs(Config.output_dir, exist_ok=True)
os.makedirs(os.path.join(Config.output_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(Config.output_dir, "logs"), exist_ok=True)


# 类别映射：根据数据集文档，分类code对应的类别名称
CODE2LABEL = {
    100: "民生", 101: "文化", 102: "娱乐", 103: "体育", 104: "财经",
    106: "房产", 107: "汽车", 108: "教育", 109: "科技", 110: "军事",
    112: "旅游", 113: "国际", 114: "股票", 115: "农业", 116: "电竞"
}

CODE2ID = {code: i for i, code in enumerate(CODE2LABEL.keys())}
ID2LABEL = {i: name for i, name in enumerate(CODE2LABEL.values())}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}


# 解析数据集的单行数据
def parse_line(line: str):
    parts = line.strip().split("_!_")
    if len(parts) < 4:
        return None
    news_id, code_str, category_name, title = parts[0], parts[1], parts[2], parts[3]
    try:
        label = CODE2ID[int(code_str)]
    except ValueError:
        return None
    # 只保留标题作为文本，关键词可选拼接
    text = title.strip()
    if not text:
        return None
    return text, label

# 加载数据集文件
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

# 加载数据集
print("-------加载数据中-------")
train_samples = load_data(Config.train_data_path)
dev_samples = load_data(Config.dev_data_path)
test_samples = load_data(Config.test_data_path)

print(f"训练集：{len(train_samples)} 条，验证集：{len(dev_samples)} 条，测试集：{len(test_samples)} 条")

# 转换为 Hugging Face Dataset 格式
train_dataset = HFDataset.from_list(train_samples)
dev_dataset = HFDataset.from_list(dev_samples)
test_dataset = HFDataset.from_list(test_samples)


print("-------加载分词器与模型-------")
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

# 对文本进行分词
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=Config.max_length,
        return_tensors=None
    )

# 对数据集进行分词
train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset = dev_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 设置数据集格式（指定标签列）
train_dataset = train_dataset.rename_column("label", "labels")
dev_dataset = dev_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

# 设置输出格式
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    Config.model_name,
    num_labels=Config.num_labels,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)


# 定义评估指标，计算准确率和 F1分数
def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "F1_weighted": f1}


# 配置 SwanLab 可视化
swanlab.init(
    project=Config.swanlab_project,
    experiment_name=Config.experiment_name,
    config=Config.swanlab_config
)

# 启动 TensorBoard → SwanLab 同步（用于捕获 Trainer 的日志）
swanlab.sync_tensorboard_torch()


# 配置训练参数
training_args = TrainingArguments(
    output_dir=os.path.join(Config.output_dir, "checkpoints"),
    num_train_epochs=Config.num_epochs,
    per_device_train_batch_size=Config.batch_size,
    per_device_eval_batch_size=Config.batch_size,
    learning_rate=Config.learning_rate,
    weight_decay=Config.weight_decay,
    warmup_ratio=Config.warmup_ratio,
    logging_dir=os.path.join(Config.output_dir, "logs"),
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    report_to="tensorboard",
    seed=42
)


# 创建 Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# 创建 Trainer 并训练
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


# 保存模型
save_path = os.path.join(Config.output_dir, "best_model")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"模型已保存至：{save_path}！")


# 在测试集上进行评估
test_results = trainer.evaluate(test_dataset)
print(f"测试集结果：{test_results}")

# 详细分类报告
print("\n 详细分类报告：")
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids
print(classification_report(true_labels, pred_labels, target_names=list(ID2LABEL.values())))

# 关闭 swanlab，结束训练
swanlab.finish()
print("训练完成！")


