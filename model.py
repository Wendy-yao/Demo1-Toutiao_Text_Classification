from transformers import AutoModelForSequenceClassification

# 加载预训练模型
def build_model(config: dict, id2label: dict, label2id: dict):
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=15,
        id2label=id2label,
        label2id=label2id,
    )
    return model
