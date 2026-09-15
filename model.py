from transformers import AutoModelForSequenceClassification

# 加载预训练模型
def build_model(config: dict, ID2LABEL: dict, LABEL2ID: dict):
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=15,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model
