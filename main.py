import ssl
import certifi

_original_create_default_context = ssl.create_default_context

def create_certifi_context(*args, **kwargs):
    kwargs.setdefault("cafile", certifi.where())
    return _original_create_default_context(*args, **kwargs)

ssl.create_default_context = create_certifi_context


import os
import time
import torch
import swanlab
import warnings
from transformers import AutoTokenizer, set_seed, AutoModelForSequenceClassification
from data_load import load_config, load_data, build_dataset, ID2LABEL, LABEL2ID
from model import build_model
from train_eval import build_dataloaders, build_optimizer, build_scheduler, train, predict
from metrics import all_metrics, print_report

warnings.filterwarnings("ignore")


# 初始化 SwanLab
def setup_swanlab(config: dict):
    if not config.get("use_swanlab", False):
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
    config = load_config("config.json")

    # 在模型创建之前固定随机种子
    set_seed(config["seed"])

    # 创建目录添加时间戳
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config["output_dir"], f"run_{run_timestamp}")
    best_model_dir = os.path.join(run_dir, "best_model")
    os.makedirs(run_dir, exist_ok=True)
    print(f"本次实验目录：{run_dir}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 加载数据
    train_samples = load_data(config["train_data_path"])
    dev_samples = load_data(config["dev_data_path"])
    test_samples = load_data(config["test_data_path"])
    print(f"\n 数据加载完毕\n 训练集：{len(train_samples)} 条，验证集：{len(dev_samples)} 条，测试集：{len(test_samples)} 条")

    print("\n 加载分词器与模型")
    # 分词
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_dataset = build_dataset(train_samples, tokenizer, config["max_length"])
    dev_dataset = build_dataset(dev_samples, tokenizer, config["max_length"])
    test_dataset = build_dataset(test_samples, tokenizer, config["max_length"])

    # 模型
    model = build_model(config, ID2LABEL, LABEL2ID).to(device)

    # DataLoader
    train_loader, dev_loader, test_loader = build_dataloaders(
        train_dataset, dev_dataset, test_dataset,
        tokenizer, config["batch_size"], config["seed"],
    )

    # 优化器和调度器
    num_training_steps = len(train_loader) * config["num_epochs"]
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, num_training_steps)

    swanlab_started = False
    try:
        # 初始化 SwanLab
        swanlab_started = setup_swanlab(config)
        log_fn = swanlab.log if swanlab_started else None

        # 训练
        print("开始训练！")
        best_epoch, best_macro_f1, _ = train(
            model, train_loader, dev_loader,
            optimizer, scheduler, device, config,
            ID2LABEL, best_model_dir, log_fn=log_fn,
        )
        tokenizer.save_pretrained(best_model_dir)
        print(f"最佳模型已保存至：{best_model_dir}")

        # 加载最佳模型，在测试集上进行评估
        print("\n加载最佳模型进行测试集评估")
        model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
        test_preds, test_labels = predict(model, test_loader, device)
        test_metrics = all_metrics(test_labels, test_preds, ID2LABEL, prefix="test_")

        print("\n测试集指标：")
        for k, v in test_metrics.items():
            print(f"  {k} = {v:.4f}")

        print("\n详细分类报告：")
        print(print_report(test_labels, test_preds, ID2LABEL))

    finally:
        if swanlab_started:
            swanlab.finish()

    print("训练完成！")


if __name__ == "__main__":
    main()
