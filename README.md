# demo1

基于 `bert-base-chinese` 的中文新闻文本分类项目，使用 PyTorch 手写训练、验证和测试流程，对今日头条文本分类数据进行 15 类分类。

## 项目结构

```text
demo1/
├── main.py              # 主入口：串联数据、模型、训练、测试
├── demo1.py             # 可选兼容入口
├── data_load.py         # 配置读取、类别映射、数据解析、Dataset 构建
├── model.py             # 模型加载与分类头构建
├── train_eval.py        # 手写训练、验证、预测流程
├── metrics.py           # 手写 accuracy、precision、recall、F1、分类报告
├── config.json          # 训练配置
├── requirements.txt     # 依赖列表
├── README.md
├── .gitignore
└── data_demo1/
    ├── train_3k.txt
    ├── dev_1k.txt
    └── test_1k.txt
```

## 功能说明

- 不再依赖 `transformers.Trainer`，改为手写训练、验证和测试循环。
- 使用 `DataLoader`、`AdamW`、线性 warmup scheduler 完成模型训练。
- 使用验证集 `macro_f1` 选择最优模型。
- 手写实现 `accuracy`、各类别 `precision / recall / F1`、`macro-F1`、`weighted-F1` 和分类报告。
- 每次训练会创建带时间戳的实验目录，避免覆盖旧模型。
- 支持通过 `config.json` 修改训练参数。
- 支持通过环境变量配置本地模型路径和 SwanLab API Key。

## 安装依赖

建议使用 Python 3.8 环境。

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果 PowerShell 禁止执行激活脚本，可以直接使用虚拟环境中的 Python：

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```


## 配置文件

训练参数集中写在 `config.json` 中：

```json
{
  "experiment_name": "batchSize_16",
  "output_dir": "outputs/batchSize_16",
  "train_data_path": "data_demo1/train_3k.txt",
  "dev_data_path": "data_demo1/dev_1k.txt",
  "test_data_path": "data_demo1/test_1k.txt",
  "model_name": "bert-base-chinese",
  "max_length": 128,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "num_epochs": 3,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01,
  "logging_steps": 50,
  "save_total_limit": 2,
  "seed": 42,
  "use_swanlab": true,
  "swanlab_project": "text-classification",
  "report_to": "tensorboard"
}
```

如果要调整 batch size、学习率、训练轮数、数据路径、输出路径或随机种子，直接修改 `config.json`。

## 配置模型

默认模型为：

```text
bert-base-chinese
```

如果已经下载了本地模型，可以在 PowerShell 中设置环境变量：

```powershell
$env:MODEL_NAME="D:\demo1\models\bert-base-chinese"
```

程序会优先使用 `MODEL_NAME` 环境变量；如果没有设置，就使用 `config.json` 中的 `model_name`。


## 配置 SwanLab

如果 `config.json` 中：

```json
"use_swanlab": true
```

可以设置 SwanLab API Key：

```powershell
$env:SWANLAB_API_KEY="你的SwanLab API Key"
```

不要把真实 API Key 写入代码、README、`config.json` 或提交到 GitHub。

如果不想使用 SwanLab，可以把 `config.json` 改成：

```json
"use_swanlab": false
```

## 运行

推荐运行主入口：

```powershell
python main.py
```

如果 `demo1.py` 已经改成兼容入口，也可以运行：

```powershell
python demo1.py
```

## 训练流程

当前训练流程由 `train_eval.py` 手写实现：

1. 构建 `DataLoader`
2. 前向传播计算 loss
3. `loss.backward()` 反向传播
4. 梯度裁剪
5. `optimizer.step()` 更新参数
6. `scheduler.step()` 更新学习率
7. 每个 epoch 后在验证集评估
8. 如果验证集 `macro_f1` 提升，则保存最佳模型
9. 最后加载最佳模型，在测试集上预测并输出报告

## 指标说明

`metrics.py` 中手写实现了：

- `accuracy`
- 每个类别的 `precision`
- 每个类别的 `recall`
- 每个类别的 `F1`
- `macro_f1`
- `weighted_f1`
- 类似 `sklearn.classification_report` 的文本报告

其中：

- `macro_f1`：各类别 F1 的简单平均，适合观察类别整体表现。
- `weighted_f1`：按各类别样本数加权的 F1，更容易受大类别影响。
- 最佳模型按照验证集 `macro_f1` 保存。

## 输出结果

每次运行会创建一个新的时间戳目录，避免覆盖历史实验：

```text
outputs/batchSize_16/
└── run_YYYYMMDD_HHMMSS/
    ├── best_model/
    ├── history.json
    └── test_metrics.json
```

其中：

- `best_model/`：验证集 `macro_f1` 最优的模型权重和 tokenizer
- `history.json`：每个 epoch 的训练 loss、验证 loss 和验证指标
- `test_metrics.json`：测试集最终指标

## 数据说明

`data_demo1` 文件夹中包含训练集、验证集和测试集：

- `train_3k.txt`：训练集
- `dev_1k.txt`：验证集
- `test_1k.txt`：测试集

数据行格式大致为：

```text
news_id_!_code_!_category_name_!_title
```

程序当前只使用标题 `title` 作为输入文本。

公开数据前，请确认数据集的授权和发布条件。


## 安全检查

提交代码前，请确认代码和配置文件中没有包含 API Key、密码或其他敏感信息。