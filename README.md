# demo1

基于 bert-base-chinese 的中文新闻文本分类项目，使用 Hugging Face Transformers 对今日头条文本分类数据进行微调。

## 项目结构

```text
demo1/
├── demo1.py
├── config.json
├── requirements.txt
├── README.md
├── .gitignore
└── data_demo1/
    ├── train_3k.txt
    ├── dev_1k.txt
    └── test_1k.txt
```

## 功能说明

- 使用 `bert-base-chinese` 进行 15 类中文新闻标题分类
- 从 `config.json` 读取训练参数
- 在创建模型前固定随机种子
- 使用验证集 `macro_f1` 选择最优模型
- 输出 `accuracy`、`weighted_f1`、`macro_f1` 和各类别 `recall`
- 测试集只推理一次，并基于同一次预测输出分类报告
- 支持 SwanLab 和 TensorBoard 记录训练过程

## 安装依赖

建议使用 Python 3.8 环境。

```powershell
python --version
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
  "num_labels": 15,
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

请先设置 SwanLab API Key：

```powershell
$env:SWANLAB_API_KEY="你的SwanLab API Key"
```

不要把真实 API Key 写入代码、README、`config.json` 或提交到 GitHub。

如果不想使用 SwanLab，可以把 `config.json` 改成：

```json
"use_swanlab": false
```

## 运行

在项目根目录执行：

```powershell
python demo1.py
```

脚本会自动读取当前目录下的 `config.json`。

## 输出结果

训练过程中会在 `output_dir` 指定目录下生成：

```text
outputs/batchSize_16/
├── checkpoints/
├── logs/
└── best_model/
```

其中 `best_model` 保存的是验证集 `macro_f1` 最优的模型。

## 数据说明

`data_demo1` 文件夹中包含训练集、验证集和测试集：

- `train_3k.txt`：训练集
- `dev_1k.txt`：验证集
- `test_1k.txt`：测试集

公开数据前，请确认数据集的授权和发布条件。


## 安全检查

提交代码前，请确认代码和配置文件中没有包含 API Key、密码或其他敏感信息。
