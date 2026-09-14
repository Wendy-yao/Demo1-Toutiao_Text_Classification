# demo1

基于 BERT 的中文新闻文本分类项目。

## 项目结构

```text
demo1/
├── demo1.py
├── data_demo1/
│   ├── train_3k.txt
│   ├── dev_1k.txt
│   └── test_1k.txt
├── config.json
├── requirements.txt
├── README.md
└── .gitignore
```

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

## 配置 SwanLab

请先在 SwanLab 获取自己的 API Key，然后在 PowerShell 中设置环境变量：

```powershell
$env:SWANLAB_API_KEY="你的SwanLab API Key"
```

不要把真实 API Key 写入代码、README 或提交到 GitHub。

## 配置模型

默认情况下，程序使用：

```text
bert-base-chinese
```

如果已经下载了本地模型，可以设置本地模型路径：

```powershell
$env:MODEL_NAME="D:\demo1\models\bert-base-chinese"
```

请将路径替换为你自己的模型路径。

## 配置文件

训练参数集中写在 `config.json` 中，例如 batch size、学习率、训练轮数、数据路径、输出路径和随机种子。

如果要调整参数，优先修改 `config.json`，不需要直接改 `demo1.py`。

## 运行

```powershell
python demo1.py
```

也可以指定其他配置文件：

```powershell
python demo1.py --config config.json
```

## 数据说明

`data_demo1` 文件夹中包含训练集、验证集和测试集：

- `train_3k.txt`：训练集
- `dev_1k.txt`：验证集
- `test_1k.txt`：测试集

公开数据前，请确认数据集的授权和发布条件。

## 安全检查

提交代码前，请确认代码中没有包含 API Key、密码或其他敏感信息。
