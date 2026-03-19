# TAMO 服务器部署与运行指南

本文基于当前仓库实际代码整理，不按 README 的简化描述写。目标是让你把项目完整放到一台 Linux GPU 服务器上，从零完成：

1. 环境安装
2. 模型下载
3. 数据下载
4. 数据预处理
5. 自检
6. 推理
7. 训练
8. 常见问题排查

## 1. 项目现状与关键约束

先明确几个和部署直接相关的事实：

- 项目是纯 Python 项目，没有 `Dockerfile`、`docker-compose`、Web 服务入口或 systemd 配置。
- 默认工作目录就是仓库根目录，路径由 [`src/global_path.py`](/Users/zzx/02Research/PaperA/TAMO/src/global_path.py) 控制，目前固定为 `.`。
- 代码默认从本地目录读取模型，不依赖运行时在线拉取。
- 即使运行纯文本模型 `llm` / `pt_llm` / `mistral` / `pt_mistral`，数据集 `__getitem__` 也会读取 `dataset/.../<split>/graphs/*.pt`，所以不能只下载原始数据，必须先做图预处理。
- 训练和推理都会直接调用 `wandb.init()`，如果服务器不能联网，必须提前设 `WANDB_MODE=offline` 或 `WANDB_DISABLED=true`。
- 推荐环境已经被仓库收敛到了 `Python 3.9 + torch 2.2.1 + CUDA 11.8`，见 [`requirements.sh`](/Users/zzx/02Research/PaperA/TAMO/requirements.sh) 和 [`requirements.txt`](/Users/zzx/02Research/PaperA/TAMO/requirements.txt)。

## 2. 推荐服务器配置

### 最低建议

- Linux x86_64
- Python 3.9
- NVIDIA GPU 1 张
- CUDA 11.8 兼容驱动
- 64GB+ 内存
- 200GB+ 磁盘

### 更稳妥的配置

- 1 到 2 张 80GB 显存 GPU
- 128GB 内存
- 300GB+ 磁盘

原因：

- Llama-2-7B / Mistral-7B 本体就比较大。
- 还要额外存 sentence-transformers、bert-base-uncased、多个数据集和预处理图文件。
- 代码里 full fine-tuning / SFT 路径默认按双卡大显存写过 `max_memory={0:'80GiB',1:'80GiB'}`。

## 3. 目录约定

假设你在服务器上的部署目录是：

```bash
/data/TAMO
```

推荐目录结构如下：

```text
/data/TAMO
├── dataset
├── models
├── output
├── script
├── src
├── inference.py
├── table_train.py
└── server_smoke_test.py
```

由于 [`src/global_path.py`](/Users/zzx/02Research/PaperA/TAMO/src/global_path.py) 当前写死为 `.`，所以：

- 运行命令时应在仓库根目录执行
- `dataset/` 和 `models/` 也应放在仓库根目录下

如果你以后想把模型和数据放到别的挂载盘，可以把 `src/global_path.py` 改成绝对路径，例如：

```python
global_path = '/data/TAMO'
```

改完后，所有命令仍然建议在仓库根目录执行。

## 4. 上传项目到服务器

如果本地已经有代码仓库：

```bash
git clone <your-repo-url> /data/TAMO
cd /data/TAMO
```

如果是直接拷贝当前目录：

```bash
rsync -av ./ /data/TAMO/
cd /data/TAMO
```

## 5. 创建 Python 环境

推荐使用 conda：

```bash
conda create -n TAMO python=3.9 -y
conda activate TAMO
python -V
```

如果服务器没有 conda，也可以用 `venv`，但由于 PyTorch + PyG 的二进制依赖较重，conda 更稳。

## 6. 安装依赖

### 推荐方式

直接使用仓库自带安装脚本：

```bash
bash requirements.sh
```

这个脚本会按正确顺序安装：

1. `torch==2.2.1+cu118`
2. `torch-scatter` / `torch-sparse` / `torch-cluster` / `torch-spline-conv`
3. 其余 Python 依赖

### 安装完成后建议验证

```bash
python - <<'PY'
import torch
import torch_geometric
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("torch_geometric:", torch_geometric.__version__)
PY
```

## 7. 下载模型

仓库需要的模型不止大语言模型，还包括图预处理时用到的 embedding 模型：

- `sentence-transformers/all-roberta-large-v1`
- `google-bert/bert-base-uncased`
- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-2-7b-chat`
- `osunlp/TableLlama`
- `mistralai/Mistral-7B-v0.1`
- `mistralai/Mistral-7B-Instruct-v0.2`

直接执行：

```bash
huggingface-cli login
bash download_models.sh
```

### 重要坑 1：Llama-2 chat 路径名不一致

代码里 [`src/model/__init__.py`](/Users/zzx/02Research/PaperA/TAMO/src/model/__init__.py) 对 `7b_chat` 的路径写的是：

```text
./models/meta-llama/Llama-2-7b-chat-hf
```

但 [`download_models.sh`](/Users/zzx/02Research/PaperA/TAMO/download_models.sh) 下载到的是：

```text
./models/meta-llama/Llama-2-7b-chat
```

所以下载完以后，必须二选一：

### 方案 A：重命名目录

```bash
mv ./models/meta-llama/Llama-2-7b-chat ./models/meta-llama/Llama-2-7b-chat-hf
```

### 方案 B：建软链接

```bash
ln -s ./Llama-2-7b-chat ./models/meta-llama/Llama-2-7b-chat-hf
```

我更建议直接重命名，少一个软链接依赖。

### 下载完成后，模型目录应至少包含

```text
models/
├── google-bert/bert-base-uncased
├── meta-llama/Llama-2-7b-hf
├── meta-llama/Llama-2-7b-chat-hf
├── mistralai/Mistral-7B-v0.1
├── mistralai/Mistral-7B-Instruct-v0.2
├── sentence-transformers/all-roberta-large-v1
└── tablellama
```

## 8. 下载数据集

### 当前仓库自带的数据

仓库里已经带了 `StructProbe` 的 Hugging Face disk 数据：

```text
dataset/structProbe/structProbe/{train,test,validation}
```

但现在仓库里还没有其他公开数据集的完整本地副本。

### 下载公开数据集

在仓库根目录执行：

```bash
python download_dataset.py
```

它会下载并保存到：

- `dataset/wikisql/wikisql`
- `dataset/wtq/wikitablequestions`
- `dataset/fetaqa/fetaqa`
- `dataset/hitab/hitab`

### 下载后建议检查

```bash
find dataset -maxdepth 3 -type d | sort
```

## 9. 生成图预处理产物

这是部署里最容易漏掉的一步，也是项目能否运行的硬前置。

### 为什么必须做

例如：

- [`src/dataset/structProbe.py`](/Users/zzx/02Research/PaperA/TAMO/src/dataset/structProbe.py)
- [`src/dataset/wtq.py`](/Users/zzx/02Research/PaperA/TAMO/src/dataset/wtq.py)
- [`src/dataset/wikisql.py`](/Users/zzx/02Research/PaperA/TAMO/src/dataset/wikisql.py)

这些数据集在 `__getitem__` 中都会直接读取：

```text
dataset/.../<split>/graphs/{index}.pt
```

如果没做预处理，训练和推理都会直接报文件不存在。

### 一键预处理

```bash
bash script/data_preprocess.sh
```

它会执行：

- `src.dataset.preprocess.structProbe_hyper`
- `src.dataset.preprocess.hitab_hyper`
- `src.dataset.preprocess.wtq_hyper`
- `src.dataset.preprocess.wikisql_hyper`
- `src.dataset.preprocess.fetaqa_hyper`

### 预处理做了什么

每个数据集大致会生成：

- `nodes/*.csv`
- `hyperedges/*.csv`
- `edges/*.csv`
- `graphs/*.pt`

其中 `graphs/*.pt` 是训练和推理实际会用到的内容。

### 预处理依赖哪些模型

预处理会调用 [`src/utils/lm_modeling.py`](/Users/zzx/02Research/PaperA/TAMO/src/utils/lm_modeling.py) 中的 `sbert`：

- 本地路径固定为 `./models/sentence-transformers/all-roberta-large-v1`

所以如果这个模型没下好，预处理会失败。

### 重要坑 2：`wtq_permute` 默认不会生成

仓库脚本里 [`src/dataset/preprocess/wtq_hyper.py`](/Users/zzx/02Research/PaperA/TAMO/src/dataset/preprocess/wtq_hyper.py) 的最后一行是：

```python
# random_dataset()
```

也就是说，默认执行 `bash script/data_preprocess.sh` 后：

- `structProbe_permute` 会生成
- `wtq_permute` 不会生成

但部分训练脚本和推理脚本会用到 `--second_dataset wtq_permute`。如果你要跑这些命令，需要手动生成 `wtq_permute`。

### 生成 `wtq_permute` 的办法

把 [`src/dataset/preprocess/wtq_hyper.py`](/Users/zzx/02Research/PaperA/TAMO/src/dataset/preprocess/wtq_hyper.py) 末尾改成：

```python
if __name__ == '__main__':
    step_one()
    step_two()
    random_dataset()
```

然后重新执行：

```bash
python -m src.dataset.preprocess.wtq_hyper
```

如果你只是想先把项目在服务器上跑通，不需要鲁棒性实验，可以先不要使用任何 `wtq_permute` 相关命令。

### 预处理完成后建议检查

以 StructProbe 为例：

```bash
find dataset/structProbe/train -maxdepth 2 -type d | sort
ls dataset/structProbe/train/graphs | head
```

以 WikiSQL 为例：

```bash
ls dataset/wikisql/train/graphs | head
ls dataset/wikisql/answers.json
```

`answers.json` 也是 WikiSQL 运行必需文件，由预处理脚本生成。

## 10. 设置运行环境变量

推荐在每次运行前先设置：

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
```

说明：

- `CUDA_VISIBLE_DEVICES=0`：先固定用单卡，排查问题最简单。
- `WANDB_MODE=offline`：避免服务器无法联网时 `wandb` 卡住。
- `TOKENIZERS_PARALLELISM=false`：减少 tokenizer 并行警告。

如果你完全不想启用 wandb，也可以：

```bash
export WANDB_DISABLED=true
```

## 11. 服务器自检

仓库已经提供了正式的自检脚本：

- [`server_smoke_test.py`](/Users/zzx/02Research/PaperA/TAMO/server_smoke_test.py)
- [`server_smoke_test.sh`](/Users/zzx/02Research/PaperA/TAMO/server_smoke_test.sh)

### 先做不加载大模型的自检

```bash
SMOKE_SKIP_MODEL=1 bash server_smoke_test.sh
```

这一步会检查：

- 核心依赖导入
- CUDA 是否可见
- PyG 扩展是否可用
- 数据集是否能从本地读取
- 图文件目录是否存在

### 再做完整自检

```bash
bash server_smoke_test.sh
```

默认等价于：

```bash
SMOKE_MODEL_NAME=7b
SMOKE_PROMPT_TYPE=llama2
SMOKE_DATASET=structprobe
python server_smoke_test.py
```

### 切换到 Mistral 自检

```bash
SMOKE_MODEL_NAME=mistral_7b \
SMOKE_PROMPT_TYPE=mistral \
SMOKE_DATASET=wtq_orig \
bash server_smoke_test.sh
```

如果这一步通过，说明服务器上的基础运行链路已经通了。

## 12. 最小可运行推理

### 12.1 StructProbe + Llama2 chat 推理

在修复完 `7b_chat` 目录名后执行：

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python inference.py \
  --dataset structprobe \
  --second_dataset structprobe_permute \
  --prompt_type llama2 \
  --model_name llm \
  --llm_model_name 7b_chat \
  --seed 42 \
  --project structprobe \
  --max_txt_len 1024 \
  --max_new_tokens 128 \
  --output_dir output/llama2-infer \
  --llm_num_virtual_tokens 8 \
  --patience 3
```

输出会落到：

```text
output/llama2-infer/structprobe/
output/llama2-infer/structprobe_permute/
```

### 12.2 WTQ + TableLlama 推理

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python inference.py \
  --dataset wtq_orig \
  --prompt_type tablellama \
  --model_name llm \
  --llm_model_name table_llama_7b \
  --seed 42 \
  --project wtq_orig \
  --max_txt_len 1024 \
  --max_new_tokens 128 \
  --output_dir output/tablellama-infer \
  --llm_num_virtual_tokens 8 \
  --patience 3
```

### 12.3 WTQ + Mistral 推理

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python inference.py \
  --dataset wtq_orig \
  --prompt_type mistral \
  --model_name mistral \
  --llm_model_name mistral_7b_instruct \
  --seed 42 \
  --project wtq_orig \
  --max_txt_len 1024 \
  --max_new_tokens 128 \
  --output_dir output/mistral-infer \
  --llm_num_virtual_tokens 8 \
  --patience 3
```

## 13. 最小可运行训练

### 13.1 先跑纯文本 Prompt Tuning

这是比 TAMO 超图模型更容易先跑通的一条线。

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python table_train.py \
  --dataset structprobe \
  --second_dataset structprobe_permute \
  --prompt_type llama2 \
  --model_name pt_llm \
  --llm_model_name 7b \
  --seed 42 \
  --project structprobe \
  --max_txt_len 1024 \
  --max_new_tokens 128 \
  --output_dir output/llama2-pt \
  --llm_num_virtual_tokens 8 \
  --patience 3
```

### 13.2 跑 TAMO 超图版本

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

python table_train.py \
  --dataset structprobe \
  --second_dataset structprobe_permute \
  --prompt_type llama2 \
  --model_name table_hypergraph_llm \
  --llm_model_name 7b \
  --seed 42 \
  --project structprobe \
  --max_txt_len 1024 \
  --max_new_tokens 128 \
  --output_dir output/llama2-tamo \
  --llm_num_virtual_tokens 8 \
  --gnn_model_name hyper \
  --gnn_num_layers 1 \
  --patience 3
```

### 13.3 训练产物位置

checkpoint 和结果通常会出现在：

```text
output/<run-name>/<dataset>/
```

包括：

- `*_checkpoint_best.pth`
- 预测结果 `.csv` 实际上是按 JSONL 写的
- `score.txt`

checkpoint 文件名规则见 [`src/utils/ckpt.py`](/Users/zzx/02Research/PaperA/TAMO/src/utils/ckpt.py)。

## 14. 用已训练 checkpoint 做推理

`inference.py` 支持通过 `--llm_ckpt_path` 加载 checkpoint：

```bash
python inference.py \
  --dataset structprobe \
  --prompt_type llama2 \
  --model_name table_hypergraph_llm \
  --llm_model_name 7b \
  --llm_ckpt_path output/llama2-tamo/structprobe/<your_checkpoint>.pth \
  --seed 42 \
  --project structprobe_eval \
  --max_txt_len 1024 \
  --max_new_tokens 128 \
  --output_dir output/llama2-tamo-eval \
  --gnn_model_name hyper \
  --gnn_num_layers 1 \
  --patience 3
```

## 15. 典型部署流程建议

如果你的目标是“尽快确认服务器可跑”，建议按这个顺序：

1. 克隆代码到服务器
2. 创建 `Python 3.9` 环境
3. 执行 `bash requirements.sh`
4. 下载所有模型
5. 修正 `Llama-2-7b-chat-hf` 目录名
6. 执行 `python download_dataset.py`
7. 执行 `bash script/data_preprocess.sh`
8. 先跑 `SMOKE_SKIP_MODEL=1 bash server_smoke_test.sh`
9. 再跑 `bash server_smoke_test.sh`
10. 最后跑一次最小推理命令

## 16. 常见问题排查

### 问题 1：`model path not found`

先检查：

```bash
find models -maxdepth 3 -type d | sort
```

大概率原因：

- 模型没下载完
- `7b_chat` 目录名没修正
- 你不是在仓库根目录执行命令
- 你改过 `global_path`，但目录没同步调整

### 问题 2：`dataset path not found`

检查：

```bash
find dataset -maxdepth 3 -type d | sort
```

大概率原因：

- 没执行 `python download_dataset.py`
- 运行目录不对
- `global_path` 和实际数据路径不一致

### 问题 3：`graphs/*.pt` 不存在

说明你没有完成预处理。执行：

```bash
bash script/data_preprocess.sh
```

然后确认：

```bash
ls dataset/structProbe/train/graphs | head
```

### 问题 4：PyG 扩展导入失败

如果 `torch_scatter` / `torch_sparse` 导入失败，一般是：

- PyTorch 版本和 PyG wheel 不匹配
- CUDA 版本不匹配

最稳妥的做法是：

- 重新建一个干净环境
- 直接按 [`requirements.sh`](/Users/zzx/02Research/PaperA/TAMO/requirements.sh) 的顺序安装

### 问题 5：`wandb` 卡住或报网络错误

运行前加：

```bash
export WANDB_MODE=offline
```

或者：

```bash
export WANDB_DISABLED=true
```

### 问题 6：显存不够

优先尝试：

- 减小 `--batch_size`
- 减小 `--eval_batch_size`
- 减小 `--max_txt_len`
- 单卡先跑
- 先用 `llm` 或 `pt_llm` 路径排查，不要一上来就跑超图模型

### 问题 7：`wtq_permute` 不存在

这是仓库当前逻辑导致的，不是你机器的问题。按上文“重要坑 2”手动生成即可。

## 17. 一套可直接复制的从零部署命令

```bash
cd /data
git clone <your-repo-url> TAMO
cd TAMO

conda create -n TAMO python=3.9 -y
conda activate TAMO

bash requirements.sh

huggingface-cli login
bash download_models.sh
mv ./models/meta-llama/Llama-2-7b-chat ./models/meta-llama/Llama-2-7b-chat-hf

python download_dataset.py
bash script/data_preprocess.sh

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

SMOKE_SKIP_MODEL=1 bash server_smoke_test.sh
bash server_smoke_test.sh

python inference.py \
  --dataset structprobe \
  --second_dataset structprobe_permute \
  --prompt_type llama2 \
  --model_name llm \
  --llm_model_name 7b_chat \
  --seed 42 \
  --project structprobe \
  --max_txt_len 1024 \
  --max_new_tokens 128 \
  --output_dir output/llama2-infer \
  --llm_num_virtual_tokens 8 \
  --patience 3
```

## 18. 当前文档未覆盖的内容

以下内容当前仓库里没有成熟成品，因此本文不展开：

- 对外 HTTP API 部署
- Web 页面部署
- Docker 化部署
- systemd 守护进程配置
- 多用户共享模型缓存策略

如果你的目标不是“在服务器上手工运行训练/推理”，而是“封装成长期在线服务”，需要在这个仓库外再补一层服务化包装。
