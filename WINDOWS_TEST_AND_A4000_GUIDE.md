# MaSS13K 开发分支：Windows 本机测试与 A4000 正式训练指南

检查日期：2026-11-22。本文针对本机 Windows 电脑熟悉流程、验证运行，以及之后在 A4000 上正式训练的场景。

## 1. 检查结论与范围

**开发分支的 PV 二分类流程已有原生 Windows 适配，代码和依赖发布情况支持在 Windows 上运行；本次没有完成依赖安装、CUDA 算子或模型运行实测，不能把它当作已通过 Windows 训练验证。** 请按第 6、7 节自行执行验收。

仓库实际分支名是 **`develop`**，没有名为 `development` 的本地分支或本地远程跟踪分支。本次检查的提交是：

```text
cdcab4513d90c7780de639780a50f3402dd3a892  适配windows
```

本地 `develop` 与 `origin/develop` 指向上述同一提交；这里是本地记录，未重新拉取远程分支。

| 范围 | 检查结果 |
| --- | --- |
| PV 训练、续跑、测试入口 `mmsegmentation/tools/pv.py` | 使用 Python、相对路径解析与子进程启动，不依赖 Bash |
| PV 运行配置 | 覆盖通用配置的 `fork`/`nccl`，使用 Windows 支持的 `spawn`/`gloo`；默认单 GPU 启动 |
| Windows 安装入口 | 已有 `mmsegmentation/setup_windows.ps1`，锁定 Python 3.10 对应的一组依赖 |
| MMCV Windows 二进制包 | 官方索引提供 `mmcv-2.1.0-cp310-cp310-win_amd64.whl`，对应 CUDA 12.1 / PyTorch 2.1 |
| 本地数据、原始权重 | 文件存在，已统计 image/label 数量；本次未逐文件解码、检查标签像素或验证权重内容 |
| 真实 GPU 前向、反向、续跑、评估 | 本次未执行，按第 6、7 节验收 |
| 原论文七分类 benchmark | 保留了 Linux 路径、`SyncBN`、`fork`/`nccl` 和 Bash 分布式启动方式，不能直接照原 README 命令在 Windows 运行 |

MMCV 的 Windows wheel 已从[官方索引](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html)核对。下文完整操作以开发分支的 **General 光伏二分类** 为主，不把七分类源权重直接当作已训练好的 PV 模型。

### 当前电脑已经发生的环境变更

本次检查期间已创建 `massformer` Conda 环境（Python 3.10.22），并安装 pip 24.3.1、setuptools 75.3.2。之后按要求停止下载，未完成 PyTorch 和项目依赖安装。下文是在告诉你后续怎么操作，不代表已执行。

因此，这台电脑通常可从 `conda activate massformer` 继续；以你执行 `conda env list` 的结果为准。现有 `aienv` 未用于安装项目依赖。

## 2. 本机配置与两台机器的分工

| 项目 | 本机读取结果 | 对使用流程的影响 |
| --- | --- | --- |
| 型号 | ASUS TUF Gaming F16 FX607JV_FX607JV | 接电使用，保持散热，避免测试中休眠 |
| CPU | Intel Core i9-13980HX，24 核 / 32 线程 | 无需按线程数增大 DataLoader workers |
| 内存 | 16 GB | 本机先用 workers=0 熟悉流程，再用 workers=2 验证 spawn |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU | 用于算子、少量训练和推理验证 |
| 显存 | 8188 MiB，约 8 GB | 是否容纳 512×512、batch=2、FP32，须实际试运行；未测量峰值 |
| NVIDIA 驱动 | 566.07 | 版本条件支持本项目选择的 CUDA 12.1 运行时 |
| 系统 | Windows 11 家庭版中文版，64 位，版本 10.0.26200 | 使用原生 PowerShell / Anaconda Prompt |
| C 盘剩余空间 | 检查时约 192 GiB | 随使用变化；环境、下载缓存、断点和日志均会占空间 |
| Conda 程序 | `C:\ZZworkspace\tools\anaconda3\Scripts\conda.exe` | 已安装，无需再装一套 Anaconda |

`nvidia-smi` 显示的 CUDA 12.7 是驱动支持能力，不等于项目必须安装 CUDA 12.7。这里固定使用 PyTorch cu121 wheel；按二进制安装路线通常无需另装 CUDA Toolkit、cuDNN 或 Visual Studio 编译工具。驱动兼容关系参见 [NVIDIA CUDA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)。

| 用途 | 本机 RTX 4060 Laptop | 正式训练 A4000 |
| --- | --- | --- |
| 数据量 | 小规模流程验证；可另做全量文件检查 | `General/train`、`General/val`、`General/test` 完整划分 |
| 分辨率、精度、batch | 512×512、FP32、batch=2；显存不足时先排除资源占用 | 保持正式实验的 512×512、FP32、batch=2 |
| workers | 先 0，之后 2 | 默认 2；根据目标机内存与磁盘情况调整 |
| 训练长度 | 内置 smoke，或第 7 节的 2 epoch、4 张训练图 | 默认 50 epoch |
| 输出位置 | `work_dirs/windows_trial` 或临时 smoke 目录 | `work_dirs/massformer_pv_general_pretrained` |
| 结果用途 | 熟悉安装、训练、续跑、测试、可视化 | 正式模型与实验结果 |

本次没有读取 A4000 所在机器的系统、内存、显存和驱动。目标机仍应独立执行环境与算子检查，不用本机测试推算正式训练耗时。

## 3. 代码、数据与权重目录

本机项目目录：

```text
C:/ZZworkspace/zyworkspace/project/
├── codes/dis/MaSS13K/
│   ├── WINDOWS_TEST_AND_A4000_GUIDE.md
│   ├── WINDOWS_PV.md
│   └── mmsegmentation/
│       ├── setup_windows.ps1
│       ├── pv_settings.example.json
│       ├── tools/pv.py
│       └── work_dirs/
├── datasets/dis/General/
│   ├── train/image/*.tif
│   ├── train/label/*.tif
│   ├── val/image/*.tif
│   ├── val/label/*.tif
│   ├── test/image/*.tif
│   └── test/label/*.tif
└── weights/cv/MaSS13K/iter_80000.pth
```

已统计的本地文件数量：

| 划分 | image 数量 | label 数量 |
| --- | ---: | ---: |
| train | 17648 | 17648 |
| val | 1878 | 1878 |
| test | 4802 | 4802 |

原始权重文件大小为 192189849 字节。存在与大小检查不等于权重能正确加载，加载由后续小规模试运行验证。

PV 入口要求：图像为 512×512、三通道 uint8 TIFF；标签为同尺寸单通道 uint8 TIFF，像素只能是 `0=背景`、`1=光伏`，不能用 0/255 标签替代。每一对 image/label 文件名相同，扩展名使用 `.tif`。三个划分的文件名交集也会被检查，但文件名无交集不代表做过内容去重或地理来源泄漏检查。

General 光伏数据需要使用你自己的已有数据；论文提供的 MaSS13K 数据集不能直接替换成这一二分类目录。原始论文数据与权重的[下载链接](https://drive.google.com/drive/folders/1NooQARLS4epj6GTsf_6hzphl1APUdW7t?usp=sharing)来自仓库 README；本次未核实云端文件内容。本机上述数据和权重已经存在。

在 PowerShell 中检查代码位置与分支：

```powershell
Set-Location C:\ZZworkspace\zyworkspace\project\codes\dis\MaSS13K
git status --short
git branch --show-current
git log -1 --oneline
# 只有当前不在 develop 且工作区允许切换时才执行：
git switch develop
Set-Location .\mmsegmentation
```

**除特别注明外，后续命令都在 `MaSS13K/mmsegmentation` 目录执行。** Windows 终端使用 PowerShell 语法，不复制 Bash 的 `\` 换行符。

## 4. 使用 Conda 配置项目环境

### 4.1 激活或创建 Python 3.10 环境

推荐从开始菜单打开 **Anaconda Prompt**，输入：

```text
conda env list
```

环境已存在时：

```text
conda activate massformer
```

新电脑、或者尚无该环境时，先创建再激活：

```text
conda create -n massformer python=3.10 -y
conda activate massformer
```

确认实际解释器，输出应位于 `.../envs/massformer/python.exe`，版本为 3.10.x：

```text
python --version
python -c "import sys; print(sys.executable)"
python -m pip --version
```

本机环境通常位于 `C:\Users\23792\.conda\envs\massformer`。不在 base、aienv 或其他任务的环境中安装这套项目依赖。

如果想使用普通 PowerShell，可先在 Anaconda Prompt 中运行 `conda init powershell`，然后关闭并重新打开 PowerShell。若 PowerShell 提示禁止加载 profile，先使用 Anaconda Prompt 完成操作；安装脚本本身可通过下面的单次 `ExecutionPolicy Bypass` 调用，无需永久修改系统策略。Conda 的安装方式参见[官方 Windows 文档](https://docs.conda.io/projects/conda/en/stable/user-guide/install/windows.html)。

### 4.2 执行仓库安装脚本

下面命令适用于已激活 `massformer` 的 PowerShell：

```powershell
Set-Location C:\ZZworkspace\zyworkspace\project\codes\dis\MaSS13K\mmsegmentation
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

如果留在 Anaconda Prompt（cmd），先切换目录再执行同一个脚本：

```bat
cd /d C:\ZZworkspace\zyworkspace\project\codes\dis\MaSS13K\mmsegmentation
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

这是需要联网下载的步骤。PyTorch Windows CUDA wheel 本身约 2.47 GB，安装后的环境与缓存还会占更多空间；建议为环境、缓存和短流程输出预留至少 15–20 GB，不含数据集和正式训练输出。

脚本依次安装构建工具、CUDA PyTorch、项目依赖、MMCV wheel，再 editable 安装当前仓库的 MMSegmentation，最后执行 `pip check`。安装出错时脚本会停止，先解决所在阶段的错误，再继续。

关键版本如下，完整直接依赖以 `requirements/pv-windows.txt` 为准：

| 组件 | 固定版本 |
| --- | --- |
| Python | 3.10.x |
| pip / setuptools | 24.3.1 / 75.3.2 |
| PyTorch / torchvision | 2.1.0+cu121 / 0.16.0+cu121 |
| MMCV | 2.1.0，包含 CUDA 扩展 |
| MMEngine / MMDetection | 0.10.3 / 3.3.0 |
| NumPy / OpenCV | 1.26.4 / 4.11.0.86 |
| SciPy / Matplotlib | 1.14.1 / 3.9.4 |
| TensorBoard | 2.18.0 |
| MMSegmentation | editable 安装本仓库修改版 |

PyTorch cu121 安装组合可在[官方历史版本页面](https://pytorch.org/get-started/previous-versions/#v210)核对。MMDetection 3.3.0 要求 MMCV 小于 2.2.0，见[版本检查源码](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/__init__.py)。不要单独升级 torch、MMCV 或 NumPy，也不要用 `mmcv-lite` 替代；模型需要 CUDA 可变形卷积，[MMCV 官方说明](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html)明确区分了两者。

### 4.3 手工分步安装：用于定位脚本错误

如果脚本已成功，不需要再执行这一节。需要分步排查时，在相同环境、相同工作目录逐条运行；每条成功后再运行下一条：

```powershell
python -m pip install "pip==24.3.1" "setuptools==75.3.2" wheel
python -m pip install "torch==2.1.0" "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/pv-windows.txt
python -m pip install "mmcv==2.1.0" --only-binary=mmcv -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

MMCV 的下载应是 `cp310-cp310-win_amd64.whl`。如果转为编译 `.tar.gz` 或找不到匹配包，先核对 Python 3.10、64 位、PyTorch 2.1.0/cu121 和网络；保留二进制限制，不要为了继续安装而改用 lite 版。

## 5. 设置本地路径

目录层级与第 3 节一致时，默认路径即可使用。建议为本机建立本地设置文件：

```powershell
Copy-Item pv_settings.example.json pv_settings.local.json
notepad pv_settings.local.json
```

若本地文件已经存在，直接编辑，不要重复覆盖。可使用以下设置：

```json
{
  "data_root": "C:/ZZworkspace/zyworkspace/project/datasets/dis/General",
  "pretrained_checkpoint": "C:/ZZworkspace/zyworkspace/project/weights/cv/MaSS13K/iter_80000.pth",
  "work_dir": "work_dirs/windows_local_practice",
  "batch_size": 2,
  "num_workers": 0,
  "epochs": 50,
  "seed": 42,
  "test_images": 8,
  "sample_names": []
}
```

`windows_local_practice` 只是本机独立输出目录名；这份设置的 epoch 仍为 50，**直接运行 `pv.py train` 会在全量数据上训练**。本机练习优先使用第 6 节 smoke 或第 7 节限定样本的短流程。

路径在 JSON 中优先用 `/`；使用 `\` 时必须写成 `\\`。相对路径以 `mmsegmentation` 为基准。设置优先级为命令行 > 指定/本地 JSON > 内置默认值。`pv_settings.local.json` 已被 Git 忽略，迁移到 A4000 时单独配置。

## 6. 本机 Windows 验收

### 6.1 检查依赖、解释器和 GPU

```powershell
python -m pip check
python -c "import torch, torchvision, mmcv, mmengine, mmdet, mmseg; print(torch.__version__, torchvision.__version__, mmcv.__version__, mmengine.__version__, mmdet.__version__); print(mmseg.__file__); print('CUDA:', torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
nvidia-smi
```

应看到 CUDA 可用、RTX 4060 Laptop、上述固定版本；`mmseg.__file__` 应指向当前仓库。`pip check` 应输出 `No broken requirements found`。导入成功还不能证明 CUDA 扩展可运行，继续下一步。

### 6.2 完整文件检查与真实算子检查

```powershell
python tools/pv.py check --full-data --num-workers 0
```

该命令会逐文件解码检查 image/label、尺寸、类型和标签值，检查源权重路径，执行 GPU 可变形卷积的前向/反向以及 `point_sample`。GPU 检查成功会输出：

```text
CUDA deformable convolution forward/backward and point_sample: PASS
```

仅排查数据/配置时可执行：

```powershell
python tools/pv.py check --full-data --skip-gpu --num-workers 0
```

`--skip-gpu` 仍需要安装配置解析相关依赖，只是跳过 GPU 检查；它不代表 GPU 训练已通过。

### 6.3 内置最小训练、续跑和测试

先关闭不需要的 GPU 应用，然后执行：

```powershell
python tools/pv.py check --smoke --epochs 2 --num-workers 0
```

默认 batch=2 时，仅使用 4 张训练图、2 张验证图和 2 张测试图，分两次训练进程完成 2 epoch，并检查完整断点、续跑计数、测试指标和 TensorBoard 写入。仍使用真实 512×512 图像、原始权重和 FP32 计算。

成功标志是末尾的 `SMOKE PASS`，且进程正常退出。内部输出使用临时目录，退出后清理；它适合验证，不保留可供之后浏览的训练结果。临时断点需要约 2 GB 空间，建议多留余量。

若还需要确认 Windows 多进程加载，用下面命令再验证 workers=2：

```powershell
python tools/pv.py check --smoke --epochs 2 --num-workers 2
```

workers=0 的成功只覆盖单进程数据加载。workers=2 的成功才覆盖这里配置的 spawn 数据加载路径。

**本机 8 GB 显存是否足够以实际结果为准。** 出现 OOM 时先退出相关训练进程，关闭额外 GPU 应用，确认显存释放，再重试。workers=0 主要减少系统内存和多进程开销，不是降低模型显存占用的办法。当前解码器包含全局池化后的 BatchNorm，入口明确禁止训练 batch=1；也不要为了本机验证直接修改 A4000 正式配置或开启未经验证的 AMP。若清理资源后仍 OOM，记录已通过的算子/推理检查，把训练验收留到 A4000，不能把 OOM 写成 Windows 算子不支持。

## 7. 保留输出的本机短流程：练习训练、续跑与测试

想实际打开断点、指标、预测图和 TensorBoard，可使用本节替代内置 smoke 的流程练习。数据仍引用原始 General 目录，只将配置里的样本数量限制为 4/2/2，不复制或改动数据。

### 7.1 生成独立的短流程配置

在激活 `massformer` 的 **PowerShell**、`mmsegmentation` 目录执行以下整段命令。它只生成配置；后面才启动训练。

```powershell
@'
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / 'tools'))
import pv

values = pv.settings(argparse.Namespace(
    settings=None, work_dir='work_dirs/windows_trial', epochs=2,
    batch_size=2, num_workers=0, test_images=2))
output = Path(values['work_dir'])
if list(output.glob('epoch_*.pth')):
    raise RuntimeError('Trial already has checkpoints. Skip generation and resume, or use a new directory.')
output.mkdir(parents=True, exist_ok=True)

cfg = pv.build_config(values)
cfg.train_dataloader.dataset.indices = 4
cfg.val_dataloader.dataset.indices = 2
cfg.test_dataloader.dataset.indices = 2
cfg.train_cfg.val_interval = 1
cfg.train_cfg.dynamic_intervals = []
cfg.fixed_train_dataset.indices = 4
cfg.custom_hooks[0].dataset.indices = 4
cfg.custom_hooks[0].num_samples = 2
cfg.custom_hooks[0].sample_names = None
cfg.custom_hooks[0].interval = 1
cfg.dump(str(output / 'train_config.py'))
cfg.train_cfg.max_epochs = 1
cfg.dump(str(output / 'first_epoch_config.py'))

test = pv.build_config(values, testing=True)
test.load_from = str(output / 'epoch_2.pth')
test.test_dataloader.dataset.indices = 2
test.dump(str(output / 'test_config.py'))
print('Prepared:', output)
'@ | python -
```

该片段依赖已安装的项目环境，本次未执行验证。生成配置使用正式 PV 配置作为基础，覆盖样本数、输出目录、2 epoch、每 epoch 验证和本机 workers=0。要专门练习 spawn，可在生成前把片段中的 `num_workers=0` 改为 `2`。

### 7.2 先训练 1 epoch，再从断点续跑到 2 epoch

```powershell
python tools/train.py work_dirs/windows_trial/first_epoch_config.py
```

成功后应出现 `work_dirs/windows_trial/epoch_1.pth` 和 `last_checkpoint`。随后使用原来总计 2 epoch 的配置续跑：

```powershell
python tools/train.py work_dirs/windows_trial/train_config.py --resume
```

第二条命令恢复模型、优化器、调度器与 epoch/iteration，最终应生成 `epoch_2.pth`。不要在这套短流程上运行 `pv.py train --resume --work-dir work_dirs/windows_trial`：通用 PV 入口会重新生成全量训练配置，不能用于续跑这个限定样本的练习。

### 7.3 测试 2 张图并打开 TensorBoard

```powershell
python tools/test.py work_dirs/windows_trial/test_config.py work_dirs/windows_trial/epoch_2.pth
python -m tensorboard.main --logdir work_dirs/windows_trial --host localhost --port 6006
```

打开 `http://localhost:6006`。需要停止 TensorBoard 时在其终端按 `Ctrl+C`。端口占用时改为 6007。训练或测试运行时也可以在另一个激活相同环境的终端启动 TensorBoard。

检查以下输出：

| 输出 | 应看到的内容 |
| --- | --- |
| `work_dirs/windows_trial/epoch_1.pth`、`epoch_2.pth` | 两次完整断点 |
| `work_dirs/windows_trial/test/metrics.json` | `images` 为 2，保存总体和每类指标 |
| `work_dirs/windows_trial/test/per_image_metrics.csv` | 两张测试图的指标 |
| `work_dirs/windows_trial/test/pred_masks/*.png` | 两张原始 0/1 预测标签 |
| 时间戳目录下 `vis_data/events.out.tfevents.*` | TensorBoard 事件 |

TensorBoard 的 Scalars 可看 loss、学习率和验证/测试指标，Images 可看固定训练样本与测试对比图。此短流程仅用于跑通操作，2 epoch、4 张训练图的精度不能作为正式成绩。

## 8. A4000 上正式训练

### 8.1 迁移准备

复制代码、完整数据集和原始 `iter_80000.pth`；目标机重新创建 Conda 环境，不复制整套环境文件夹。本机练习结果保留在 `windows_trial`，正式训练另起目录，并从原始七分类权重做二分类全参数微调。

如果 A4000 机器是 Windows，按第 4 节安装，将路径改为目标机实际位置。如果是 Linux，在 Python 3.10 的 Conda 环境中使用第 4.3 节同样的逐条 pip 命令即可；`requirements/pv-windows.txt` 中这组包可用于该安装路线，MMCV 索引会按平台选择 Linux wheel，不运行 `.ps1`。

目标机先查看 `nvidia-smi`，确认 GPU/驱动与剩余显存，再进行下面的检查。本次未在 A4000 上安装或测试。

### 8.2 A4000 设置示例

在目标机的 `mmsegmentation/pv_settings.local.json` 中配置。保持第 3 节目录层级时可用以下相对路径；否则改成目标机自己的绝对路径：

```json
{
  "data_root": "../../../../datasets/dis/General",
  "pretrained_checkpoint": "../../../../weights/cv/MaSS13K/iter_80000.pth",
  "work_dir": "work_dirs/massformer_pv_general_pretrained",
  "batch_size": 2,
  "num_workers": 2,
  "epochs": 50,
  "seed": 42,
  "test_images": 64,
  "sample_names": []
}
```

正式参数为 FP32、50 epoch、batch=2、seed=42；AdamW 解码器学习率 `1e-4`，主干 `1e-5`，首 epoch 预热后余弦衰减。训练集使用 `train`，每 5 epoch 在 `val` 验证，末 epoch 也验证；`test` 用于最后独立测试。

不因 A4000 显存更大自动扩大 batch 或学习率。保留现有边缘损失与主干 BatchNorm 设置，训练 `drop_last=True` 防止最后一批只有 1 张。

### 8.3 验收、开始训练和续跑

```text
conda activate massformer
```

进入目标机的 `mmsegmentation` 后，依次执行：

```powershell
python tools/pv.py check --full-data --smoke
python tools/pv.py train
```

第一条会全量检查数据并执行隔离的小规模运行，成功后再开始正式训练。它不会先执行 50 epoch 的全量训练。

每个 epoch 保存完整断点（含优化器、调度器），仅保留最近两个；默认不保存独立的 best 模型。中断后继续：

```powershell
python tools/pv.py train --resume
# 或指定当前正式实验中的断点：
python tools/pv.py train --resume work_dirs/massformer_pv_general_pretrained/epoch_25.pth
```

续跑是继续到原定总计 50 epoch，不是额外再训练 50 epoch。保持数据内容、batch、总 epoch、seed 不变；入口会检查已有记录中的 batch/epoch/seed。只调整机器相关的路径与 workers。Ctrl+C 后，以实际成功落盘的最近断点恢复。

`iter_80000.pth` 是初始化权重，不能以 `--resume` 恢复为 PV 实验。首次从七分类转二分类时，分类层 weight/bias 尺寸不匹配并重新初始化是预期现象；其余兼容参数加载。若恢复已有 PV 二分类断点仍出现分类层尺寸不匹配，需要核对配置与权重路径。

重新开始一个正式实验时，使用新目录：

```powershell
python tools/pv.py train --work-dir work_dirs/pv_a4000_run2
```

旧 `work_dirs/massformer_pv_pretrained_50ep` 对应旧划分，不当作当前 General/train-val-test 划分的断点或成绩。

### 8.4 全量独立测试与结果查看

```powershell
python tools/pv.py test
# 或明确指定某个正式断点：
python tools/pv.py test --checkpoint work_dirs/massformer_pv_general_pretrained/epoch_50.pth --test-images 64
python tools/pv.py tensorboard
```

默认选择正式输出目录的最近断点，测试全部 4802 张图并导出全部预测标签。`--test-images 64` 只限制写入 TensorBoard 的对比图数量，不限制测试样本数。`--test-images 0` 不写对比图，`--test-images all` 写全部对比图。重复测试会更新同名指标文件并新增事件会话，需保留不同结果时另行归档测试目录。

正式输出位置：

| 相对正式输出目录的路径 | 用途 |
| --- | --- |
| `epoch_N.pth`、`last_checkpoint` | 最近两个完整断点及最新断点位置 |
| `train_config.py`、`pv_settings.resolved.json` | 实际训练配置与解析后的设置 |
| `console.log`、时间戳目录下 `.log` / `.json` | 控制台及框架日志 |
| 时间戳目录下 `vis_data/events.out.tfevents.*` | 训练、验证曲线与固定训练图 |
| `fixed_train_samples/samples.json` | 固定训练图路径清单 |
| `test/metrics.json` | 整体、每类指标和混淆矩阵 |
| `test/per_image_metrics.csv` | 全部测试图的 PV 指标 |
| `test/pred_masks/*.png` | 原始 0/1 预测 mask |
| `test/时间戳/vis_data/events.out.tfevents.*` | 测试指标与选中对比图 |

指标单位为百分数；mIoU/mDice 是两类均值，同时关注 `pv` 类指标。0/1 PNG 用普通看图软件打开时可能几乎全黑，这是原始标签的数值显示问题；直观比较优先看 TensorBoard 对比图。测试不计算训练 loss，也不用于选择最优模型。

## 9. 单张无标签图片推理

`pv.py test` 需要配对标签以计算指标。对一张新图做预测可使用已有 demo，配置与 checkpoint 必须匹配。

例如在本机短流程结束后：

```powershell
New-Item -ItemType Directory -Force work_dirs/windows_trial/demo | Out-Null
$sampleImage = (Get-ChildItem C:/ZZworkspace/zyworkspace/project/datasets/dis/General/test/image -Filter *.tif -File | Select-Object -First 1).FullName
python demo/image_demo.py "$sampleImage" work_dirs/windows_trial/train_config.py work_dirs/windows_trial/epoch_2.pth --device cuda:0 --out-file work_dirs/windows_trial/demo/overlay.png
```

对自己的新图，将 `$sampleImage` 改成实际路径；先使用与 PV 数据一致的 512×512 三通道图像。此命令输出彩色叠加图，无需标签；它不导出整幅遥感 GeoTIFF，也不处理地理坐标或大图切片拼接。

正式使用时换成 A4000 正式训练生成的 `train_config.py` 和对应 `epoch_N.pth`。原始七分类 `iter_80000.pth` 与二分类配置不匹配，不能用它演示已经训练好的光伏检测效果。该 demo 流程本次未运行实测。

## 10. 常见问题

| 现象 | 处理方式 |
| --- | --- |
| `conda` 不存在或 `conda activate` 不生效 | 使用 Anaconda Prompt，核对 Conda 路径；PowerShell 可初始化后重开 |
| PowerShell profile / `.ps1` 被执行策略阻止 | 先使用 Anaconda Prompt；安装脚本用文中的 `-NoProfile -ExecutionPolicy Bypass` 单次调用 |
| `torch.cuda.is_available()` 为 False | 核对 `sys.executable`、torch 版本是否带 cu121、`nvidia-smi` 是否正常；不要靠安装系统 Toolkit 替代 CUDA PyTorch |
| `No module named mmcv._ext` / DLL load failed | 检查是否误装 lite、wheel 是否匹配 PyTorch/CUDA/Python；重建匹配环境，系统运行库缺失时按具体 DLL 报错补齐 |
| MMCV 开始源码编译 | 返回固定版本及官方 wheel 索引；保留 `--only-binary=mmcv` |
| NumPy ABI、`_ARRAY_API` 等错误 | 核对是否被升级至 NumPy 2，恢复锁定的 1.26.4 |
| `FileNotFoundError` | 核对 data/weight 路径、当前目录和 JSON；源权重与训练断点是不同用途 |
| TIFF 无法读取、形状/标签值不符合 | 用全量检查定位文件；检查通道、512×512、uint8、0/1 标签及 `.tif` 配对 |
| Windows DataLoader 卡住或 `WinError 1455` | 先 workers=0，减少后台程序，确认系统内存/分页文件空间；通过后再测试 workers=2 |
| CUDA OOM | 区分系统内存与显存；退出额外 GPU 应用重试。batch=1 不适用于当前训练入口，正式参数保持不变 |
| 单张样本 BatchNorm 错误 | 保持 batch≥2 和 `drop_last=True`；不要用 batch=1 规避显存问题 |
| 已有输出目录拒绝重新训练 | 正式实验用 `--resume`，新实验换 `--work-dir`；短流程按第 7 节原配置续跑 |
| TensorBoard 没数据 | 核对 logdir 指向正确实验目录；内置 smoke 清理后不保留事件，用第 7 节练习查看 |
| 6006 端口冲突 | TensorBoard 用 `--port 6007`，浏览器同步修改地址 |

## 11. 推荐执行顺序与验收记录

本机建议顺序：激活/创建环境 → 安装 → 版本与 GPU 检查 → 完整数据与算子检查 → 选内置 smoke 或保留输出的短流程 → 查看续跑、指标、预测与 TensorBoard → 需要时另测 workers=2。

| 验收项 | 通过条件 | 本次状态 |
| --- | --- | --- |
| 环境依赖 | `pip check` 无冲突，版本正确，mmseg 路径指向仓库 | 待你执行 |
| 数据 | 全量解码与标签检查通过，划分文件名无交集 | 仅完成数量统计 |
| GPU 扩展 | DCN 前向/反向、point_sample 显示 PASS | 待你执行 |
| 训练与续跑 | smoke PASS，或短流程从 epoch 1 正确恢复到 2 | 待你执行 |
| Windows spawn | workers=2 的少量训练/测试完成 | 待你执行 |
| 测试与可视化 | 测试样本数、mask、指标正确，TensorBoard 可查看 | 待你执行 |
| A4000 正式训练 | 在目标机独立通过检查后启动完整实验 | 待你执行 |

正式训练迁移到 A4000 后，重新配置目标机路径和环境、执行验收，再从原始权重开始新实验；本机练习断点不作为正式训练起点。
