# 原生 Windows：General 光伏二分类

适用于 PowerShell / Anaconda Prompt、单张 NVIDIA A4000，Python 3.10。
当前配置使用原始 MaSS13K 七分类权重进行二分类全参数微调，保留已修正的边缘损失。
两类为 `0=bg`、`1=pv`；分类器内部另有 no-object 输出，不是第三个语义类别。

## 1. 复制目录与权重

盘符不限，保持以下相对层级即可：

```text
zyworkspace/project/
├── codes/dis/MaSS13K/
│   ├── WINDOWS_PV.md
│   └── mmsegmentation/
│       ├── tools/pv.py
│       ├── setup_windows.ps1
│       └── pv_settings.example.json
├── datasets/dis/General/
│   ├── train/image/*.tif
│   ├── train/label/*.tif
│   ├── val/image/*.tif
│   ├── val/label/*.tif
│   ├── test/image/*.tif
│   └── test/label/*.tif
└── weights/cv/MaSS13K/iter_80000.pth
```

请单独复制权重和数据，Git 不包含这些大文件。不要复制旧电脑的 conda 环境目录，需在 Windows 重建。
旧 `work_dirs/massformer_pv_pretrained_50ep` 可以归档保留，但它使用了之前的 `val/train_old` 划分，
不可作为此次标准划分训练的断点或测试成绩。新实验从原始 `iter_80000.pth` 开始。

本次检查的数据：

| 用途 | 子目录 | 配对数量 |
| --- | --- | ---: |
| 训练 | train | 17648 |
| 验证 | val | 1878 |
| 独立测试 | test | 4802 |

图像为 512×512、三通道 uint8 TIFF，标签为相同尺寸的单通道 uint8 TIFF，值为 0/1。
同一对 image/label 文件名相同。已完整检查所有文件格式、标签值、配对和三个划分的文件名交集；
文件名互不重叠不代表已做内容去重或地理来源泄漏检查。

## 2. 安装 Windows 环境

先安装 Anaconda/Miniconda 和支持 CUDA 12.1 的 NVIDIA 驱动。在 PowerShell 中进入本地 `mmsegmentation` 目录：

```powershell
conda create -n massformer python=3.10 -y
conda activate massformer
cd D:\zyworkspace\project\codes\dis\MaSS13K\mmsegmentation
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

`D:` 仅为示例。Anaconda Prompt 也可执行上述 `powershell ...` 安装命令。
脚本使用当前 conda 环境的 Python，锁定 PyTorch 2.1.0/cu121、torchvision 0.16.0、
MMCV 2.1.0、MMEngine 0.10.3、MMDetection 3.3.0、NumPy 1.26.4，并以 editable 方式安装本仓库。
根目录 `requirements.txt` 是同一组依赖的 pip 入口；优先使用上面的脚本，便于定位安装阶段的错误。

MMCV 使用[官方 CUDA 12.1 / torch 2.1 wheel 索引](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html)，
其中提供 Python 3.10 Windows 的 MMCV 2.1.0 wheel。安装脚本要求二进制 wheel，下载失败时会停止。
本模型实际依赖 CUDA 可变形卷积，`mmcv-lite` 不适用；也不要将 MMCV 或 PyTorch 单独升级到最新版。
参见 [MMCV 安装文档](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html)。

## 3. 设置与环境检查

默认目录结构一致时无需创建设置文件。需要调整时：

```powershell
Copy-Item pv_settings.example.json pv_settings.local.json
```

编辑 `pv_settings.local.json`。相对路径均以 `mmsegmentation` 为基准，支持 JSON 中的正斜杠 Windows 路径，
例如 `D:/zyworkspace/project/datasets/dis/General`。命令行参数优先于 JSON；本地 JSON 已加入 Git 忽略。
`sample_names` 可填写固定展示的训练图文件名；空列表表示默认均匀选取 4 张。

```powershell
python tools/pv.py check --full-data --smoke
```

该命令先完整验证所有数据，再在真实 GPU 上运行可变形卷积前向/反向及采样算子，
最后仅用两个 batch 的训练图（默认 batch size 2 时为 4 张）、2 张验证图和 2 张测试图，
检查训练、断点续跑、评估、预测导出和 TensorBoard。
试运行使用临时目录，退出后清理，不写正式训练输出，也不表示模型精度已得到验证。
此步骤会在临时目录短暂保存两个包含优化器的权重，请留出约 2 GB 空间。

只检查文件和配置可用 `python tools/pv.py check --full-data --skip-gpu`。
CUDA 不可用时检查驱动、当前 conda 环境以及 PyTorch 是否为 CUDA 版本；不能跳过算子检查就认为 Windows 已可训练。
如 Windows DataLoader 启动失败，可先用 `--num-workers 0` 排查。

## 4. 训练与断点续跑

```powershell
python tools/pv.py train
```

默认 50 epoch、FP32、batch size 2、2 个数据加载进程、seed 42。
AdamW 解码器学习率 `1e-4`、主干 `1e-5`；首 epoch 预热，随后余弦衰减。
不自动扩大 batch 或学习率。主干 BatchNorm 固定统计量，训练丢弃不足一批的末尾样本，避免解码器 BatchNorm 遇到单样本。
数据加载使用 Windows 支持的 spawn；本入口为单 GPU，不使用 Linux shell / NCCL 分布式启动脚本。

每个 epoch 保存完整 checkpoint（含优化器与调度器），保留最近两个。
每 5 epoch 在 `val` 上评估，最后一个 epoch 也评估；不使用 `test` 挑选模型。
不自动启动 ResNet 对照或独立测试。

```powershell
python tools/pv.py train --resume
# 也可以指定此实验的完整 checkpoint
python tools/pv.py train --resume work_dirs/massformer_pv_general_pretrained/epoch_25.pth
```

续跑恢复已完成的 epoch、优化器和调度器，继续到原定总 epoch；不是在断点上额外训练 50 epoch。
最近一次尚未保存的 epoch 会重新计算。跨电脑复制整个新实验目录后，入口会处理 `last_checkpoint` 中过期的 Linux/Windows 路径。
保持数据内容、batch、epoch 总数、seed 不变；迁移时只调整数据/输出路径和 workers。

需要另一次实验时使用新目录，例如：

```powershell
python tools/pv.py train --work-dir work_dirs/pv_run2 --epochs 50
```

7 类源权重加载时，仅分类层 weight/bias 尺寸不匹配并保留新初始化，其余兼容参数加载。
这一提示在首次微调时正常；如果续跑二分类 checkpoint 仍有分类层尺寸不匹配，应停止并检查权重路径。
预计耗时应从 A4000 上正式训练日志的 `time`/`eta` 判断；当前电脑的耗时不能直接当作目标机估计。

## 5. 测试与 TensorBoard

训练完成后显式测试最新 checkpoint：

```powershell
python tools/pv.py test
# 指定 checkpoint 或图片展示数
python tools/pv.py test --checkpoint work_dirs/massformer_pv_general_pretrained/epoch_50.pth --test-images 64
python tools/pv.py tensorboard
```

浏览器打开 `http://localhost:6006`。端口冲突时使用 `--port 6007`。
TensorBoard 递归读取正式输出目录内的事件文件，同时显示训练、验证和独立测试。
默认测试全部 4802 张、导出全部原始 0/1 预测 PNG；仅比较图默认均匀选取 64 张写入事件文件。
`--test-images all` 展示全部、`--test-images 0` 只记指标。重复测试会新增一个事件日志会话；图片仅存事件文件中，
不会另外保存一套用于展示的对比 PNG。TensorBoard 打开页面时读取事件，不会重新运行预测。

Scalars 包括训练总 loss、各解码层分类/mask/dice loss、边缘 loss、学习率、耗时，以及验证 mIoU/mDice/mFscore 等。
测试为 `test/*` 整体指标、`test_class/bg/*` 和 `test_class/pv/*` 各类别指标、`test_samples/pv_*` 逐图指标。
测试通过预测计算 IoU/Dice 等，不计算训练 loss。指标单位为百分数，m 表示对背景与 PV 两类取算术平均，
`aAcc` 表示所有有效像素的总体正确率。

Images 中 `fixed_train__*` 为固定训练样本的原图、标注、当前预测和叠加图，step 为已完成 epoch（0、5、10……）。
`test_images/*` 为独立测试比较图，step 为 checkpoint epoch；逐图指标横轴为样本序号。
训练 loss 的 step 是 iteration，验证指标遵循 MMEngine LoggerHook 的训练 iteration。

正式输出默认在 `mmsegmentation/work_dirs/massformer_pv_general_pretrained/`：

| 文件/目录 | 用途 |
| --- | --- |
| `epoch_N.pth`、`last_checkpoint` | 最近两个完整断点、最近断点位置 |
| `train_config.py`、`pv_settings.resolved.json` | 实际训练配置、解析后的路径与参数 |
| `console.log`、时间戳目录中的 `.log` / `.json` | 控制台输出、MMEngine 训练与验证日志 |
| 时间戳目录下 `vis_data/events.out.tfevents.*` | 训练/验证曲线与固定训练图 |
| `fixed_train_samples/samples.json` | 固定训练图的原始路径清单 |
| `test/metrics.json` | 整体/每类别指标、混淆矩阵、样本数 |
| `test/per_image_metrics.csv` | 全部测试图的 PV 指标、是否展示、TensorBoard tag |
| `test/pred_masks/*.png` | 全部测试图的原始 0/1 预测标签 |
| `test/时间戳/vis_data/events.out.tfevents.*` | 测试指标及选中图片 |
| `test/test_config.py`、`test/console.log` | 本次测试配置与日志 |
| `training_process.json` | 训练子进程 PID，供完成提醒工具使用 |

可选 Windows 桌面提醒：训练启动后，在第二个同环境终端运行：

```powershell
python tools/watch_pv_training.py work_dirs/massformer_pv_general_pretrained --epochs 50
```

该工具只监控并提醒，不会自动开始测试。Windows 通知需允许 PowerShell 发送系统通知。

## 验证范围

迁移改动在当前 Linux/WSL 的 `massformer` 环境验证，包含真实 GPU 小规模训练、spawn 数据加载、
checkpoint 续跑、验证、测试与 TensorBoard 事件。原生 Windows/A4000 的实际驱动与算子仍需在目标机运行第 3 步确认。
原始 MaSS13K 七分类 benchmark 配置和论文运行方式保留；当前 PV 入口以本文为准。
