# Windows 上用 MaSSFormer 做遥感光伏板分割

## 1. 模型和权重用途

MaSSFormer 是基于定制 MMSegmentation 1.2.2 / MMDetection 的模型，主体为 ResNet_m-50、高分辨率像素解码器（含 DCNv2）和查询式 Transformer 解码器。

提供的 iter_80000.pth 已核查：分类头形状为 8×256，表示七类语义加 no-object；原始类别为 background、person、building、tree、ground、sky、water，**没有光伏板类别**。使用流程为：迁移预训练主体参数 → 用 General 的 train 标签微调背景/光伏分类头及模型 → 在 val 上选模型 → 在 test 上评估。

preview 仅展示原始七类结果。infer 要求有明确 background/solar_panel 元数据的已训练二分类权重，并严格匹配全部参数；初始化文件与一次迭代检查模型不能作为最终光伏模型。

## 2. 环境创建与依赖安装

### 2.1 已有环境与首次创建

先查看环境；已有 massformer 时直接激活，继续第 2.2 节，不用重复创建或删除：

```bat
conda env list
conda activate massformer
python --version
```

**仅当 massformer 不存在时**，执行：

```bat
conda create -n massformer python=3.11 -y
conda activate massformer
python --version
```

兼容组合为 Python 3.11、PyTorch 2.0.1 + CUDA 11.8、TorchVision 0.15.2、MMCV 2.1.0、MMEngine 0.10.3、MMDetection 3.3.0、NumPy 1.26.4。

### 2.2 手动安装

以下沿用本项目已验证的 Windows 依赖组合；本地项目安装使用作者 README 的 pip install -e .。在 Anaconda Prompt 中激活 massformer 后，逐条执行：

```bat
pip install "setuptools==69.5.1" wheel
pip install "numpy==1.26.4"
pip install "torch==2.0.1" "torchvision==0.15.2" --index-url https://download.pytorch.org/whl/cu118
pip install --only-binary=mmcv "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

然后自行进入 mmsegmentation 目录，执行：

```bat
pip install -r requirements/windows-solar.txt
pip install -e .
```

第一条安装本项目补充的 Windows 依赖，第二条是作者 README 原有的项目安装命令。依赖已经安装完成时，只需执行 pip install -e .。本文使用的 Python 3.11 / PyTorch 2.0.1 / MMCV 2.1.0 是已验证的 Windows 组合，与作者原版 Python 3.10 / PyTorch 1.12.1 / MMCV 2.0.0rc4 不同，不要混装。

### 2.3 可选：删除重建环境

仅在需要彻底重建时执行删除；正常使用、安装中断或新开终端都不需要删除。删除前先退出 massformer，再在 conda env list 中确认名称：

```bat
conda deactivate
conda env list
conda env remove -n massformer -y
conda create -n massformer python=3.11 -y
conda activate massformer
```

创建并激活完成后，按第 2.2 节逐条安装。

### 2.4 安装后检查与显卡选择

下面的依赖检查也可在任意目录执行，前提是 massformer 已正确激活，项目已完成 editable 安装：

```bat
pip check
python -c "import sys, torch, mmcv, mmengine, mmdet, mmseg; from mmcv.ops import ModulatedDeformConv2d; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print(mmcv.__version__, mmengine.__version__, mmdet.__version__); print(mmseg.__file__)"
```

CUDA 应可用，mmseg 路径应指向目标电脑当前项目。当前主配置使用 **device='cuda:1'、launcher='none'**，只使用目标电脑第二张可见 RTX A4000；tools/train.py、训练中的 val 和 tools/test.py 都按该配置使用同一张卡。下方预览和推理示例也指定 cuda:1。编号从 0 开始，未设置 CUDA_VISIBLE_DEVICES 时对应 nvidia-smi 的 GPU 1；若曾设置 CUDA_VISIBLE_DEVICES=1，则第二张物理卡会变成进程内 cuda:0，不能再按 cuda:1 选择。按本文直接配置 cuda:1 时应让两张卡都对进程可见。沿用已配置的 PyTorch 2.0.1/cu118 和 MMCV 2.1.0，无需因 nvidia-smi 显示 CUDA 12.8 而改装成 cu128。

若旧终端曾按早期文档设置项目 PYTHONPATH，在 Anaconda Prompt 中用 **set PYTHONPATH=** 清除，再激活 massformer；不需要旧隔离依赖目录。终端激活问题可直接改用开始菜单中的 Anaconda Prompt，不需要执行 PowerShell 专用命令。

## 3. 可选的数据、权重和原模型检查

日常使用只需要 tools/train.py 训练、tools/test.py 测试、tools/solar.py infer 推理。下列 check、inspect 用于排查数据和权重问题；prepare 的权重转换在首次训练时自动完成，无需手动执行；preview 仅诊断原七分类模型。本节无需每次训练前重复执行。

以下项目命令由你自行在 mmsegmentation 目录中执行。检查和预览命令的默认数据为工作区下 datasets/dis/General，源权重为 weights/cv/MaSS13K/iter_80000.pth；位置不同时给 check 传 --data-root，给 inspect、preview 传 --checkpoint；preview 的图像目录用 --input 指定。General 使用 train/val/test 下的 image、label 目录，输入为 8 位 RGB，标签为同尺寸单通道 0/1 或 0/255。训练路径由第 5 节的配置文件决定。

```bat
python tools/solar.py inspect
python tools/solar.py check --limit 50
```

需要全量检查时：

```bat
python tools/solar.py check --limit 0
```

原始七类预览，无需标注：

```bat
python tools/solar.py preview --device cuda:1 --limit 1 --output work_dirs/target_pretrained_preview
```

该结果报告 task=mass13k_7class_preview，不是光伏结果。输出已存在会拒绝覆盖；重跑时换一个新的 --output 目录。

检查和预览命令的默认数据/源权重路径按仓库位置推导；训练和正式测试使用配置文件中的路径。配置中的相对 work_dir，以及命令行相对 --input、--checkpoint、--output、--work-dir 等路径，均按当前终端目录解析。正式训练使用项目原有 tools/train.py，正式测试使用原有 tools/test.py，均已接入显卡选择和 Windows 兼容处理；tools/solar.py 保留数据检查、权重检查/转换、预览和推理，已移除 train 子命令。

## 4. 使用主配置在第二张 A4000 上训练、验证和测试

当前 massformer_r50_solar_general.py 已切换为 Windows 单张 RTX A4000 的完整数据训练配置：cuda:1、batch size 4、512×512 裁块、FP32、无梯度累积。激活 massformer，自行在 mmsegmentation 目录中执行：

```bat
python tools/train.py configs/massformer/massformer_r50_solar_general.py
```

默认训练 20000 次，每 1000 次用完整 val 验证、保存一次迭代权重，并对固定四张验证图保存四栏效果图；loss、学习率等仍每次迭代记录。三个 split 的 indices 均为 None：train 用于更新，val 用于选模型，test 只在单独执行测试时使用。数据与源权重路径保留配置中的 C 盘路径，换电脑后若位置不同，先修改 data_root 和 pretrained_checkpoint。

训练输出为 work_dirs/massformer_solar_general/exp1/train，测试输出为同一实验的 exp1/test，预测掩码放入 exp1/test/masks。保留用户已设置的 exp1/train 目录和 batch size 4；已有初始化权重时新训练会拒绝覆盖，新实验换新目录，已有训练快照则按第 5 节续训。

训练结束后执行：

```bat
python tools/test.py configs/massformer/massformer_r50_solar_general.py
```

默认评估本实验最后一次更新的 iter_20000.pth，测试完整 test。训练中仍按 val/mIoU 保存 best_mIoU_iter_N.pth；若要报告验证集最佳模型的测试分数，先将 test_checkpoint 改为实际最佳权重路径。最佳迭代号由训练结果决定，不能预先写死为 20000。

本次完成配置适配和本机检查，不代表已在目标 A4000 完成训练。此前两步快速流程记录保留在 [quick_run.md](quick_run.md)，历史权重和事件文件未删除。

可选代码检查，日常训练和模型评估无需执行：

```bat
python -m unittest discover -s tests -p test_solar_workflow.py -v
```

tests/ 是代码测试目录；General/test/ 是测试图像与标签目录，两者用途不同。

## 5. 训练配置与续训

训练使用项目原有 **tools/train.py**，默认全部以 **configs/massformer/massformer_r50_solar_general.py** 及其继承配置为准，只有显式命令行参数才覆盖；训练循环、反向传播、验证和保存继续复用 MMEngine Runner。

配置分为公共、train、val、test 四部分，可调参数已添加同行行尾中文说明。val 和 test 分别定义 pipeline、dataloader、evaluator；验证间隔由 train_cfg.val_interval 控制，验证日志随训练保存。当前设置如下：

| 配置项 | 当前值及用途 |
|---|---|
| data_root / pretrained_checkpoint | 保留原 C 盘 General 和 iter_80000.pth 路径，目标电脑位置不同时修改。 |
| device / launcher | 'cuda:1' / 'none'，只使用第二张可见 A4000，单进程单卡。 |
| train_dataloader.batch_size | 4，每次更新输入四张图，无梯度累积。 |
| val/test_dataloader.batch_size | 1，逐张评估，不影响训练 batch。 |
| 三个 dataloader.dataset.indices | None，分别使用完整 train、val、test。 |
| num_workers / persistent_workers | 0 / False，保留 Windows 主进程加载方式；有需要再按目标 CPU/内存调整。 |
| crop_size | (512, 512)，训练裁块及预处理尺寸。 |
| train_cfg.max_iters / val_interval | 20000 / 1000，总更新次数及完整 val 验证间隔。 |
| param_scheduler[0].end | 20001，匹配 max_iters+1，避免当前 MMEngine 版本提前将最后一次训练学习率降至 0。 |
| default_hooks.logger.interval / log_processor.window_size | 1 / 1，每次迭代记录当前批次原始 loss 等标量。 |
| default_hooks.checkpoint.interval / max_keep_ckpts | 1000 / 3，每 1000 次保存权重，常规快照最多保留三份，最佳权重另存。 |
| custom_hooks[0].interval / fixed_vis_names | 1000，固定同四张 val 图，保存原图/GT/预测/叠加；结束时也保存。 |
| optim_wrapper | FP32 AdamW，基础学习率 1e-4，骨干倍率 0.1；权重衰减、梯度裁剪等在同一区域说明。 |
| visualizer.vis_backends | LocalVisBackend + TensorboardVisBackend，同时保留本地文件和事件。 |
| work_dir / test_work_dir | work_dirs/massformer_solar_general/exp1 下的 train / test。 |
| test_checkpoint | 默认由 work_dir 和 train_cfg.max_iters 生成最终迭代权重路径；使用最佳模型时填写实际 best_mIoU_iter_N.pth。 |
| load_from / resume | None / False，首次迁移训练；续训时按下方说明设置。 |

修改 data_root、crop_size 后，引用它们的加载器和预处理会在重新读取配置时一起更新。修改 max_iters 时同步将调度器 end 设为 max_iters+1，默认 test_checkpoint 会随 max_iters 更新；验证、权重保存、图片保存间隔分别配置。

20000 次是训练起始方案，应根据 val 曲线判断是否收敛。以当前 train=17648、batch size=4 计算，约 4412 次更新遍历一次 train，20000 次约为 4.53 轮。训练使用 train 更新参数，val 选择模型，不用 test 选模型；七类源权重迁移时只移除分类层，重新学习背景/光伏类别。

临时覆盖某项时才传参数，例如下面只更换输出目录，其余仍按配置：

```bat
python tools/train.py configs/massformer/massformer_r50_solar_general.py --work-dir work_dirs/solar_experiment_2
```

另选配置时直接替换第一个配置文件路径。日常参数在配置文件中修改；临时覆盖使用原生 --cfg-options，例如：

```bat
python tools/train.py configs/massformer/massformer_r50_solar_general.py --cfg-options train_dataloader.batch_size=2
```

命令行 --work-dir 只覆盖本次训练输出，不会重新计算配置中的 test_checkpoint 和 test_work_dir；新实验建议在配置内同步修改这三项。更改训练总次数不会自动修改学习率调度结束点，更改验证间隔也不会自动修改权重保存周期。--cfg-options 在配置加载后生效，不会重新计算 Python 变量引用；临时改 data_root 或 crop_size 时，需同时覆盖实际数据加载器、裁剪变换和预处理中的对应项，因此这类修改建议直接编辑配置文件。

配置中 load_from 非空时使用该文件；否则首次单卡训练从 pretrained_checkpoint 生成迁移初始化。更换源权重时修改 pretrained_checkpoint，并确认 load_from=None、resume=False。转换函数复用 tools/solar.py 的权重处理，原权重保持不变。配置中 AMP、resume 和 launcher 同样有效，可用 --amp / --no-amp、--resume / --no-resume、--launcher 显式覆盖；同一项的专用显式参数优先于 --cfg-options。

本任务使用 launcher='none' 的单卡模式。原入口的分布式和自定义 Runner 分支保留；如另行进行分布式迁移，应先用 tools/solar.py prepare 转换一次，再将生成路径写入 load_from，避免多个进程同时生成初始化文件。

未传 --work-dir 时使用配置的 work_dir；只有配置也未设置时，才使用 work_dirs/配置文件名（不含 .py）。输出目录会保存继承合并、命令行覆盖后的实际运行配置，源配置文件不改写。

输出包含 init_from_mass13k.pth（无分类头的迁移初始化）、iter_*.pth（快照）、best_mIoU_iter_*.pth（验证最佳模型）及日志。已有初始化文件时，新训练会拒绝覆盖。需要新实验换 --work-dir；已有有效训练快照时才使用 --resume：

```bat
python tools/train.py configs/massformer/massformer_r50_solar_general.py --resume
```

续训必须指向**原实验的同一个工作目录**；原目录与配置不同则加 --work-dir 指定。配置 resume=True 时，不传 --resume 也会续训；做新的独立实验需设为 False 或显式加 --no-resume。只有初始化文件、尚未保存训练快照的失败运行不具备续训条件，可修复原因后使用新的工作目录重新开始。

显存不足可选择以下一种方案。临时启用 AMP：

```bat
python tools/train.py configs/massformer/massformer_r50_solar_general.py --amp --work-dir work_dirs/solar_amp
```

使用较小裁块时，先在配置文件中将 crop_size 改为 (256, 256)，再执行：

```bat
python tools/train.py configs/massformer/massformer_r50_solar_general.py --work-dir work_dirs/solar_256
```

这些示例继承配置中的 resume 和 load_from，开始新实验应确认 resume=False。AMP 未在本次验证中实测，如算子半精度报错可使用默认 FP32。小裁块改变可见上下文，精度需重新评估。

## 6. 测试集评估与图像推理

训练期间使用 train 更新参数、val 选择最佳模型，不会自动评估 test。正式评估使用项目原有 tools/test.py，读取配置的 test_dataloader、test_evaluator；当前为 General/test/image 和 test/label。

当前 A4000 主配置的测试权重和输出路径：

```python
test_checkpoint = f'{work_dir}/iter_{train_cfg["max_iters"]}.pth'
test_work_dir = 'work_dirs/massformer_solar_general/exp1/test'
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'],
                      output_dir=f'{test_work_dir}/masks')
```

先完成训练，再运行测试。默认读取 iter_20000.pth；评估验证集最佳模型时，将 test_checkpoint 改为对应实验实际生成的 best_mIoU_iter_N.pth。

随后测试命令只需配置文件：

```bat
python tools/test.py configs/massformer/massformer_r50_solar_general.py
```

测试按当前配置选择目标电脑第二张 A4000（cuda:1），加载 test_checkpoint，并将日志和结果 JSON 保存到 test_work_dir。测试权重优先级为命令行显式 checkpoint > 配置 test_checkpoint > 配置 load_from；输出目录优先级为显式 --work-dir > test_work_dir > work_dir。原来的长命令仍可用于临时覆盖，不必日常重复填写。

即使配置 resume=True，测试也不会改为自动续训或读取其他权重。当前评估输出逐类结果及 mIoU、mDice、mFscore 等平均指标，不更新模型参数。原七类权重、迁移初始化或缺少光伏类别元信息的权重会被拒绝。

图像推理使用 tools/solar.py infer：加载模型并生成分割图，支持无标注图像和滑窗预测，不依赖 test_dataloader。直接指定 --input，例如：

```bat
python tools/solar.py infer --device cuda:1 --checkpoint work_dirs/massformer_solar_general/exp1/train/iter_20000.pth --input "E:/待预测图像/example.tif" --output work_dirs/solar_single
```

将示例路径替换成真实文件或目录。先预测少量图可增加 --limit 10；处理全部输入图像时不加该参数。重跑时换新的 --output，避免与已有结果重名。

infer 仍可通过 --labels 标注目录附加计算光伏前景指标，作为预测结果分析；正式测试流程采用上面的 tools/test.py。infer 不传 --labels 时只生成预测结果和运行报告。

输出为 mask（0/1）、binary（0/255，白色光伏）、color、overlay 四类同尺寸 PNG，以及 report.json。带标签时累计全体处理像素的 TP/FP/FN/TN，输出光伏 IoU、Dice/F1、precision、recall、像素准确率，取值 0～1，分母为零时为 null；这些前景指标与训练的两类平均 mIoU 不同。

默认滑窗 512、步长 384，重叠区域平均类别分数再分类；要求 0 < stride <= tile-size。小图补到 256 的倍数后裁回，512 图块只用一个窗口。整幅 RGB 和融合分数保存在 CPU 内存，超大镶嵌图宜先用 GIS 流程裁块。输出保留子目录层级，输入输出目录分开；PNG 不保留 GeoTIFF 的 CRS/仿射变换，如需地理参考，应另行写回与输入匹配的 GeoTIFF。

## 7. 日志结构、快速全流程与常见错误

本节目录和数值描述的是接入 TensorBoard 前的历史短流程；旧日志保持原样。当前新增的 TensorBoard 后端及逐次原始数值记录见第 8 节。

训练和测试不要求保存在同一目录。以下历史运行的训练目录为 work_dirs/massformer_solar_general/train，测试目录为 work_dirs/massformer_solar_general/test；val 是训练过程中的验证，日志保存在训练目录。建议为同一次实验使用一个父目录，再分 train/、test/，本次快速检查采用此形式。

2026-11-23 按用户纠正，清理上次独立 quick 配置产物后，已使用 massformer 环境在本机 RTX 4060 Laptop GPU（cuda:0）直接执行主配置完成短流程：

```bat
python tools/train.py configs/massformer/massformer_r50_solar_general.py
python tools/test.py configs/massformer/massformer_r50_solar_general.py
```

本节描述此前直接修改主配置执行的历史两步检查：2 次更新、2 张 val 和 2 张 test，每次更新输出日志。当前主配置已改为第 4～5 节的 A4000 完整数据方案，历史运行仍保留在原目录，独立 quick 配置已删除。

历史短流程的日志和快照间隔为 1、验证间隔为 2；当前 A4000 配置日志间隔仍为 1，快照、验证、固定图片间隔均为 1000。验证/测试逐类表在阶段结束时输出。

当前 work_dirs 只包含 .cache/ 和本次 massformer_solar_general/。旧环境核查、预览及历次检查目录已删除，其历史说明保留在 changes.md。新目录核心结构为：

```text
work_dirs/massformer_solar_general/
├─ train/                         训练和 val 的配置、权重、时间戳日志
│  └─ 20261123_105810/
│     ├─ 20261123_105810.log       人工阅读的训练/验证日志
│     └─ vis_data/scalars.json     每行一个 JSON：训练 loss、学习率、验证指标等
├─ test/                          测试配置、预测 PNG、时间戳日志
│  ├─ masks/                      本次两张 0/1 预测 PNG
│  └─ 20261123_105844/
│     ├─ 20261123_105844.log       逐类表和平均测试指标
│     └─ 20261123_105844.json      一个 JSON 对象：平均测试指标
├─ train_console.log              本次额外捕获的训练控制台输出
├─ test_console.log               本次额外捕获的测试控制台输出
├─ cleanup.json                   清理范围和原文件哈希
└─ summary.json                   本次结果与核查汇总
```

原生 .log、配置副本、scalars.json、权重和测试结果 JSON 自动生成；父目录中的控制台日志和两个汇总 JSON 是本次额外记录。每次执行新增时间戳日志目录，保存对应的配置副本。相同 work-dir 根目录下的同名配置会更新，因此训练和测试分目录更清楚。

| 文件/字段 | 阅读方式 |
|---|---|
| train/*.py、时间戳/vis_data/config.py | 合并继承和显式覆盖后的实际运行配置。 |
| Iter(train) [1/2] | 第 1 次更新，共 2 次；loss 是训练损失，可能按窗口平滑。 |
| Iter(val) [2/2] | 第 2 个验证批次，共 2 批；指标随训练 step 记录，不更新参数。 |
| Iter(test) [2/2] | 第 2 个测试批次，共 2 批；查看前面的 Load checkpoint 行确认使用的权重。 |
| iter_*.pth、last_checkpoint | 前者是含优化器/调度器的续训快照，后者只是最近快照的路径文本。 |
| best_mIoU_iter_2.pth | 本次 val 最佳模型，用于测试/推理；不含优化器和调度器。 |
| init_from_mass13k.pth | 七分类转光伏任务的初始化，不是已训练光伏模型。 |

本次日志 loss 为 193.8103、180.0586；val mIoU=40.33%，test mIoU=55.50%，test 光伏类 IoU=12.96%。这些只是两次更新和两张图的流程检查结果，不能当作全量精度。已核对 loss 有限、正确权重加载、两张 PNG 尺寸和取值，保存的实际配置与本次主配置相符，以及原预训练权重哈希不变。

完整文件树、逐项日志解释、权重大小及复现记录见 [quick_run.md](quick_run.md)。此前全量数据检查、环境重建、15 项代码测试等历史结论见 changes.md；历史产物已清理。**正式 20000 次训练、全量测试精度、AMP 和目标第二张 RTX A4000 尚未实测。**

| 现象 | 排查与处理 |
|---|---|
| no such option: -m；出现 wheelpython | 多条命令粘成一行。每次只复制一条，按回车执行，成功后再执行下一条。 |
| 找不到 requirements/windows-solar.txt、setup.py、tools/solar.py | 相对路径未指向对应文件，自行确认当前项目目录或改用绝对路径。 |
| -r / -e 引用了错误项目 | -r 的相对文件及 -e . 依赖当前目录；切换至项目目录，或显式指定文件/项目的绝对路径。安装目标环境由当前 pip 决定。 |
| 单引号被当作路径或包名的一部分 | Anaconda Prompt 使用英文双引号，不能照抄 PowerShell 单引号。 |
| 环境已存在 | 激活已有环境；不要重复创建。确需清空时按第 2.3 节单独重建。 |
| CondaError: Run conda init before conda activate，或 PowerShell profile.ps1 被禁止 | 从开始菜单打开 Anaconda Prompt，再激活 massformer。 |
| 找不到源权重或数据 | 训练时修改配置中的 data_root、pretrained_checkpoint 或 load_from；检查、预览、推理使用对应的 --data-root、--checkpoint 或 --input。 |
| 无法保存或输出已存在 | 对新实验更换 --work-dir，对预测更换 --output；有训练快照时才在原目录续训。 |
| TensorBoard 没有 Images | 检查当前服务的 --logdir 是否包含本次训练目录。当前 A4000 实验位于 exp1，历史固定图检查位于 fixed_visualization_run；在原 TensorBoard 终端按 Ctrl+C 后用第 9 节命令重启，再刷新浏览器。固定图片要到第 1000 次更新才首次保存，只启动新命令但仍访问旧服务端口会继续看到旧目录。 |
| No module named mmcv._ext / DLL load failed | 核对 CUDA 版 PyTorch、匹配的 MMCV wheel、Python 位数与 VC++ 运行库；不要装 mmcv-lite。 |
| CUDA 不可用 / invalid device ordinal | 核对 CUDA 版 PyTorch、驱动与可见 GPU 数；训练仅有一张可见卡时将配置 device 改为 cuda:0，或临时传 --cfg-options device=cuda:0。 |
| Expected more than 1 value per channel | 使用二分类配置且保留 norm_eval，不要关闭单图训练的 BN 固定设置。 |
| 窗口 view/shape 错误 | 使用新入口，训练裁块须为 256 的倍数。 |
| TRAINED 2-class checkpoint / missing cls_embed / class semantics are unverified | 传入了原七类、初始化或类别语义不明确的权重。先微调，使用有明确光伏元数据的训练输出。 |

训练、测试和推理入口默认把 Matplotlib 缓存放在 work_dirs/.cache/matplotlib，尊重已设置的 MPLCONFIGDIR；tools/train.py 和 tools/test.py 在中文 Windows 下对环境信息采集的 UnicodeDecodeError 记录警告并继续，不屏蔽实际训练或测试异常。

## 8. TensorBoard：逐次 loss、学习率、验证指标和配置

之前采用 MMEngine LoggerHook + LocalVisBackend，输出 .log、配置副本和 JSON。现在继续使用同一套原生记录流程，在主配置公共设置中同时启用本地文件和 TensorBoard：

```python
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(vis_backends=vis_backends)
log_processor = dict(by_epoch=False, window_size=1, log_with_hierarchy=True)
```

训练设置中继续使用：

```python
default_hooks = dict(logger=dict(interval=1, log_metric_by_epoch=False),
                     checkpoint=dict(interval=1000, save_best='mIoU', max_keep_ckpts=3))
```

logger.interval=1 决定每次训练循环都写入，window_size=1 决定记录当前批次原始值；只将 interval 改为 1 而保留默认窗口 10，仍可能记录平均 loss。验证按 train_cfg.val_interval 执行，在对应训练 step 写入验证指标。保留 LocalVisBackend 后，原生 .log 和 JSON 仍然生成。

这里按用户要求记录常规指标和配置，不增加各层权重或梯度的直方图。无需新建训练循环或 TensorBoard Hook。[MMEngine 原生后端说明](https://mmengine.readthedocs.io/en/stable/api/generated/mmengine.visualization.TensorboardVisBackend.html)。

### 安装与启动

本机 massformer 已安装并通过验证。其他电脑激活 massformer 后执行一次：

```bat
pip install "tensorboard==2.14.0" "protobuf==4.25.8"
```

这两个依赖也已写入 requirements/windows-solar.txt，按第 2 节安装时会一起安装。不需要安装 TensorFlow。

在新的 Anaconda Prompt 中激活 massformer，自行进入 mmsegmentation 目录，执行：

```bat
tensorboard --logdir work_dirs/massformer_solar_general --host 127.0.0.1 --port 6006
```

保持该终端运行，用浏览器打开 [http://127.0.0.1:6006](http://127.0.0.1:6006)。训练可在另一个已激活 massformer 的终端执行，页面会读取不断更新的事件文件。页面上的 Smoothing 设为 0 可直接查看未平滑的数值。

### Scalars 中没有 loss 时

此前快速检查只有两个 loss 点。若已经有 `train/loss` 卡片，但图上没有曲线、纵轴却是 0～1，先在 Scalars 左侧取消勾选 **Ignore outliers in chart scaling**（缩放图表时忽略异常值），Smoothing 保持 0。TensorBoard 2.14.0 的该选项会按数值的 5%～95% 分位区间计算纵轴；只选一个运行、只有两个不同数值时，两端点都被排除，得到空范围而使用默认 0～1，导致 loss 曲线落在可视范围外。取消勾选后应立即恢复；必要时点击图表下方最右侧的四向箭头按钮重置坐标。该选项是网页显示设置，不需要重新训练或生成事件文件，浏览器会记住其状态。

直接打开 [Scalars 页面](http://127.0.0.1:6006/#scalars)，清空运行筛选和标签筛选，勾选 `four_panel_run\train\20261123_172349\vis_data`（界面也可能用 `/` 显示路径），展开 `train` 分组，找到 `train/loss`。如果之前停留在 Images 页面，需要切换到 Scalars；如果只筛选了 test 运行，则没有训练 loss。必要时点击页面刷新或浏览器强制刷新。

2026-11-23 已直接核查当前 6006 服务：logdir 为 `work_dirs/massformer_solar_general/fixed_visualization_run`，Scalars 插件已启用，当时最新四栏训练运行包含 39 个标量标签。服务的 `train/loss` 接口实际返回 step=1 的 193.8116455 和 step=2 的 166.3241882。随后在独立 Chrome 会话复现了上述纵轴问题：仅选最新训练、Smoothing=0、异常值过滤开启时，loss 和 grad_norm 的纵轴均为 [0, 1]；关闭过滤后 loss 纵轴恢复 [160, 200]、grad_norm 恢复 [1000, 9000]，两条曲线均可见。训练只有两次更新，所以每条曲线只有两个点；结束后不会继续产生新点。原生测试日志没有训练 loss，测试指标仍查看对应测试 JSON 和 .log。

主配置 `configs/massformer/massformer_r50_solar_general.py` 的可调设置已添加同行末尾的中文说明，并显式列出常用优化器、梯度裁剪、参数组和主要损失系数。当前已按第 4～5 节切换 A4000 完整数据训练，device='cuda:1'，日志仍每次迭代记录。上面的两点曲线问题与数值属于此前快速检查；新训练在 Scalars/Images 中选择 exp1/train/实际时间戳/vis_data。

### 页面内容及文件位置

| 页面/标签 | 内容 |
|---|---|
| Scalars / train/loss | 每次训练循环的总 loss。 |
| train/decode.loss_cls、loss_mask、loss_dice、edge_loss 等 | 各项损失及解码器辅助损失。 |
| train/base_lr、train/lr | 基础学习率及记录的参数组学习率；本模型主干倍率为 0.1。 |
| train/grad_norm | 裁剪前梯度范数。 |
| train/time、train/data_time、memory | 耗时、数据处理时间及峰值分配显存。 |
| val/mIoU、val/mDice、val/mFscore 等 | 验证阶段的平均指标，记录在对应训练 step。 |
| Text / config | 该次实际运行的完整配置，包括 batch size、优化器等超参数；不是单独的 HParams 面板。 |

新训练的事件文件由 TensorboardVisBackend 自动生成：

```text
训练工作目录/
└─ 时间戳/
   └─ vis_data/
      ├─ config.py
      ├─ scalars.json
      ├─ 时间戳.json
      └─ events.out.tfevents.*
```

旧目录没有事件文件时不会自动出现在 TensorBoard。本次为核查新配置，另外使用 work_dirs/massformer_solar_general/tensorboard_check 作为训练 work-dir，保留原来的 train/ 和 test/。实际事件文件位于 tensorboard_check/20261123_120332/vis_data/，上面的 logdir 命令可以直接找到它。

最初后端接入检查的两个 train/loss 点在 step=1、2，值约为 193.8103、166.3077；val/mIoU 在 step=2，值约 40.33，并包含 config/text_summary 文本。新的数值 JSON 使用 train/、val/ 前缀，与接入前旧 JSON 的无前缀键不同。

以下为此前 TensorBoard 两步验证的历史记录：当时将 PolyLR.end 从 2 改为 3，确认两轮均使用非零学习率、模型参数均有变化。该次输出位于 work_dirs/massformer_solar_general/tensorboard_run/train 与 test，train 时间戳为 20261123_163000、test 时间戳为 20261123_163034；旧结果保留。当前 A4000 配置及 exp1 路径见第 4～5 节。

### 新配置重跑结果（2026-11-23）

使用同一个主配置、没有额外参数覆盖，完成 2 次训练循环、2 张 val、2 张 test。TensorBoard train/loss 分别约为 193.8103、166.0430；主干学习率分别约为 1e-5、5.3589e-6。两次权重快照逐项比较确认第二轮有 414 个状态条目变化。val mIoU=40.33%，test mIoU=49.82%，仅用于流程检查。

只查看这次运行可执行：

```bat
tensorboard --logdir work_dirs/massformer_solar_general/tensorboard_run --port 6006
```

在 Scalars 中选择 train/20261123_163000/vis_data 对应的运行，查看 train/loss、train/lr、val/mIoU 等。原生 LoggerHook 的测试结果通过 test/20261123_163034/20261123_163034.json 和 .log 保存；test 的事件文件本次记录配置 Text，不自动记录测试标量。最新完整记录见 [quick_run.md](quick_run.md) 文末。

## 9. 训练中定期保存固定四张图的分割效果

用户已选定验证集排序前四张：

1. F47Mosaic_F47D001008_00000018.tif
2. F48Mosaic_F48D001002_00000043.tif
3. F48Mosaic_F48D002001_00000057.tif
4. F48Mosaic_F48D002003_00000091.tif

源图位于 General/val/image，标签位于 val/label。主配置的 fixed_vis_names 显式列出这些名称，重新训练或续训仍使用同样的文件；更换图片时修改名称及相应路径。当前 A4000 配置的 custom_hooks[0].interval=1000，每 1000 次更新保存一次；此前快速流程是每次保存。draw_last=True 保证最后一次训练循环也保存，不重复写同一步。

这四张图的预测独立于 val_dataloader：即使快速配置的 val.dataset.indices=2，快照仍固定预测四张。快照不计算并混入 val 分数、不更新模型，不改变原来的验证集指标。仅进行 eval/no-grad 前向，并恢复各子模块的训练标志及随机状态。

每个保存点生成四张 RGB PNG，每张从左至右为 Original（原始 RGB）、GT（黑白真实掩码）、Prediction（黑白预测掩码）、Overlay（预测与原图叠加），均带标题。GT/预测图中白色为光伏、黑色为背景；叠加图仅对预测光伏区域混合橙色，背景保留原图像素，alpha=0.5 可调透明度。512×512 样本生成 2048×544 对比图，其中标题条高 32 像素。通过现有 Visualizer.add_image 和 Local/TensorBoard 后端保存：

```text
训练 work_dir/
└─ 本次时间戳/
   └─ vis_data/
      ├─ vis_image/
      │  ├─ fixed_val_01_图片名_1000.png
      │  ├─ fixed_val_02_图片名_1000.png
      │  ├─ ...
      │  ├─ fixed_val_01_图片名_2000.png
      │  └─ ...
      └─ events.out.tfevents.*
```

文件名最后的数字为训练全局迭代号；正式训练会是 1000、2000 等。打开 TensorBoard 的 Images，选择 fixed_val_01_… 到 fixed_val_04_… 四个标签，使用 step 滑块查看同一张图在不同训练阶段的效果。本功能为程序自动生成的中间分割结果，权重快照、标量日志与图片分别保存。

当前 A4000 配置输出为 work_dirs/massformer_solar_general/exp1/train 和 exp1/test。此前四栏与双栏检查结果仍分别保留在 fixed_visualization_run/four_panel_run 和 fixed_visualization_run/train 下。仍使用原有两条入口命令，不另建配置或训练循环。

```bat
python tools/train.py configs/massformer/massformer_r50_solar_general.py
python tools/test.py configs/massformer/massformer_r50_solar_general.py
```

只查看当前 A4000 实验的图像和曲线可运行（训练产生事件后，在页面选择 train/实际时间戳/vis_data）：

```bat
tensorboard --logdir work_dirs/massformer_solar_general/exp1 --port 6006
```

若 6006 端口仍是此前读取 fixed_visualization_run 的服务，先在该 TensorBoard 终端 Ctrl+C，再用上面命令启动并刷新页面；旧服务不会读取位于 exp1 的新事件。第 8 节读取 massformer_solar_general 总目录的命令也可同时查看新旧实验。

新增的 FixedSampleVisualizationHook 只负责固定样本快照；原 SegVisualizationHook 的批次抽图保持原设置，不会额外把整个验证集都保存成图片。

此前双栏版本已用主配置完成两步训练、2 张 val 评估及 2 张 test 评估。其历史固定样本图位于 fixed_visualization_run/train/20261123_170130/vis_data/vis_image：step=1、2 各四张，共八张，每张尺寸 1024×512（两个 512×512 面板）。TensorBoard Images 有四个 fixed_val 标签，每个均有 step 1、2，已核对本地 PNG 与事件内图像像素一致，且同一图在不同 step 的 GT 面板完全一致。17 项代码测试通过，含间隔、重复回调、模型/RNG 状态恢复检查。测试阶段时间戳为 20261123_170316；这些仍是极短流程结果。

2026-11-23 用户反馈页面没有 Images，现场核查发现 6006 端口仍读取旧 tensorboard_run。已将该 TensorBoard 服务重启到 fixed_visualization_run，并通过网页的 images/tags 接口确认四个固定图像标签可用。浏览器刷新后可直接访问 http://127.0.0.1:6006/#images，选择 train/20261123_170130/vis_data 对应运行。

此前按用户要求改为四栏对比并重新验证。当时的新运行选择 four_panel_run/train/20261123_172349/vis_data，仍可使用现有 http://127.0.0.1:6006/#images；旧 train/20261123_170130/vis_data 是双栏历史结果。已核对新运行有四个固定图像标签、各有 step=1、2，本地八张 PNG 均为 2048×544，与 TensorBoard 图像像素一致。原图和 GT 跨 step 相同，叠加图背景与原图完全相同。18 项代码测试通过，训练、验证及测试流程也已完成。
