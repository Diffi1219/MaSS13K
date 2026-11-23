# 使用主配置完成快速训练、验证和测试的记录

执行日期：2026-11-23。本次直接修改并执行 configs/massformer/massformer_r50_solar_general.py，使用 massformer 环境、本机单张 RTX 4060 Laptop GPU（cuda:0）。主配置当前保留两次更新的快速参数，不能作为正式模型精度。

本记录对应 TensorBoard 接入前的 train/20261123_105810 与 test/20261123_105844。当前主配置已增加 TensorBoard，并将日志窗口改为 1、数值标签加上 train/val 前缀；旧记录不改写，新记录与查看方法见 use.md 第 8 节。

## 清理范围

此前按用户要求清空 work_dirs 的 17 个旧目录、113 个文件（约 4.68 GB）。用户进一步明确必须使用主配置后，本次清理上次独立 quick_run 的 20 个文件、1,205,781,683 字节，并删除独立 quick 配置文件；核对绝对路径和无重解析点后，再生成主配置的运行产物。当前 work_dirs 仅保留 .cache 和 massformer_solar_general。

数据位于工作区 datasets/dis/General，原预训练权重位于 weights/cv/MaSS13K/iter_80000.pth，均在清理目录外。清理前原权重 SHA-256：73FFB40AB51510160625E6B95FFBA550C104B3F2798BADC872F492CCC342AF0A。

## 复现命令

先激活 massformer，自行在 mmsegmentation 目录逐条执行：

```bat
python tools/train.py configs/massformer/massformer_r50_solar_general.py
python tools/test.py configs/massformer/massformer_r50_solar_general.py
```

本次训练使用 General/train 前两对图像、更新 2 次；第 2 次更新后使用 General/val 前两对图像验证，保存最佳权重；测试加载该权重，评估 General/test 前两对图像。所有快速参数直接写在主配置中：max_iters=2、val_interval=2、调度 end=2、三个数据集 indices=2、日志和保存间隔=1。本次执行后保留这些参数，恢复完整训练的方法见文末。

训练与测试分阶段运行，前一条成功后再执行下一条。重跑新训练需使用新的输出目录，或明确清理本例目录后重跑；已有初始化权重不会被覆盖。

## 执行结果与日志说明

两条命令均成功结束。训练完成 2 次更新，随后验证 2 张图；测试读取本次保存的 best_mIoU_iter_2.pth，评估另外的 2 张 test 图并导出两张 512×512、取值 0/1 的 PNG。

| 检查项 | 实际记录 |
|---|---|
| 第 1 次更新日志中的 loss | 193.8103 |
| 第 2 次更新日志中的 loss | 180.0586 |
| val mIoU / mDice | 40.33% / 44.65% |
| test mIoU / mDice | 55.50% / 60.98% |
| test 光伏类 IoU / Dice | 12.96% / 22.95% |
| 测试权重 | train/best_mIoU_iter_2.pth |

这些分数来自两次更新、两张图的流程检查。val 与 test 使用不同图像，其数值不能用于判断模型已收敛，也不能作为全量测试精度。val 中光伏类未被预测时，部分 Precision/Fscore 分母为零而显示 nan，均值会忽略 nan；判断光伏效果应同时看逐类 IoU/Dice。

已核对训练损失有限、权重迭代号和二分类元信息正确、测试使用指定最佳权重；iter_2.pth 包含 optimizer 和 param_schedulers，best_mIoU_iter_2.pth 不包含这两项。原预训练权重的 SHA-256 保持不变；主配置按用户要求改为快速设置，训练和测试保存的实际配置均核对为本次主配置。

## 目录结构与生成来源

训练和测试不要求放在同一目录。本例用一个实验父目录 massformer_solar_general 归组，阶段产物分别放 train/、test/。val 属于训练过程，所以日志写在 train/，不单独生成 val/。

下面是本次实际结构，时间戳为框架生成的目录名：

```text
work_dirs/
├─ .cache/matplotlib/                         自动重建的绘图库缓存
└─ massformer_solar_general/
   ├─ cleanup.json                           本次额外记录的清理范围和原文件哈希
   ├─ summary.json                           本次额外汇总的结果及核查记录
   ├─ train_console.log                      本次额外捕获的训练标准输出/错误
   ├─ test_console.log                       本次额外捕获的测试标准输出/错误
   ├─ train/
   │  ├─ massformer_r50_solar_general.py 实际合并后的训练配置
   │  ├─ init_from_mass13k.pth                移除原七类分类头后的迁移初始化
   │  ├─ iter_1.pth                          第 1 次更新的续训快照
   │  ├─ iter_2.pth                          第 2 次更新的续训快照
   │  ├─ best_mIoU_iter_2.pth                 本次 val mIoU 最佳模型
   │  ├─ last_checkpoint                     文本文件，指向最近的 iter 快照
   │  └─ 20261123_105810/
   │     ├─ 20261123_105810.log               本次训练和验证的原生日志
   │     └─ vis_data/
   │        ├─ config.py                     本次运行配置副本
   │        ├─ scalars.json                  训练与验证数值，每行一个 JSON 对象
   │        └─ 20261123_105810.json           同批数值的时间戳日志
   └─ test/
      ├─ massformer_r50_solar_general.py 实际合并后的测试配置
      ├─ masks/
      │  ├─ F47Mosaic_F47D001008_00000016.png  第 1 张预测，0/1 类别图
      │  └─ F47Mosaic_F47D001008_00000017.png  第 2 张预测，0/1 类别图
      └─ 20261123_105844/
         ├─ 20261123_105844.log               本次测试的原生日志
         ├─ 20261123_105844.json              测试平均指标，一个 JSON 对象
         └─ vis_data/config.py               本次测试配置副本
```

`.log`、运行配置、权重、scalars.json 和测试指标 JSON 由原训练/测试流程生成。父目录中的 console.log、cleanup.json、summary.json 是本次额外保存的执行记录；直接运行前述两条命令也会自动生成原生 .log，无需自己重定向才能得到日志。

每次执行会新建时间戳目录。续训沿用原 train 目录时会新增时间戳日志，根目录配置副本会更新；时间戳内的 config.py 保留对应那次的设置。将训练和测试放入不同子目录，可以避免同名根目录配置互相覆盖。

## 如何阅读日志

先看 train/20261123_105810/20261123_105810.log 中的三类记录：

```text
Iter(train) [1/2] ... loss: 193.8103 ...
Iter(train) [2/2] ... loss: 180.0586 ...
Iter(val)   [2/2] ... mIoU: 40.3300  mDice: 44.6500 ...
```

- Iter(train) [1/2]：第 1 次训练更新，总共 2 次；不是 epoch。
- loss：总训练损失；decode.loss_cls、loss_mask、loss_dice、edge_loss 是各项损失，d0～d5 是解码器各层辅助损失。默认日志平滑窗口为 10，短流程的后续日志值可能是已有更新的平均值，不能全部当作该单批原始损失。
- base_lr 是基础学习率，lr 对应日志中记录的参数组；主干有 0.1 倍学习率倍率。本例主配置的调度 end=2，结束位置日志显示 lr=0；恢复完整训练时再将 end 改为计划总更新次数，例如 20000。
- grad_norm：裁剪前梯度范数，不能当作裁剪后的值；time/data_time 表示迭代和数据处理耗时，memory 表示记录的 GPU 峰值分配内存，单位 MB。
- Iter(val) [2/2]：第 2 个验证批次，共 2 批；当前每批 1 张。验证不更新参数。逐类表中的 solar_panel 行是光伏类指标，mIoU/mDice 是类别平均指标，aAcc 是整体像素准确率。本项目的原生评估指标按百分数显示。

再看 test/20261123_105844/20261123_105844.log：

```text
Load checkpoint from work_dirs/massformer_solar_general/train/best_mIoU_iter_2.pth
Iter(test) [2/2] ... mIoU: 55.5000  mDice: 60.9800 ...
```

第一行确认实际加载哪个模型，最后一行是本次测试平均分数。逐类结果在 .log 的表格中；20261123_105844.json 是平均指标的机器可读版本。本次测试默认没有 scalars.json，测试结果通过单独 JSON 保存。

训练 scalars.json 本次有 3 行：两行训练记录加一行验证记录。验证行的 step=2 表示它对应第 2 次训练更新，并不代表又训练了一次；文件适合后续画 loss 和验证曲线。

主配置现在的训练日志和快照间隔均为 1，验证间隔为 2，便于短流程查看。恢复完整训练时可将日志间隔改为 50、验证和快照间隔改为 1000，详见文末。

训练中提示缺少 decode_head.cls_embed.weight/bias，是首次迁移时主动移除原分类头的结果。Windows 编译器环境文本解码警告已有兼容处理，本次训练和测试均成功继续执行。

## 哪些文件用来续训、测试或查看结果

| 文件 | 主要用途 |
|---|---|
| iter_2.pth | 续训，包含模型、优化器、调度器和运行状态；本次约 452 MB。 |
| best_mIoU_iter_2.pth | 测试或推理，包含模型参数及元信息；本次约 151 MB，不含优化器和调度器。 |
| init_from_mass13k.pth | 尚未学习光伏分类头的迁移初始化，本次约 151 MB。 |
| last_checkpoint | 最近续训快照的路径，不是模型权重。 |
| 原生 .log | 人工阅读训练过程、验证/测试逐类表和平均指标。 |
| train/.../vis_data/scalars.json | 读取训练及验证数值、绘制曲线。 |
| test/.../20261123_105844.json | 读取测试平均指标。 |
| test/masks/*.png | 查看逐像素预测，0=背景、1=光伏，直接看可能接近全黑。 |

目录中占空间的主要是权重快照。清理工作目录会失去相应的实验权重、日志及续训状态；已完成记录的旧实验此次按用户要求删除。完整训练前应按下一节调整同一个主配置，并使用新的输出目录。

## 恢复完整训练时修改同一个主配置

| 字段 | 当前快速值 | 完整训练示例 |
|---|---|---|
| device | 'cuda:0' | 目标双卡电脑使用第二张卡时改为 'cuda:1'。 |
| train/val/test_dataloader.dataset.indices | 三者均为 2 | 三者均改为 None，使用各自完整数据；不要遗漏。 |
| train_cfg.max_iters | 2 | 20000，按验证曲线调整。 |
| train_cfg.val_interval | 2 | 1000。 |
| param_scheduler[0].end | 2 | 与计划训练总次数一致，例如 20000。 |
| default_hooks.logger.interval | 1 | 例如 50。 |
| default_hooks.checkpoint.interval | 1 | 例如 1000。 |
| work_dir / test_work_dir | 本次 train / test | 改为新实验的 train / test 目录。 |
| load_from / resume | None / False | 新实验保持 None / False；续训再按原实验设置。 |
| test_checkpoint | 当前 best_mIoU_iter_2.pth | 训练完成后填写该次实际最佳权重。 |

output_dir 已写为 f'{test_work_dir}/masks'，在文件中更改 test_work_dir 后，重新加载配置会同步变更预测目录。命令始终使用同一个 massformer_r50_solar_general.py，无需另建 quick 配置。

## 训练目录进一步核查：原始 loss 与实际参数更新

读取本次保存的 checkpoint 和 MMEngine 0.10.3 调度代码后，需将前文“两次更新”精确理解为两次训练循环及 optimizer.step 调用：本次实际上仅第 1 次使模型参数发生变化。

原因是当前 PolyLR 的 begin=0、end=2、eta_min=0；该版本内部 total_iters=end-begin-1=1，第 1 次训练后调度即将学习率降为 0。日志钩子先记录、调度钩子再执行、最后保存 checkpoint，所以第 1 行日志仍显示非零学习率，而 iter_1.pth 中优化器学习率已经为 0；第 2 次循环仍执行前向、反向和优化器步骤，但没有新的权重变化。

已逐项计算模型 state_dict 的内容哈希，iter_1.pth、iter_2.pth、best_mIoU_iter_2.pth 的模型参数完全一致。进一步将初始化与 iter_1 共有的 658 个条目逐项比较，其中 412 个条目变化，确认第一轮确有更新。它们的完整文件内容仍不同：iter 快照包含优化器、调度器、迭代号和日志历史，best 快照不包含优化器和调度器。本次唯一一次 val 发生在 iter=2，故保存的最佳模型名为 best_mIoU_iter_2.pth。

从 iter_2.pth 的 message_hub 读取原始 loss，并与 scalars.json 核对：

| 训练循环 | 原始批次 loss | 日志平滑 loss |
|---|---|---|
| 1 | 193.8103 | 193.8103 |
| 2 | 166.3069 | 180.0586 |

第二行日志 loss=(193.8102875+166.3069305)/2，是当前已有两次记录的平均值。不同图像的 loss 变化和日志均值变化，不等同于第二次产生了新的参数更新。

本次仅核查和补充说明，保留现有训练文件与配置，不重新运行或替换结果。若后续希望两次短训练均使用非零学习率，应延长调度，例如将 end 改为 3 或保留更长的正式调度；不能把当前两次循环理解为两次有效权重更新。

## 最新：TensorBoard 新配置重新运行（2026-11-23）

按用户要求重新执行当前主配置，保留上面的旧记录和旧运行目录。主配置当前输出改为 work_dirs/massformer_solar_general/tensorboard_run/train 和 test；PolyLR.end 改为 3，max_iters 仍为 2，解决第二轮零学习率问题。

实际执行以下两条命令，无其他命令行覆盖：

```bat
python tools/train.py configs/massformer/massformer_r50_solar_general.py
python tools/test.py configs/massformer/massformer_r50_solar_general.py
```

两条命令均成功，数据范围仍为 train/val/test 各前两对图像，使用本机 cuda:0。训练时间戳为 20261123_163000，测试时间戳为 20261123_163034。

| 核查项 | 本次结果 |
|---|---|
| 两次原始 loss | 193.8103、166.0430，与 TensorBoard 事件一致。 |
| 两次主干学习率 | 1e-5、5.3589e-6，均大于 0。 |
| 初始权重到 iter_1 | 共有状态条目中有 412 个变化。 |
| iter_1 到 iter_2 | 414 个模型状态条目变化，第二轮有效更新。 |
| val mIoU / mDice | 40.33% / 44.65%。 |
| test mIoU / mDice | 49.82% / 51.16%。 |
| test 光伏类 IoU / Dice | 1.71% / 3.37%。 |
| 测试权重 | tensorboard_run/train/best_mIoU_iter_2.pth。 |
| 预测 PNG | tensorboard_run/test/masks 下 2 张，512×512，0/1。 |

TensorBoard EventAccumulator 已确认 train/loss 包含 step=1、2，train/lr 两点均非零，包含 val/mIoU 和 config/text_summary。事件文件分别由程序自动生成在 train/20261123_163000/vis_data 和 test/20261123_163034/vis_data；测试原生指标位于 test/20261123_163034/20261123_163034.json，测试事件本次仅配置文本。

本次仍为极短流程核查，分数不代表正式精度。未新增人工 summary.json、cleanup.json 或控制台重定向文件；新增产物均来自 train.py/test.py 的原生运行。当前主配置保留这次的输出路径、快速参数和 TensorBoard 设置。

## 固定四张验证图的中间效果保存（2026-11-23）

用户选择验证集排序前四张，名称在主配置 fixed_vis_names 中显式保存。新增固定样本 Hook 复用原模型 val_step 和 SegLocalVisualizer，将同一张图的 GT 叠加与预测叠加左右拼接，经原生 Local/TensorBoard 后端写出。正式训练间隔为 1000；当前两步快速模式 interval=1，用于验证两个训练阶段。

本次仍直接执行主配置的 train.py 和 test.py，输出统一改为 work_dirs/massformer_solar_general/fixed_visualization_run，保留之前实验。训练时间戳 20261123_170130，测试时间戳 20261123_170316；训练两轮、val 评估两张、test 评估两张，额外快照在每轮结束时预测固定四张，快照不参与 val 分数计算。

已核查：

- 17 项代码测试通过，包括固定文件/step、间隔、重复回调以及异常情况下恢复训练状态。
- train/20261123_170130/vis_data/vis_image 中共八张 fixed_val PNG，四张样本在 step=1、2 各一张；均为 1024×512 RGB，左 GT、右预测，alpha=0.5。
- TensorBoard Images 四个标签各有 step 1、2，与对应本地 PNG 像素一致；同一图两个阶段的 GT 面板完全一致，确认使用固定文件和固定几何处理。
- 已目视检查第一张对照图：左侧真实光伏区域为橙色，右侧为当前模型预测叠加。短训练下预测不佳属当前模型结果，图像仅用于观察随训练变化。
- 训练、验证和测试成功完成，val mIoU=40.33%、test mIoU=49.82%，不作为正式精度；四张图的额外前向没有更新模型或改变训练/冻结 BN 状态。

查看本次图片与曲线：

```bat
tensorboard --logdir work_dirs/massformer_solar_general/fixed_visualization_run --port 6006
```

在 Images 选择 fixed_val_01～04 标签，通过 step 查看不同迭代。完整训练前将 custom_hooks[0].interval 改为 1000，并恢复三个数据集 indices=None 及正式训练计划，详见 use.md 第 5、9 节。

## 四栏对比布局更新（2026-11-23）

按用户要求，将固定样本图改为 Original | GT | Prediction | Overlay 四栏，标题高 32 像素。GT/预测为黑白二值图，白色表示光伏；Overlay 仅给预测光伏区域混合橙色，背景及原图栏不压暗。采用原模型 val_step 和现有可视化器 add_image 后端，不改变训练、四张固定样本或保存频率。

当前主配置直接输出到 fixed_visualization_run/four_panel_run/train 和 test；旧双栏图片保留。使用原 train.py、test.py 仅传主配置路径完成两轮训练、两张 val 和两张 test 检查。新训练时间戳为 20261123_172349，本地图片位于其 vis_data/vis_image。

- 18 项代码测试通过，新增四栏尺寸、掩码语义、原图像素保真和仅预测前景混色检查。
- step=1、2 各四张，共八张 2048×544 RGB PNG；TensorBoard 各标签均有这两个 step，事件图像与本地 PNG 像素一致。
- 原图和 GT 在两个 step 保持相同，掩码像素只有 0/255；预测为背景的位置，Overlay 像素与原图完全一致；已目视检查实际四栏示例。
- 当前 6006 端口的日志根目录本来就是 fixed_visualization_run，不需要重启服务。网页 images/tags 和 images 元数据接口已确认加载 four_panel_run/train/20261123_172349/vis_data 及四个图像标签、step 1/2。

在 http://127.0.0.1:6006/#images 刷新并选择上述 four_panel_run 运行即可查看新布局。当前仍只训练两轮，预测质量不代表正式效果。
