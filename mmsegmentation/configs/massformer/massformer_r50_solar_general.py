"""General 光伏分割：Windows 单张 RTX A4000（第二张可见卡）完整数据训练配置。"""
_base_ = ['./massformer_r50_8xb2-90k_mass13k-1024x1024.py']  # 继承作者模型及其基础配置；本文件同名项覆盖继承值，未列出的网络结构继续继承。


# ============================================================================
# 1. 公共设置：train / val / test 共用
# ============================================================================
custom_imports = dict(  # 注册模型依赖、光伏数据读取和固定图像可视化组件。
    imports=[
        'mmdet.models',  # 模型依赖的 MMDetection 组件。
        'mmseg.datasets.solar',  # General 数据集及图像、标签读取组件。
        'mmseg.engine.hooks.fixed_sample_visualization_hook',  # 固定四张图的效果图保存组件。
    ],
    allow_failed_imports=False)  # 组件导入失败即报错。
data_root = 'C:/ZZworkspace/zyworkspace/project/datasets/dis/General'  # 数据根目录，包含 train、val、test 下的 image 和 label。
device = 'cuda:1'  # 目标电脑第二张可见 RTX A4000；未设置 CUDA_VISIBLE_DEVICES 时对应 nvidia-smi 的 GPU 1。
launcher = 'none'  # 单进程单卡；原 train.py/test.py 据此不启动分布式通信。
randomness = dict(seed=42)  # 随机种子，影响初始化、采样和增强；不保证跨设备完全一致。
env_cfg = dict(  # Windows 运行环境，通常无需随训练任务调整。
    cudnn_benchmark=False,  # 是否搜索更快的 cuDNN 算法，开启后可能影响复现性。
    mp_cfg=dict(
        mp_start_method='spawn',  # Windows 子进程启动方式，保留 spawn。
        opencv_num_threads=0),  # 禁用 OpenCV 内部多线程，避免与加载进程竞争。
    dist_cfg=dict(backend='gloo'))  # 分布式后端；launcher='none' 时不启动通信。
vis_backends = [  # 同时记录本地 JSON/PNG 和 TensorBoard 事件。
    dict(type='LocalVisBackend'),  # 写入 vis_data 中的标量 JSON、配置和图片。
    dict(type='TensorboardVisBackend'),  # 写入事件文件，供 Scalars、Images 和 Text 展示。
]
visualizer = dict(vis_backends=vis_backends)  # 应用上述后端；仅改顶层 vis_backends 不会更新继承的可视化器。
log_processor = dict(  # 处理标量；实际写入频率由 default_hooks.logger.interval 控制。
    by_epoch=False,  # 日志横轴按迭代次数，匹配 IterBasedTrainLoop。
    window_size=1,  # 记录当前批次原始值；大于 1 会对窗口内数值求平均。
    log_with_hierarchy=True)  # 标签分为 train/、val/ 等层级，总训练损失为 train/loss。
num_classes = 2  # 背景和光伏两类；变更类别还需同步数据集元信息、标签和分类权重。
crop_size = (512, 512)  # 训练裁块及训练预处理尺寸，按（高，宽）填写；增大会增加显存占用。
model = dict(  # 在作者模型上覆盖光伏任务设置；训练、验证、测试共用。
    backbone=dict(
        init_cfg=None,  # 禁用骨干单独初始化下载，改用下方预训练权重。
        norm_cfg=dict(
            type='BN',  # 普通 BatchNorm，适配当前单卡流程。
            requires_grad=True),  # BN 的缩放和偏置参与梯度更新。
        norm_eval=True),  # 固定骨干 BN 运行统计，适配小批次微调，不冻结上述可学习参数。
    decode_head=dict(
        num_classes=num_classes,  # 分割头类别数，跟随顶层 num_classes。
        norm_eval=True,  # 固定解码头 BN 运行统计，适配小批次训练。
        loss_cls=dict(
            class_weight=[1.0, 1.0, 0.1],  # 背景、光伏、无目标 query 的分类权重；末项不是第三个分割类别。
            loss_weight=2.0),  # 分类损失系数，显式沿用作者值。
        loss_mask=dict(loss_weight=5.0),  # 掩码损失系数，显式沿用作者值。
        loss_dice=dict(loss_weight=5.0)),  # Dice 损失系数，显式沿用作者值；其余损失设置继续继承。
    data_preprocessor=dict(
        size=crop_size,  # 训练预处理目标尺寸，与训练裁块保持一致。
        test_cfg=dict(size_divisor=256)),  # 验证/测试补齐到该尺寸倍数，不是缩放成 256。
    test_cfg=dict(mode='whole'))  # 验证/测试整图预测；与顶层 test_cfg（测试循环）用途不同。


# ============================================================================
# 2. train 训练：tools/train.py 使用，更新模型参数
# ============================================================================
work_dir = 'work_dirs/massformer_solar_general/exp1/train'  # 训练权重、配置及时间戳日志目录；新实验换新目录，续训沿用原目录。
pretrained_checkpoint = 'C:/ZZworkspace/zyworkspace/project/weights/cv/MaSS13K/iter_80000.pth'  # 作者七类源权重；首次训练排除分类层，生成光伏初始化权重。
load_from = None  # 非空时直接加载该权重并优先于源权重转换；resume=False 时不恢复优化器和迭代进度。
resume = False  # True 恢复训练进度；load_from=None 时从 work_dir 的 last_checkpoint 查找，不控制测试权重。
train_pipeline = [  # 顺序执行训练读取和增强，不用于 val/test。
    dict(type='LoadSolarImage'),  # 读取 8 位 RGB 图像并转换为预处理器需要的通道顺序。
    dict(type='LoadSolarAnnotations'),  # 将单通道 0/1 或 0/255 标签统一成背景 0、光伏 1。
    dict(
        type='RandomFlip',  # 随机翻转，未指定方向时默认水平翻转。
        prob=0.5),  # 每张训练图执行翻转的概率，0 表示关闭。
    dict(type='PhotoMetricDistortion'),  # 随机亮度、对比度、饱和度等增强，删除该项可关闭。
    dict(
        type='RandomCrop',  # 同步裁剪图像和标签。
        crop_size=crop_size,  # 跟随公共裁块尺寸。
        cat_max_ratio=1.0),  # 裁块单一类别比例上限；1.0 不限制，调低会尝试采样类别更均衡的裁块。
    dict(
        type='GenerateEdge',  # 生成边缘分支需要的标签，保留该步骤。
        edge_width=2),  # 边缘标签宽度，单位为像素。
    dict(type='PackSegInputs'),  # 打包图像、标签及元信息，放在流程末尾。
]
train_dataloader = dict(  # 训练加载设置；当前单卡无梯度累积，每次迭代处理一个 batch。
    _delete_=True,  # 完全替换继承的数据加载配置，防止残留作者数据集参数。
    batch_size=4,  # 单张 A4000 每次更新输入四张 512×512 图像，保留当前设置；不累积梯度。
    num_workers=0,  # Windows 在主进程加载数据；需要并行加载时再按目标电脑 CPU/内存调整。
    persistent_workers=False,  # 是否保留加载子进程；num_workers=0 时必须为 False。
    sampler=dict(
        type='InfiniteSampler',  # 按迭代训练持续循环采样，遍历完数据后继续下一轮。
        shuffle=True),  # 每轮打乱训练样本顺序。
    dataset=dict(
        type='SolarGeneralDataset',  # General 光伏数据集，定义背景/光伏类别及配对规则。
        data_root=data_root,  # 跟随公共数据根目录。
        indices=None,  # 使用完整 train；临时检查可改为所需图像对数。
        data_prefix=dict(
            img_path='train/image',  # 训练图像子目录，相对于 data_root。
            seg_map_path='train/label'),  # 训练标签子目录，与图像同名配对。
        pipeline=train_pipeline))  # 使用上述训练增强流程。
optim_wrapper = dict(  # 常用优化参数显式列出，保持此前继承的有效值；其余设置继续继承。
    type='OptimWrapper',  # 普通精度优化器封装；原入口 --amp 可切换自动混合精度。
    optimizer=dict(
        type='AdamW',  # 优化器类型，改类型时需检查该优化器接受的参数。
        lr=0.0001,  # 基础学习率；骨干实际学习率再乘 backbone.lr_mult。
        weight_decay=0.05,  # 权重衰减强度，各参数组再乘 decay_mult。
        eps=1e-8,  # AdamW 数值稳定项，通常保持默认值。
        betas=(0.9, 0.999)),  # 一阶、二阶动量系数，通常沿用作者值。
    clip_grad=dict(
        max_norm=0.01,  # 梯度范数裁剪上限，日志 grad_norm 记录裁剪前的值。
        norm_type=2),  # 裁剪使用 L2 范数。
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),  # 骨干使用 0.1 倍基础学习率和 1 倍权重衰减。
            'query_embed': dict(lr_mult=1, decay_mult=0.0),  # query 位置嵌入使用基础学习率，不做权重衰减。
            'query_feat': dict(lr_mult=1, decay_mult=0.0),  # query 特征使用基础学习率，不做权重衰减。
            'level_embed': dict(lr_mult=1, decay_mult=0.0),  # 层级嵌入使用基础学习率，不做权重衰减。
        },
        norm_decay_mult=0.0))  # 归一化层参数不做权重衰减。
param_scheduler = [dict(  # 学习率计划；总训练次数变化时同步修改 end。
    type='PolyLR',  # 多项式学习率衰减。
    eta_min=0.0,  # 衰减后的最低学习率。
    power=0.9,  # 衰减曲线指数。
    begin=0,  # 从第 0 次迭代开始调度。
    end=20001,  # 与下方 max_iters=20000 配套；MMEngine 0.10.3 用 max_iters+1 避免最后一次训练学习率提前变为 0。
    by_epoch=False)]  # 每次迭代更新学习率，与训练循环一致。
train_cfg = dict(  # 训练循环及验证频率，验证使用第 3 节配置。
    type='IterBasedTrainLoop',  # 按迭代次数训练，一次迭代处理一个 batch。
    max_iters=20000,  # 正式训练起始方案：更新 20000 次，根据完整 val 的收敛情况调整并同步修改调度器 end。
    val_interval=1000)  # 每 1000 次更新评估完整 val，用于观察指标和选择最佳权重。
default_hooks = dict(  # 原生训练 Hook，其他计时、调度等 Hook 继续继承。
    logger=dict(
        interval=1,  # 每隔多少次训练写 .log、JSON 和 TensorBoard；1 表示每次迭代记录。
        log_metric_by_epoch=False),  # 验证指标按全局迭代号写入，与训练横轴一致。
    checkpoint=dict(
        interval=1000,  # 每 1000 次更新保存 iter_N.pth，与日志/图片间隔独立。
        save_best='mIoU',  # 按验证集该指标另存最佳权重，不使用 test 集选模型。
        max_keep_ckpts=3))  # 最多保留的常规迭代权重数，最佳权重单独管理；-1 表示不限制。


# ============================================================================
# 3. val 验证：训练中计算指标、选择最佳权重，不更新参数
# ============================================================================
val_pipeline = [  # 无随机增强；验证间隔见 train_cfg.val_interval。
    dict(type='LoadSolarImage'),  # 读取原始验证图像。
    dict(type='LoadSolarAnnotations'),  # 读取真实标签用于计算指标。
    dict(type='PackSegInputs'),  # 打包模型输入。
]
val_dataloader = dict(  # 验证日志随训练保存到 work_dir。
    _delete_=True,  # 完全替换继承的验证数据配置。
    batch_size=1,  # 每次验证前向的图像数，不影响训练 batch_size。
    num_workers=0,  # 验证数据加载子进程数。
    persistent_workers=False,  # num_workers=0 时必须为 False。
    sampler=dict(
        type='DefaultSampler',  # 遍历配置的验证子集一次。
        shuffle=False),  # 保持验证顺序固定。
    dataset=dict(
        type='SolarGeneralDataset',  # 与训练保持相同类别及标签规则。
        data_root=data_root,  # 跟随公共数据根目录。
        indices=None,  # 每次验证使用完整 val，不只抽取前两对。
        data_prefix=dict(
            img_path='val/image',  # 验证图像子目录，相对于 data_root。
            seg_map_path='val/label'),  # 验证标签子目录。
        pipeline=val_pipeline))  # 使用上述无随机增强流程。
val_cfg = dict(type='ValLoop')  # 原生验证循环，只前向算分，不更新参数。
val_evaluator = dict(  # 验证指标，最佳模型按 default_hooks.checkpoint.save_best 选择。
    type='IoUMetric',  # 原生分割评估器。
    iou_metrics=['mIoU', 'mDice', 'mFscore'])  # 输出 IoU、Dice、F-score 等指标，以百分比表示。
fixed_vis_names = [  # 验证集排序前四张的文件名主干，不含扩展名；不受 val_dataloader.indices 限制。
    'F47Mosaic_F47D001008_00000018',  # 第 1 张固定图，后续保存点使用同一张。
    'F48Mosaic_F48D001002_00000043',  # 第 2 张固定图。
    'F48Mosaic_F48D002001_00000057',  # 第 3 张固定图。
    'F48Mosaic_F48D002003_00000091',  # 第 4 张固定图。
]
custom_hooks = [dict(  # 额外前向保存原图/GT/预测/叠加四栏，不更新参数，也不代替 val 指标计算。
    type='FixedSampleVisualizationHook',  # 通过现有后端保存 PNG 和 TensorBoard Images。
    data_root=data_root,  # 固定样本路径的公共根目录。
    samples=[dict(
        img_path=f'val/image/{name}.tif',  # 原图相对路径，更换数据时同步修改目录和扩展名。
        seg_map_path=f'val/label/{name}.tif',  # GT 相对路径，必须与原图尺寸一致。
    ) for name in fixed_vis_names],  # 按上述文件名生成固定列表，不随机抽样。
    interval=1000,  # 每 1000 次更新保存固定四张验证图的四栏对比，不受 val_interval 控制。
    draw_last=True,  # 结束时未到间隔也保存最终效果，已在间隔点保存则不重复。
    alpha=0.5)]  # 预测光伏颜色的混合比例，0 为原图、1 为纯色；不影响原图/GT/预测栏。


# ============================================================================
# 4. test 测试：tools/test.py 单独执行，不参与训练及选模型
# ============================================================================
test_checkpoint = f'{work_dir}/iter_{train_cfg["max_iters"]}.pth'  # 默认测试最后一次更新的权重；评估验证集最佳模型时改为实际 best_mIoU_iter_N.pth，最佳迭代号无法预先确定。
test_work_dir = 'work_dirs/massformer_solar_general/exp1/test'  # 本实验测试日志、配置及指标 JSON 目录，与 exp1/train 配对。
test_pipeline = [  # 与 val 独立定义，便于分别调整测试预处理。
    dict(type='LoadSolarImage'),  # 读取测试原图。
    dict(type='LoadSolarAnnotations'),  # 读取测试 GT 用于算分，无标签图像请使用 infer。
    dict(type='PackSegInputs'),  # 打包模型输入。
]
test_dataloader = dict(  # 单独执行 tools/test.py 时使用，训练结束后不会自动执行。
    _delete_=True,  # 完全替换继承的测试数据配置。
    batch_size=1,  # 每次测试前向的图像数，不影响训练 batch_size。
    num_workers=0,  # 测试数据加载子进程数。
    persistent_workers=False,  # num_workers=0 时必须为 False。
    sampler=dict(
        type='DefaultSampler',  # 遍历配置的测试子集一次。
        shuffle=False),  # 保持测试顺序固定。
    dataset=dict(
        type='SolarGeneralDataset',  # 使用背景/光伏两类及 General 标签规则。
        data_root=data_root,  # 跟随公共数据根目录。
        indices=None,  # 单独执行 test.py 时评估完整 test，不参与训练或最佳权重选择。
        data_prefix=dict(
            img_path='test/image',  # 测试图像子目录，相对于 data_root。
            seg_map_path='test/label'),  # 测试标签子目录。
        pipeline=test_pipeline))  # 使用上述测试流程。
test_cfg = dict(type='TestLoop')  # 原生测试循环，与 model.test_cfg 的整图/滑窗预测设置不同。
test_evaluator = dict(  # 测试算分及预测掩码导出。
    type='IoUMetric',  # 原生分割评估器，读取测试 GT 算分。
    iou_metrics=['mIoU', 'mDice', 'mFscore'],  # 测试报告指标，与验证指标独立设置。
    output_dir=f'{test_work_dir}/masks')  # 仅保存预测类别掩码 PNG，不保存测试日志。
