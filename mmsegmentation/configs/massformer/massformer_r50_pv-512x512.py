_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/mass13k.py']
custom_imports = dict(imports='mmdet.models', allow_failed_imports=False)
# crop_size = (1024,1024)
crop_size = (512,512)
data_root = '/home/diffi/workspace/project/datasets/dis/General'
metainfo = dict(
    classes=('bg', 'pv'),
    palette=[
        [0, 0, 0],
        [255, 255, 255],
    ]
)   
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))
# num_classes = 7
num_classes = 2
model = dict(
    type='EncoderDecoder_multi',
    ratio = [1,0],
    data_preprocessor=data_preprocessor,
    backbone=dict( # 主干网络
        type='ResNet_m',
        depth=50,
        deep_stem=False,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        # Single GPU, batch size 2: keep pretrained BatchNorm statistics fixed.
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='/home/notebook/data/personal/S9055029/cache/resnet50-0676ba61.pth')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/diffi/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')),
        
    decode_head=dict(
        type='MaSSFormerHead',
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,

        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True),

        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=6,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='GELU'),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),

        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),

        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    
    dict(type='RandomFlip', prob=0.5,  direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=2),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(4096, 3072), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='PVDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='val/image',
            seg_map_path='val/label'
        ),
        img_suffix='.tif',
        seg_map_suffix='.tif',
        # metainfo=metainfo,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='PVDataset',
        test_mode=True,
        data_root=data_root,
        data_prefix=dict(
            img_path='train_old/image',
            seg_map_path='train_old/label'
        ),
        img_suffix='.tif',
        seg_map_suffix='.tif',
        # metainfo=metainfo,
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='PVDataset',
        test_mode=True,
        data_root=data_root,
        data_prefix=dict(
            img_path='train_old/image',
            seg_map_path='train_old/label'
        ),
        img_suffix='.tif',
        seg_map_suffix='.tif',
        # metainfo=metainfo,
        pipeline=test_pipeline
    )
)
# optimizer
embed_multi = dict(lr_mult=1, decay_mult=0.0)
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
# learning policy
max_epochs = 50
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.1,
        begin=0, end=1, by_epoch=True, convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        begin=1,
        end=max_epochs,
        T_max=max_epochs - 1,
        eta_min=1e-6,
        by_epoch=True, convert_to_iter_based=True)
]

# Intentionally train on General/val and evaluate on General/train_old.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                 val_interval=5)
val_cfg = dict(type='ValLoop', fp16=False)
test_cfg = dict(type='TestLoop', fp16=False)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5,
        max_keep_ckpts=2, save_best=None),
    sampler_seed=dict(type='DistSamplerSeedHook'))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'],
                     nan_to_num=0)
test_evaluator = val_evaluator
log_processor = dict(by_epoch=True)
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', name='visualizer',
                  vis_backends=vis_backends)

# These are training images. Use a deterministic pipeline so the same view is
# compared at epochs 0 (initial checkpoint), 5, 10, 15, ... .
fixed_train_dataset = dict(
    type='PVDataset', data_root=data_root,
    data_prefix=dict(img_path='val/image', seg_map_path='val/label'),
    img_suffix='.tif', seg_map_suffix='.tif', test_mode=True,
    pipeline=test_pipeline)
custom_hooks = [dict(
    type='FixedTrainVisualizationHook', interval=5,
    dataset=fixed_train_dataset,
    sample_names=[
        'F47Mosaic_F47D001008_00000018.tif',
        'F48Mosaic_F48D001002_00000043.tif',
        'F48Mosaic_F48D002001_00000057.tif',
        'F48Mosaic_F48D002003_00000091.tif',
    ])]
randomness = dict(seed=42)
work_dir = 'work_dirs/massformer_pv_resnet50_50ep'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

# pretrained checkpoint
# Experiment A: cached ImageNet backbone, fresh segmentation head.
load_from = None
resume = False
