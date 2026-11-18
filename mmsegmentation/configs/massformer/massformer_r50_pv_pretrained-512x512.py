"""Experiment B: same training settings, MaSS13K full-model initialization."""
_base_ = ['./massformer_r50_pv-512x512.py']

model = dict(backbone=dict(init_cfg=None))
# MMEngine loads all shape-compatible tensors; the 7-class classifier has
# incompatible weight/bias shapes and keeps its fresh 2-class initialization.
# This is fine-tuning: do not resume the source optimizer, schedule or epoch.
load_from = '/home/diffi/workspace/project/weights/cv/MaSS13K/iter_80000.pth'
resume = False
work_dir = 'work_dirs/massformer_pv_pretrained_50ep'

# Keep a recoverable full checkpoint every epoch; image logging and evaluation
# retain their five-epoch intervals from the shared configuration.
default_hooks = dict(checkpoint=dict(
    interval=1, save_optimizer=True, save_param_scheduler=True))
