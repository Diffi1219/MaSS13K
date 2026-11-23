# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work_dirs/.cache/matplotlib'))

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from tools.solar import prepare_checkpoint, require_cuda


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default=None,
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def load_config(args):
    """Keep config defaults unless a command-line option is explicitly given."""
    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.launcher is not None:
        cfg.launcher = args.launcher
    cfg.setdefault('launcher', 'none')

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    cfg.work_dir = osp.abspath(cfg.work_dir)

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'
    elif args.amp is False and cfg.optim_wrapper.type == 'AmpOptimWrapper':
        cfg.optim_wrapper.type = 'OptimWrapper'
        for key in ('loss_scale', 'dtype', 'use_fsdp'):
            cfg.optim_wrapper.pop(key, None)

    # resume training
    if args.resume is not None:
        cfg.resume = args.resume
    return cfg


def prepare_training_checkpoint(cfg):
    """Opt-in MaSS13K transfer; ordinary configs keep native loading behavior."""
    if cfg.get('resume', False) or cfg.get('load_from') or not cfg.get('pretrained_checkpoint'):
        return
    if cfg.launcher != 'none':
        raise ValueError('For distributed transfer, prepare the checkpoint once with '
                         'tools/solar.py prepare and set load_from in the config.')
    destination = Path(cfg.work_dir) / 'init_from_mass13k.pth'
    cfg.load_from = str(prepare_checkpoint(cfg.pretrained_checkpoint, destination))


class WindowsRunner(Runner):
    """Keep native training; tolerate localized compiler diagnostics in logs."""

    def _log_env(self, env_cfg):
        try:
            super()._log_env(env_cfg)
        except UnicodeDecodeError as error:
            import platform
            import torch
            self.logger.warning(f'Compiler environment text could not be decoded: {error}')
            self.logger.info(dict(Python=sys.version, platform=platform.platform(),
                                  PyTorch=torch.__version__, CUDA=torch.version.cuda,
                                  env_cfg=env_cfg, seed=self.seed, launcher=self.launcher))
            self.logger.info(f'Config:\n{self.cfg.pretty_text}')


def main():
    args = parse_args()
    cfg = load_config(args)
    # Distributed launchers choose their device using LOCAL_RANK as before.
    if cfg.launcher == 'none' and cfg.get('device') is not None:
        require_cuda(cfg.device)
    prepare_training_checkpoint(cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner_class = WindowsRunner if os.name == 'nt' else Runner
        runner = runner_class.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
