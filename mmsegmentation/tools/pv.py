"""Portable, single-GPU PV training entry point (Windows and Linux)."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(data_root='../../../../datasets/dis/General',
                pretrained_checkpoint='../../../../weights/cv/MaSS13K/iter_80000.pth',
                work_dir='work_dirs/massformer_pv_general_pretrained',
                batch_size=2, num_workers=2, epochs=50, seed=42,
                test_images=64, sample_names=[])


def absolute(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def settings(args):
    values = DEFAULTS.copy()
    source = absolute(args.settings) if args.settings else ROOT / 'pv_settings.local.json'
    if args.settings or source.exists():
        local = json.loads(source.read_text(encoding='utf-8-sig'))
        unknown = set(local) - set(values)
        if unknown:
            raise ValueError(f'Unknown settings: {sorted(unknown)}')
        values.update(local)
    for key in values:
        override = getattr(args, key, None)
        if override is not None or (key == 'test_images' and hasattr(args, key)):
            values[key] = override
    for key in ('data_root', 'pretrained_checkpoint', 'work_dir'):
        values[key] = str(absolute(values[key]))
    if values['batch_size'] < 2:
        raise ValueError('Training batch_size must be >= 2 (decoder BatchNorm).')
    if values['epochs'] < 2 or values['num_workers'] < 0:
        raise ValueError('epochs must be >= 2 and num_workers >= 0.')
    if values['test_images'] is not None and values['test_images'] < 0:
        raise ValueError('test_images must be nonnegative or all.')
    return values


def latest_checkpoint(work_dir):
    """Resolve copied Linux/Windows last_checkpoint files on the current host."""
    directory = Path(work_dir)
    marker = directory / 'last_checkpoint'
    if marker.exists():
        saved = marker.read_text(encoding='utf-8').strip()
        relocated = directory / saved.replace('\\', '/').split('/')[-1]
        # Prefer the current experiment directory after copying to another PC.
        if relocated.is_file():
            return relocated.resolve()
    candidates = []
    for path in directory.glob('epoch_*.pth'):
        match = re.fullmatch(r'epoch_(\d+)\.pth', path.name)
        if match:
            candidates.append((int(match[1]), path))
    if not candidates:
        raise FileNotFoundError(f'No epoch checkpoint in {directory}')
    return max(candidates, key=lambda item: item[0])[1].resolve()


def build_config(values, testing=False):
    from mmengine.config import Config
    name = 'massformer_r50_pv_test-512x512.py' if testing else 'massformer_r50_pv_pretrained-512x512.py'
    cfg = Config.fromfile(ROOT / 'configs' / 'massformer' / name)
    cfg.data_root = values['data_root']
    cfg.load_from = values['pretrained_checkpoint']
    cfg.resume = False
    cfg.work_dir = str(Path(values['work_dir']) / 'test') if testing else values['work_dir']
    cfg.randomness.seed = values['seed']
    cfg.max_epochs = values['epochs']
    cfg.train_cfg.max_epochs = values['epochs']
    cfg.train_cfg.dynamic_intervals = [(values['epochs'], 1)]
    cfg.param_scheduler[1].end = values['epochs']
    cfg.param_scheduler[1].T_max = values['epochs'] - 1
    for split in ('train', 'val', 'test'):
        loader = cfg[f'{split}_dataloader']
        loader.dataset.data_root = values['data_root']
        loader.num_workers = values['num_workers']
        loader.persistent_workers = values['num_workers'] > 0
        if values['num_workers'] == 0:
            loader.pop('prefetch_factor', None)
    cfg.train_dataloader.batch_size = values['batch_size']
    cfg.fixed_train_dataset.data_root = values['data_root']
    if testing:
        cfg.custom_hooks[0].max_images = values['test_images']
        cfg.test_evaluator.output_dir = str(Path(cfg.work_dir) / 'pred_masks')
    else:
        cfg.custom_hooks[0].dataset.data_root = values['data_root']
        if values['sample_names']:
            cfg.custom_hooks[0].sample_names = values['sample_names']
    # Explicitly keep FP32; --amp is deliberately absent from this entry point.
    assert cfg.optim_wrapper.type == 'OptimWrapper'
    return cfg


def audit_data(root, full=False):
    import cv2
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    report, stems = {}, {}
    for split in ('train', 'val', 'test'):
        directory = Path(root) / split
        images = {p.name: p for p in (directory / 'image').glob('*.tif')}
        labels = {p.name: p for p in (directory / 'label').glob('*.tif')}
        if not images or images.keys() != labels.keys():
            raise ValueError(f'{split}: empty or unmatched image/label files; '
                             f'missing labels={list(images.keys()-labels.keys())[:5]}, '
                             f'missing images={list(labels.keys()-images.keys())[:5]}')
        stems[split] = set(images)
        names = sorted(images)
        checked = names if full else [names[i] for i in sorted(set(
            np.linspace(0, len(names)-1, min(16, len(names)), dtype=int)))]

        def inspect(name):
            image = cv2.imread(str(images[name]), cv2.IMREAD_UNCHANGED)
            label = cv2.imread(str(labels[name]), cv2.IMREAD_UNCHANGED)
            if image is None or label is None:
                raise ValueError(f'Unreadable TIFF: {directory}/{name}')
            if image.shape != (512, 512, 3) or image.dtype != np.uint8:
                raise ValueError(f'{images[name]}: expected 512x512 RGB uint8; got {image.shape}, {image.dtype}')
            if label.shape != (512, 512) or label.dtype != np.uint8 or not np.isin(label, [0, 1]).all():
                raise ValueError(f'{labels[name]}: expected single-channel uint8 values 0/1')
            return int(label.sum()), label.size

        with ThreadPoolExecutor(max_workers=4) as pool:
            pixels = list(pool.map(inspect, checked))
        report[split] = dict(pairs=len(names), checked=len(checked),
                             pv_pixel_percent=round(100*sum(x[0] for x in pixels)/sum(x[1] for x in pixels), 3))
    overlaps = {f'{a}/{b}': len(stems[a] & stems[b])
                for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    report['filename_overlap'] = overlaps
    # Filenames are only a first check; equal names do not establish equal contents.
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if any(overlaps.values()):
        raise ValueError('Split filenames overlap; check data separation before training.')
    return report


def check_gpu():
    import torch
    import mmcv
    import mmengine
    import mmdet
    from mmcv.ops import ModulatedDeformConv2d, point_sample
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: install the CUDA PyTorch/MMCV wheels and check NVIDIA driver.')
    device = torch.cuda.get_device_properties(0)
    print(f'GPU: {device.name}, {device.total_memory/1024**3:.1f} GiB')
    print(f'torch={torch.__version__}, mmcv={mmcv.__version__}, '
          f'mmengine={mmengine.__version__}, mmdet={mmdet.__version__}')
    op = ModulatedDeformConv2d(4, 4, 3, padding=1).cuda()
    x = torch.randn(2, 4, 8, 8, device='cuda', requires_grad=True)
    offset = torch.zeros(2, 18, 8, 8, device='cuda', requires_grad=True)
    mask = torch.full((2, 9, 8, 8), 0.5, device='cuda', requires_grad=True)
    output = op(x, offset, mask)
    output.square().mean().backward()
    sampled = point_sample(x.detach(), torch.rand(2, 8, 2, device='cuda'))
    if not all(torch.isfinite(t).all() for t in (output, x.grad, offset.grad, mask.grad, sampled)):
        raise RuntimeError('CUDA operator produced non-finite values.')
    torch.cuda.synchronize()
    print('CUDA deformable convolution forward/backward and point_sample: PASS', flush=True)


def launch(command, log_path, record_process=False):
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    print('Running: ' + subprocess.list2cmdline(command), flush=True)
    with Path(log_path).open('a', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        if record_process:
            (Path(log_path).parent / 'training_process.json').write_text(
                json.dumps(dict(training_pid=process.pid, command=command), indent=2), encoding='utf-8')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f'Command exited with status {code}; see {log_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'train', 'test', 'tensorboard'])
    parser.add_argument('--settings', help='JSON settings path, relative to mmsegmentation/')
    for key in ('data_root', 'pretrained_checkpoint', 'work_dir'):
        parser.add_argument('--' + key.replace('_', '-'))
    for key in ('epochs', 'batch_size', 'num_workers', 'seed'):
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    parser.add_argument('--test-images', type=lambda x: None if x == 'all' else int(x), default=argparse.SUPPRESS)
    parser.add_argument('--resume', nargs='?', const='auto', help='latest checkpoint, or an explicit checkpoint path')
    parser.add_argument('--checkpoint', help='checkpoint for test; default: latest in work-dir')
    parser.add_argument('--full-data', action='store_true', help='validate every image and label, not just sampled contents')
    parser.add_argument('--skip-gpu', action='store_true', help='check files/config only')
    parser.add_argument('--smoke', action='store_true', help='2 training epochs on 2 batches, then test 2 images; isolated temporary output')
    parser.add_argument('--port', type=int, default=6006)
    args = parser.parse_args()
    values = settings(args)
    if args.action == 'tensorboard': # 运行tensorboard
        subprocess.run([sys.executable, '-m', 'tensorboard.main', '--logdir', values['work_dir'],
                        '--host', 'localhost', '--port', str(args.port)], check=True)
        return
    cfg = build_config(values, testing=args.action == 'test')
    if args.action == 'check':
        audit_data(values['data_root'], args.full_data)
        if not Path(values['pretrained_checkpoint']).is_file():
            raise FileNotFoundError(values['pretrained_checkpoint'])
        print(f'FP32, batch={values["batch_size"]}, workers={values["num_workers"]}, '
              f'epochs={values["epochs"]}; work_dir={values["work_dir"]}', flush=True)
        if not args.skip_gpu:
            check_gpu()
        if args.smoke:
            if args.skip_gpu:
                raise ValueError('--smoke requires GPU; remove --skip-gpu')
            smoke(cfg)
        return
    if args.smoke:
        raise ValueError('--smoke is only available with check')
    checkpoint = None
    if args.action == 'test':
        checkpoint = absolute(args.checkpoint) if args.checkpoint else latest_checkpoint(values['work_dir'])
    elif args.resume:
        checkpoint = latest_checkpoint(values['work_dir']) if args.resume == 'auto' else absolute(args.resume)
        cfg.resume = True
    elif (Path(values['work_dir']) / 'last_checkpoint').exists() or list(Path(values['work_dir']).glob('epoch_*.pth')):
        raise ValueError('Work directory already contains checkpoints: use --resume or a new --work-dir.')
    if checkpoint:
        cfg.load_from = str(checkpoint)
    if not Path(cfg.load_from).is_file():
        raise FileNotFoundError(cfg.load_from)
    output = Path(cfg.work_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.action == 'train':
        # Pairing/format preflight before an expensive training run.
        audit_data(values['data_root'])
        manifest = Path(values['work_dir']) / 'pv_settings.resolved.json'
        if cfg.resume and manifest.exists():
            old = json.loads(manifest.read_text(encoding='utf-8'))
            for key in ('epochs', 'batch_size', 'seed'):
                if old[key] != values[key]:
                    raise ValueError(f'Resume setting changed: {key}; preserve the original training schedule.')
        manifest.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    config_path = output / ('test_config.py' if args.action == 'test' else 'train_config.py')
    cfg.dump(str(config_path))
    command = [sys.executable, str(ROOT / 'tools' / f'{args.action}.py'), str(config_path)]
    if args.action == 'test':
        command.append(str(checkpoint))
    elif cfg.resume:
        command.append('--resume')
    launch(command, output / 'console.log', record_process=args.action == 'train')


def smoke(cfg):
    """Use real data/model/checkpoint, without changing experiment outputs."""
    import tempfile
    from mmengine.config import Config
    with tempfile.TemporaryDirectory(prefix='massformer_smoke_') as directory:
        output = Path(directory)
        cfg.work_dir = str(output / 'train')
        cfg.log_level = 'WARNING'
        cfg.train_cfg.max_epochs = 1
        cfg.train_cfg.val_interval = 1
        cfg.train_cfg.dynamic_intervals = []
        # Keep two complete iterations per epoch for any supported batch size.
        train_samples = 2 * cfg.train_dataloader.batch_size
        cfg.train_dataloader.dataset.indices = train_samples
        cfg.val_dataloader.dataset.indices = 2
        cfg.custom_hooks[0].interval = 1
        cfg.custom_hooks[0].num_samples = 2
        cfg.custom_hooks[0].dataset.indices = train_samples
        train_config = output / 'train.py'
        cfg.dump(str(train_config))
        launch([sys.executable, str(ROOT / 'tools/train.py'), str(train_config)], output / 'train.log')
        # A second process must restore model, optimizer, counters and schedule.
        cfg.train_cfg.max_epochs = 2
        cfg.load_from = str(latest_checkpoint(output / 'train'))
        cfg.resume = True
        cfg.dump(str(train_config))
        launch([sys.executable, str(ROOT / 'tools/train.py'), str(train_config), '--resume'], output / 'train.log')
        import torch
        saved = torch.load(output / 'train/epoch_2.pth', map_location='cpu')
        assert saved['meta']['epoch'] == 2 and saved['meta']['iter'] == 4
        assert 'optimizer' in saved and 'param_schedulers' in saved
        assert len(saved['param_schedulers']) == 2
        assert all(torch.isfinite(value).all() for value in saved['state_dict'].values())
        del saved
        test = Config.fromfile(train_config)
        test.work_dir = str(output / 'test')
        test.load_from = str(output / 'train/epoch_2.pth')
        test.resume = False
        test.test_dataloader.dataset.indices = 2
        test.custom_hooks = [dict(type='TestTensorboardHook', checkpoint_epoch=None, max_images=1)]
        test.test_evaluator.output_dir = str(output / 'test/pred_masks')
        test_path = output / 'test.py'
        test.dump(str(test_path))
        launch([sys.executable, str(ROOT / 'tools/test.py'), str(test_path), test.load_from], output / 'test.log')
        report = json.loads((output / 'test/metrics.json').read_text(encoding='utf-8'))
        assert report['images'] == 2 and report['visualized_images'] == 1
        print('SMOKE PASS: FP32 training, spawn workers, checkpoint resume, validation, test and TensorBoard. '
              'Temporary smoke outputs will be removed.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
