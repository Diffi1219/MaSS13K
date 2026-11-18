"""Run the approved two 50-epoch FP32 experiments sequentially on one GPU."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

from mmengine.config import Config


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', default='work_dirs/pv_comparison_50ep')
    parser.add_argument('--order', choices=['resnet-first', 'pretrained-first'],
                        default='resnet-first')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    state_dir = Path(args.state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / 'status.json'
    if state_path.exists():
        raise RuntimeError(f'Existing comparison state: {state_path}')
    experiments = []
    sources = [
            ('resnet50', 'massformer_r50_pv-512x512.py'),
            ('mass13k_pretrained', 'massformer_r50_pv_pretrained-512x512.py')]
    if args.order == 'pretrained-first':
        sources.reverse()
    for name, filename in sources:
        cfg = Config.fromfile(str(root / 'configs/massformer' / filename))
        assert cfg.train_cfg.max_epochs == 50
        assert cfg.optim_wrapper.type == 'OptimWrapper'
        assert cfg.train_dataloader.batch_size == 2
        assert cfg.train_dataloader.dataset.get('indices') is None
        assert cfg.val_dataloader.dataset.get('indices') is None
        snapshot = state_dir / f'{name}_config.py'
        cfg.dump(str(snapshot))
        experiments.append(dict(name=name, config=str(snapshot),
                                work_dir=str((root / cfg.work_dir).resolve()),
                                status='pending'))
    state = dict(queue_pid=os.getpid(), started_at=now(), status='running',
                 order=args.order,
                 experiments=experiments)

    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(state_path)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    save()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', MPLCONFIGDIR=str(state_dir / 'matplotlib'))
    for experiment in experiments:
        work_dir = Path(experiment['work_dir'])
        work_dir.mkdir(parents=True, exist_ok=True)
        log_path = work_dir / 'console.log'
        with log_path.open('a', buffering=1) as log:
            process = subprocess.Popen(
                [sys.executable, '-u', 'tools/train.py', experiment['config'],
                 '--work-dir', str(work_dir)],
                cwd=root, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            experiment.update(status='running', pid=process.pid,
                              started_at=now(), console_log=str(log_path))
            save()
            print(json.dumps(experiment), flush=True)
            try:
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                code = process.wait()
                experiment.update(status='interrupted', exit_code=code)
                state['status'] = 'interrupted'
                save()
                return
        experiment.update(status='complete' if code == 0 else 'failed',
                          exit_code=code, finished_at=now())
        if code != 0:
            state['status'] = 'failed'
            save()
            raise SystemExit(code)
        save()
    state.update(status='complete', finished_at=now())
    save()


if __name__ == '__main__':
    main()
