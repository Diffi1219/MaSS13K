"""Watch an existing training process and send a Windows desktop notification."""
import argparse
import base64
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def process_identity(pid):
    """Include process start time to distinguish reused PIDs on either OS."""
    import psutil
    try:
        process = psutil.Process(pid)
        command = ' '.join(process.cmdline()).replace('\\', '/')
        if process.status() == psutil.STATUS_ZOMBIE or 'tools/train.py' not in command:
            return None
        return process.create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def inspect_run(output_dir, alive, epochs=50):
    """A final checkpoint alone does not imply final evaluation succeeded."""
    with (output_dir / 'console.log').open('rb') as stream:
        stream.seek(max(0, stream.seek(0, 2) - 128 * 1024))
        tail = stream.read().decode('utf-8', errors='replace')
    progress = re.findall(
        r'Epoch\(train\)\s*\[(\d+)\]\s*\[\s*(\d+)/(\d+)\]', tail)
    checkpoint = output_dir / f'epoch_{epochs}.pth'
    final_evaluation = any(int(done) == int(total) for done, total in re.findall(
        rf'Epoch\(val\)\s*\[{epochs}\]\s*\[\s*(\d+)/(\d+)\][^\n]*\bmIoU:', tail))
    complete = checkpoint.is_file() and final_evaluation
    status = 'running' if alive else ('completed' if complete else 'failed')
    result = dict(status=status, observed_at=now(), process_alive=alive,
                  final_checkpoint=str(checkpoint),
                  final_checkpoint_exists=checkpoint.is_file(),
                  final_evaluation_logged=final_evaluation,
                  automatic_test=False, automatic_resnet=False)
    if progress:
        epoch, step, total = map(int, progress[-1])
        result.update(epoch=epoch, iteration_in_epoch=step,
                      iterations_per_epoch=total)
    if status == 'failed':
        result['reason'] = ('Training exited before both the final checkpoint '
                            'and final evaluation were observed.')
    return result


def notify_windows(title, message):
    xml = ('<toast><visual><binding template="ToastGeneric"><text>' +
           escape(title) + '</text><text>' + escape(message) +
           '</text></binding></visual></toast>')
    ps_xml = xml.replace("'", "''")
    script = f"""$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml('{ps_xml}')
$toast = New-Object Windows.UI.Notifications.ToastNotification $xml
$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Microsoft.WindowsPowerShell')
$notifier.Show($toast)
Write-Output 'Notification submitted'
"""
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    powershell = shutil.which('powershell.exe') or shutil.which('powershell')
    if powershell is None:
        fallback = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
        if not fallback.is_file():
            raise RuntimeError('Windows PowerShell not found for desktop notification')
        powershell = str(fallback)
    result = subprocess.run(
        [powershell,
         '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
        capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
    return dict(status='submitted', submitted_at=now())


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--send-test', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.send_test:
        result = notify_windows('MaSS13K 提醒已设置',
                                f'训练仍在进行。{args.epochs} 个 epoch 结束或异常退出后，'
                                '将通过 Windows 桌面通知提醒你。')
        save_json(output / 'notification_setup.json', result)
        print(json.dumps(result), flush=True)
        return
    if args.interval < 1:
        parser.error('--interval must be positive')
    record = json.loads((output / 'training_process.json').read_text())
    pid = record['training_pid']
    identity = process_identity(pid)
    own_pid_file = output / 'completion_watcher.pid'
    if own_pid_file.exists():
        import psutil
        previous = int(own_pid_file.read_text())
        try:
            command = ' '.join(psutil.Process(previous).cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            command = ''
        if 'watch_pv_training.py' in command:
            raise RuntimeError(f'Completion watcher already running: {previous}')
    own_pid_file.write_text(str(os.getpid()))
    while True:
        alive = identity is not None and process_identity(pid) == identity
        status = inspect_run(output, alive, epochs=args.epochs)
        status.update(training_pid=pid, watcher_pid=os.getpid())
        save_json(output / 'training_status.json', status)
        print(json.dumps(status), flush=True)
        if status['status'] != 'running':
            completed = status['status'] == 'completed'
            title = 'MaSS13K 训练完成' if completed else 'MaSS13K 训练异常退出'
            message = (f'{args.epochs} 个 epoch 训练及最后一轮评估已完成，'
                       f'epoch_{args.epochs}.pth 已保存。后续独立测试尚未自动开始。'
                       if completed else
                       '训练未满足完整完成条件，请查看输出目录中的 console.log。')
            (output / 'training_notice.txt').write_text(
                f'{title}\n{now()}\n{message}\n', encoding='utf-8')
            try:
                status['notification'] = notify_windows(title, message)
            except Exception as error:
                status['notification'] = dict(status='failed', error=str(error))
            save_json(output / 'training_status.json', status)
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
