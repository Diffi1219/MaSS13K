"""Completion detection must distinguish training, evaluation and failures."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    'watcher', Path(__file__).resolve().parents[1] / 'tools/watch_pv_training.py')
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


class TestCompletionDetection(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        (self.output / 'console.log').write_text(
            'Epoch(train) [2][ 20/939] loss: 1.0\n')

    def test_checkpoint_before_evaluation_is_not_completion(self):
        (self.output / 'epoch_50.pth').touch()
        self.assertEqual(watcher.inspect_run(self.output, True)['status'],
                         'running')
        self.assertEqual(watcher.inspect_run(self.output, False)['status'],
                         'failed')

    def test_final_evaluation_and_checkpoint_require_process_exit(self):
        (self.output / 'epoch_50.pth').touch()
        (self.output / 'console.log').write_text(
            'Epoch(val) [50][404/404] mIoU: 88.0 mDice: 90.0\n')
        self.assertEqual(watcher.inspect_run(self.output, True)['status'],
                         'running')
        self.assertEqual(watcher.inspect_run(self.output, False)['status'],
                         'completed')

    def test_early_exit_reports_failure_and_progress(self):
        result = watcher.inspect_run(self.output, False)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual((result['epoch'], result['iteration_in_epoch'],
                          result['iterations_per_epoch']), (2, 20, 939))


if __name__ == '__main__':
    unittest.main()
