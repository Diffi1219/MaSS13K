"""Portable settings, split isolation and relocated checkpoint regression tests."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('pv_launcher', Path(__file__).parents[1] / 'tools/pv.py')
pv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pv)


class TestPVLauncher(unittest.TestCase):
    def test_default_splits_fp32_spawn_and_zero_workers(self):
        values = pv.settings(argparse.Namespace(settings=None, num_workers=0, epochs=8))
        cfg = pv.build_config(values)
        for split in ('train', 'val', 'test'):
            loader = cfg[f'{split}_dataloader']
            self.assertEqual(loader.dataset.data_prefix.img_path, f'{split}/image')
            self.assertTrue(Path(loader.dataset.data_root).is_absolute())
            self.assertEqual(loader.num_workers, 0)
            self.assertFalse(loader.persistent_workers)
            self.assertNotIn('prefetch_factor', loader)
        self.assertEqual(cfg.custom_hooks[0].dataset.data_prefix.img_path, 'train/image')
        self.assertTrue(cfg.train_dataloader.drop_last)
        self.assertEqual(cfg.env_cfg.mp_cfg.mp_start_method, 'spawn')
        self.assertEqual(cfg.optim_wrapper.type, 'OptimWrapper')
        self.assertEqual(cfg.param_scheduler[1].end, 8)
        self.assertEqual(cfg.param_scheduler[1].T_max, 7)
        self.assertEqual(cfg.train_cfg.dynamic_intervals, [(8, 1)])
        self.assertEqual(cfg.default_hooks.checkpoint.interval, 1)
        self.assertTrue(cfg.default_hooks.checkpoint.save_optimizer)

    def test_all_images_override(self):
        values = pv.settings(argparse.Namespace(settings=None, test_images=None))
        cfg = pv.build_config(values, testing=True)
        self.assertIsNone(cfg.custom_hooks[0].max_images)
        self.assertEqual(cfg.test_dataloader.dataset.data_prefix.img_path, 'test/image')
        self.assertFalse(cfg.resume)

    def test_relocated_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for saved in ['/old/linux/epoch_7.pth', r'D:\old\epoch_7.pth']:
                (root / 'last_checkpoint').write_text(saved)
                (root / 'epoch_7.pth').touch()
                self.assertEqual(pv.latest_checkpoint(root), (root / 'epoch_7.pth').resolve())
            (root / 'last_checkpoint').unlink()
            (root / 'epoch_12.pth').touch()
            self.assertEqual(pv.latest_checkpoint(root).name, 'epoch_12.pth')

    def test_batch_one_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'BatchNorm'):
            pv.settings(argparse.Namespace(settings=None, batch_size=1))

    def test_smoke_keeps_two_batches_when_batch_size_changes(self):
        from mmengine.config import Config

        class StopBeforeTraining(Exception):
            pass

        for batch_size in [2, 4, 8]:
            with self.subTest(batch_size=batch_size):
                cfg = pv.build_config(pv.settings(argparse.Namespace(
                    settings=None, batch_size=batch_size)))

                def inspect_command(command, log_path):
                    actual = Config.fromfile(command[2])
                    self.assertEqual(actual.train_dataloader.dataset.indices,
                                     2 * batch_size)
                    self.assertEqual(actual.custom_hooks[0].dataset.indices,
                                     2 * batch_size)
                    self.assertEqual(actual.val_dataloader.dataset.indices, 2)
                    raise StopBeforeTraining

                with patch.object(pv, 'launch', side_effect=inspect_command) as launch:
                    with self.assertRaises(StopBeforeTraining):
                        pv.smoke(cfg)
                    self.assertEqual(launch.call_count, 1)

    def test_data_format_and_overlap(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ['train', 'val', 'test']:
                for kind in ['image', 'label']:
                    (root / split / kind).mkdir(parents=True)
                cv2.imwrite(str(root / split / 'image' / f'{split}.tif'), np.zeros((512, 512, 3), np.uint8))
                cv2.imwrite(str(root / split / 'label' / f'{split}.tif'), np.ones((512, 512), np.uint8))
            report = pv.audit_data(root, full=True)
            self.assertEqual(report['train']['pairs'], 1)
            for kind in ['image', 'label']:
                (root / 'test' / kind / 'test.tif').rename(root / 'test' / kind / 'train.tif')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                pv.audit_data(root)


if __name__ == '__main__':
    unittest.main()
