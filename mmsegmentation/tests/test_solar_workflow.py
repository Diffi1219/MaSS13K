"""Regression tests for configuration priority and solar segmentation behavior."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('solar_cli', ROOT / 'tools/solar.py')
solar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solar)
train_spec = importlib.util.spec_from_file_location('native_train', ROOT / 'tools/train.py')
native_train = importlib.util.module_from_spec(train_spec)
train_spec.loader.exec_module(native_train)
test_spec = importlib.util.spec_from_file_location('native_test', ROOT / 'tools/test.py')
native_test = importlib.util.module_from_spec(test_spec)
test_spec.loader.exec_module(native_test)


class SolarWorkflowTests(unittest.TestCase):
    def _fixed_preview_fixture(self, directory):
        from types import SimpleNamespace
        from mmengine.structures import PixelData
        root = Path(directory)
        (root / 'image').mkdir()
        (root / 'label').mkdir()
        pairs = []
        for index in range(4):
            Image.new('RGB', (8, 8), (30, 60, 90)).save(root / f'image/{index}.tif')
            Image.fromarray(np.eye(8, dtype=np.uint8)).save(root / f'label/{index}.tif')
            pairs.append(dict(img_path=f'image/{index}.tif', seg_map_path=f'label/{index}.tif'))

        class PreviewModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.bn = torch.nn.BatchNorm2d(1).eval()
                self.dropout = torch.nn.Dropout()

            def val_step(self, data):
                import random
                assert not self.training and not torch.is_grad_enabled()
                # Deliberately consume random values: the hook must restore them.
                torch.rand(1)
                np.random.random()
                random.random()
                sample = data['data_samples'][0].clone()
                sample.pred_sem_seg = PixelData(data=torch.ones_like(sample.gt_sem_seg.data))
                return [sample]

        meta = dict(classes=('background', 'solar_panel'), palette=[[0, 0, 0], [255, 80, 0]])
        runner = SimpleNamespace(
            model=PreviewModel(), iter=0, max_iters=2, logger=Mock(),
            visualizer=SimpleNamespace(dataset_meta=dict(original=True), alpha=0.8,
                                       add_image=Mock()),
            train_dataloader=SimpleNamespace(dataset=SimpleNamespace(metainfo=meta)))
        return runner, pairs

    def test_fixed_previews_keep_samples_steps_and_training_state(self):
        import random
        from mmseg.engine.hooks.fixed_sample_visualization_hook import FixedSampleVisualizationHook
        with tempfile.TemporaryDirectory() as tmp:
            runner, pairs = self._fixed_preview_fixture(tmp)
            hook = FixedSampleVisualizationHook(tmp, pairs, interval=1)
            hook.before_train(runner)
            flags = [module.training for module in runner.model.modules()]
            torch_state = torch.get_rng_state().clone()
            np_state = np.random.get_state()
            python_state = random.getstate()
            hook.after_train_iter(runner, 0)
            hook.after_train_iter(runner, 0)  # Same callback must not duplicate images.
            runner.iter = 1
            hook.after_train_iter(runner, 1)
            calls = runner.visualizer.add_image.call_args_list
            self.assertEqual(len(calls), 8)
            self.assertEqual([call.args[0] for call in calls[:4]],
                             [call.args[0] for call in calls[4:]])
            self.assertEqual([call.kwargs['step'] for call in calls], [1]*4 + [2]*4)
            self.assertEqual([module.training for module in runner.model.modules()], flags)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
            self.assertEqual(np.random.get_state()[2:], np_state[2:])
            self.assertEqual(random.getstate(), python_state)
            self.assertEqual(runner.visualizer.dataset_meta, dict(original=True))
            self.assertEqual(runner.visualizer.alpha, 0.8)

    def test_fixed_preview_four_panels_and_foreground_only_overlay(self):
        from mmseg.engine.hooks.fixed_sample_visualization_hook import render_fixed_comparison
        image = np.full((6, 8, 3), [20, 40, 60], dtype=np.uint8)
        gt = np.zeros((6, 8), dtype=np.uint8)
        pred = np.zeros_like(gt)
        gt[1, 1] = 1
        pred[2, 3] = 1
        result = render_fixed_comparison(image, gt, pred, alpha=0.5)
        self.assertEqual(result.shape, (38, 32, 3))
        body = result[32:]
        np.testing.assert_array_equal(body[:, :8], image)
        np.testing.assert_array_equal(body[:, 8:16], np.repeat((gt*255)[..., None], 3, axis=2))
        np.testing.assert_array_equal(body[:, 16:24], np.repeat((pred*255)[..., None], 3, axis=2))
        overlay = body[:, 24:]
        np.testing.assert_array_equal(overlay[pred == 0], image[pred == 0])
        np.testing.assert_array_equal(overlay[2, 3], [137, 60, 30])
        np.testing.assert_array_equal(image[2, 3], [20, 40, 60])

    def test_fixed_preview_interval_and_error_restore_state(self):
        from mmseg.engine.hooks.fixed_sample_visualization_hook import FixedSampleVisualizationHook
        with tempfile.TemporaryDirectory() as tmp:
            runner, pairs = self._fixed_preview_fixture(tmp)
            hook = FixedSampleVisualizationHook(tmp, pairs, interval=2, draw_last=False)
            hook.before_train(runner)
            with patch.object(runner.model, 'val_step', side_effect=RuntimeError('probe')) as predict:
                hook.after_train_iter(runner, 0)
                predict.assert_not_called()
                runner.iter = 1
                with self.assertRaisesRegex(RuntimeError, 'probe'):
                    hook.after_train_iter(runner, 1)
            self.assertTrue(runner.model.training)
            self.assertFalse(runner.model.bn.training)
            self.assertTrue(runner.model.dropout.training)
            self.assertEqual(runner.visualizer.dataset_meta, dict(original=True))
            self.assertEqual(runner.visualizer.alpha, 0.8)

    def _training_args(self, config, **overrides):
        from types import SimpleNamespace
        values = dict(config=str(config), cfg_options=None, launcher=None,
                      work_dir=None, amp=None, resume=None)
        values.update(overrides)
        return SimpleNamespace(**values)

    def _training_config_file(self, directory):
        path = Path(directory) / 'experiment.py'
        path.write_text(
            f'_base_ = [{solar.SOLAR_CONFIG.as_posix()!r}]\n'
            f'work_dir = {str(Path(directory) / "experiment")!r}\n'
            "device = 'cuda:7'\n"
            "pretrained_checkpoint = 'custom_source.pth'\n"
            "load_from = 'custom_ready.pth'\n"
            "test_checkpoint = None\n"
            "resume = True\n"
            "train_cfg = dict(max_iters=37, val_interval=7)\n"
            "train_dataloader = dict(batch_size=3, dataset=dict(\n"
            "    data_root='custom_train', pipeline=[dict(type='RandomCrop', crop_size=(256, 512))]))\n"
            "val_dataloader = dict(dataset=dict(data_root='custom_val'))\n"
            "test_dataloader = dict(dataset=dict(data_root='custom_test'))\n"
            "optim_wrapper = dict(type='AmpOptimWrapper', loss_scale=128, dtype='float16')\n"
            "param_scheduler = [dict(type='PolyLR', begin=0, end=31, by_epoch=False)]\n"
            "default_hooks = dict(checkpoint=dict(interval=11))\n", encoding='utf-8')
        return path

    def test_training_keeps_config_defaults_through_command_line(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            path = self._training_config_file(tmp)
            # Exercise the actual parser and training entry without CUDA or a real run.
            with patch.object(sys, 'argv', ['train.py', str(path)]), \
                    patch.object(native_train, 'require_cuda') as cuda, \
                    patch.object(native_train, 'prepare_checkpoint') as prepare, \
                    patch.object(native_train.Runner, 'from_cfg') as runner:
                native_train.main()
            cfg = runner.call_args.args[0]
            cuda.assert_called_once_with('cuda:7')
            prepare.assert_not_called()
            runner.return_value.train.assert_called_once_with()
            self.assertEqual(cfg.train_cfg.max_iters, 37)
            self.assertEqual(cfg.train_cfg.val_interval, 7)
            self.assertEqual(cfg.train_dataloader.batch_size, 3)
            self.assertEqual(cfg.train_dataloader.dataset.data_root, 'custom_train')
            self.assertEqual(cfg.val_dataloader.dataset.data_root, 'custom_val')
            self.assertEqual(cfg.test_dataloader.dataset.data_root, 'custom_test')
            self.assertEqual(cfg.train_dataloader.dataset.pipeline[0].crop_size, (256, 512))
            self.assertEqual(cfg.pretrained_checkpoint, 'custom_source.pth')
            self.assertEqual(cfg.load_from, 'custom_ready.pth')
            self.assertTrue(cfg.resume)
            self.assertEqual(cfg.optim_wrapper.type, 'AmpOptimWrapper')
            self.assertEqual(cfg.optim_wrapper.loss_scale, 128)
            self.assertEqual(cfg.param_scheduler[0].end, 31)
            self.assertEqual(cfg.default_hooks.checkpoint.interval, 11)
            self.assertEqual(Path(cfg.work_dir), Path(tmp) / 'experiment')

    def test_training_explicit_overrides_preserve_independent_settings(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            path = self._training_config_file(tmp)
            with patch.object(sys, 'argv', [
                    'train.py', str(path), '--work-dir', str(Path(tmp) / 'override'),
                    '--no-amp', '--no-resume', '--launcher', 'none', '--cfg-options',
                    'train_cfg.max_iters=60', 'train_cfg.val_interval=5',
                    'train_dataloader.batch_size=2', 'device=cuda:0']):
                cfg = native_train.load_config(native_train.parse_args())
            self.assertEqual(cfg.train_cfg.max_iters, 60)
            self.assertEqual(cfg.train_cfg.val_interval, 5)
            self.assertEqual(cfg.train_dataloader.batch_size, 2)
            self.assertEqual(cfg.device, 'cuda:0')
            self.assertEqual(Path(cfg.work_dir), Path(tmp) / 'override')
            self.assertEqual(cfg.train_dataloader.dataset.data_root, 'custom_train')
            self.assertEqual(cfg.train_dataloader.dataset.pipeline[0].crop_size, (256, 512))
            self.assertEqual(cfg.load_from, 'custom_ready.pth')
            self.assertEqual(cfg.pretrained_checkpoint, 'custom_source.pth')
            self.assertEqual(cfg.launcher, 'none')
            self.assertFalse(cfg.resume)
            self.assertEqual(cfg.optim_wrapper.type, 'OptimWrapper')
            self.assertNotIn('loss_scale', cfg.optim_wrapper)
            self.assertNotIn('dtype', cfg.optim_wrapper)
            self.assertEqual(cfg.param_scheduler[0].end, 31)
            self.assertEqual(cfg.default_hooks.checkpoint.interval, 11)
            # Resolving an experiment never modifies its source config.
            self.assertEqual(native_train.load_config(self._training_args(path)).train_cfg.max_iters, 37)

    def test_native_training_transfer_and_resume_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._training_config_file(tmp)
            cfg = native_train.load_config(self._training_args(path))
            with patch.object(native_train, 'prepare_checkpoint') as prepare:
                native_train.prepare_training_checkpoint(cfg)  # Resume keeps native loading.
                cfg.resume = False
                native_train.prepare_training_checkpoint(cfg)  # Explicit load_from wins.
                prepare.assert_not_called()
            cfg.load_from = None
            source = Path(tmp) / 'source.pth'
            torch.save(dict(state_dict={
                'decode_head.cls_embed.weight': torch.ones(8, 256),
                'decode_head.cls_embed.bias': torch.zeros(8),
                'backbone.example': torch.ones(1)}), source)
            source_bytes = source.read_bytes()
            cfg.pretrained_checkpoint = str(source)
            native_train.prepare_training_checkpoint(cfg)
            self.assertEqual(Path(cfg.load_from), Path(cfg.work_dir) / 'init_from_mass13k.pth')
            self.assertTrue(torch.load(cfg.load_from, weights_only=False)['meta']['transfer_untrained'])
            self.assertEqual(source.read_bytes(), source_bytes)
            cfg.load_from = None
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                native_train.prepare_training_checkpoint(cfg)
            cfg.launcher = 'pytorch'
            with self.assertRaisesRegex(ValueError, 'prepare the checkpoint once'):
                native_train.prepare_training_checkpoint(cfg)

    def test_native_custom_runner_and_distributed_launcher_are_preserved(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            path = self._training_config_file(tmp)
            with patch.object(sys, 'argv', [
                    'train.py', str(path), '--cfg-options',
                    'launcher=pytorch', 'runner_type=ExampleRunner']), \
                    patch.object(native_train, 'require_cuda') as cuda, \
                    patch.object(native_train.RUNNERS, 'build') as runner:
                native_train.main()
            cfg = runner.call_args.args[0]
            self.assertEqual(cfg.launcher, 'pytorch')
            self.assertTrue(cfg.resume)
            cuda.assert_not_called()  # Distributed launcher uses LOCAL_RANK.
            runner.return_value.train.assert_called_once_with()

    def test_label_encodings_and_invalid_values(self):
        from mmseg.datasets.solar import read_mask
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'label.tif'
            for value in (1, 255):
                Image.fromarray(np.array([[0, value]], dtype=np.uint8)).save(path)
                np.testing.assert_array_equal(read_mask(path), [[0, 1]])
            Image.fromarray(np.array([[0, 2]], dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, 'unsupported labels'):
                read_mask(path)

    def test_native_evaluation_uses_selected_checkpoint_and_test_loop(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            config = self._training_config_file(tmp)
            checkpoint = Path(tmp) / 'trained.pth'
            torch.save(dict(state_dict={'decode_head.cls_embed.weight': torch.ones(3, 256)},
                            meta=dict(dataset_meta=dict(classes=solar.SOLAR_CLASSES))), checkpoint)
            with patch.object(sys, 'argv', [
                    'test.py', str(config), str(checkpoint), '--work-dir', tmp,
                    '--cfg-options', 'device=cuda:0']), \
                    patch.object(native_test, 'require_cuda') as cuda, \
                    patch.object(native_test.Runner, 'from_cfg') as runner:
                native_test.main()
            cfg = runner.call_args.args[0]
            cuda.assert_called_once_with('cuda:0')
            self.assertEqual(cfg.load_from, str(checkpoint))
            self.assertFalse(cfg.resume)  # Config resume=True must not select another weight.
            self.assertEqual(cfg.test_dataloader.dataset.data_root, 'custom_test')
            self.assertEqual(cfg.train_cfg.max_iters, 37)
            self.assertEqual(cfg.launcher, 'none')
            runner.return_value.test.assert_called_once_with()
            runner.return_value.train.assert_not_called()

    def test_native_evaluation_rejects_original_seven_class_weights(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            config = self._training_config_file(tmp)
            checkpoint = Path(tmp) / 'original.pth'
            torch.save(dict(state_dict={'decode_head.cls_embed.weight': torch.ones(8, 256)}), checkpoint)
            with patch.object(sys, 'argv', ['test.py', str(config), str(checkpoint)]), \
                    patch.object(native_test, 'require_cuda') as cuda, \
                    patch.object(native_test.Runner, 'from_cfg') as runner:
                with self.assertRaisesRegex(ValueError, 'TRAINED 2-class'):
                    native_test.main()
            cuda.assert_not_called()
            runner.assert_not_called()

    def test_native_evaluation_config_only_command(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            config = self._training_config_file(tmp)
            checkpoint = Path(tmp) / 'selected.pth'
            output = Path(tmp) / 'evaluation'
            torch.save(dict(state_dict={'decode_head.cls_embed.weight': torch.ones(3, 256)},
                            meta=dict(dataset_meta=dict(classes=solar.SOLAR_CLASSES))), checkpoint)
            with config.open('a', encoding='utf-8') as file:
                file.write(f'test_checkpoint = {str(checkpoint)!r}\n'
                           f'test_work_dir = {str(output)!r}\n')
            with patch.object(sys, 'argv', ['test.py', str(config)]), \
                    patch.object(native_test, 'require_cuda') as cuda, \
                    patch.object(native_test.Runner, 'from_cfg') as runner:
                native_test.main()
            cfg = runner.call_args.args[0]
            self.assertEqual(cfg.load_from, str(checkpoint))
            self.assertEqual(Path(cfg.work_dir), output)
            cuda.assert_called_once_with('cuda:7')
            runner.return_value.test.assert_called_once_with()

    def test_native_evaluation_explicit_checkpoint_and_work_dir_win(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            config = self._training_config_file(tmp)
            with config.open('a', encoding='utf-8') as file:
                file.write("test_checkpoint = 'configured.pth'\n"
                           "test_work_dir = 'configured_output'\n")
            with patch.object(sys, 'argv', [
                    'test.py', str(config), 'explicit.pth', '--work-dir', tmp]):
                cfg = native_test.load_config(native_test.parse_args())
            self.assertEqual(cfg.load_from, 'explicit.pth')
            self.assertEqual(Path(cfg.work_dir), Path(tmp))

    def test_native_evaluation_missing_checkpoint_is_rejected(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            config = self._training_config_file(tmp)
            with config.open('a', encoding='utf-8') as file:
                file.write('load_from = None\n')
            with patch.object(sys, 'argv', ['test.py', str(config)]):
                with self.assertRaisesRegex(ValueError, 'Set test_checkpoint'):
                    native_test.load_config(native_test.parse_args())

    def test_pairing_detects_duplicates_and_missing_label(self):
        from mmseg.datasets.solar import SolarGeneralDataset, index_images
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'image').mkdir()
            (root / 'label').mkdir()
            rgb = Image.new('RGB', (8, 8))
            rgb.save(root / 'image/a.tif')
            rgb.save(root / 'image/a.png')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                index_images(root / 'image')
            (root / 'image/a.png').unlink()
            Image.new('L', (8, 8)).save(root / 'label/b.tif')
            with self.assertRaisesRegex(ValueError, 'Unpaired'):
                SolarGeneralDataset(data_root=tmp,
                    data_prefix=dict(img_path='image', seg_map_path='label'), pipeline=[])

    def test_tiling_covers_small_and_irregular_images(self):
        from types import SimpleNamespace
        def prediction(model, image):
            h, w = image.shape[:2]
            scores = torch.zeros((2, h, w))
            scores[1] = torch.from_numpy(image[:, :, 0].copy()).float()
            return SimpleNamespace(seg_logits=SimpleNamespace(data=scores))
        for height, width in [(33, 17), (513, 777)]:
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[::2, ::3, 0] = 1
            with patch('mmseg.apis.inference_model', side_effect=prediction):
                mask = solar.predict_tiled(SimpleNamespace(num_classes=2), image, 256, 192)
            np.testing.assert_array_equal(mask, image[:, :, 0])

    def test_seven_class_and_transfer_weights_rejected_for_solar(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'original.pth'
            state = {'decode_head.cls_embed.weight': torch.ones(8, 256),
                     'decode_head.cls_embed.bias': torch.zeros(8),
                     'backbone.example': torch.ones(1)}
            torch.save(dict(state_dict=state), source)
            with self.assertRaisesRegex(ValueError, 'TRAINED 2-class'):
                solar.build_model(source, 'cuda:0')
            destination = Path(tmp) / 'init.pth'
            solar.prepare_checkpoint(source, destination)
            result = torch.load(destination, weights_only=False)
            self.assertTrue(result['meta']['transfer_untrained'])
            self.assertNotIn('decode_head.cls_embed.weight', result['state_dict'])
            self.assertIn('backbone.example', result['state_dict'])
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                solar.prepare_checkpoint(source, destination)
            # A two-class file without semantic metadata is also rejected.
            state['decode_head.cls_embed.weight'] = torch.ones(3, 256)
            state['decode_head.cls_embed.bias'] = torch.zeros(3)
            torch.save(dict(state_dict=state), source)
            with self.assertRaisesRegex(ValueError, 'semantics are unverified'):
                solar.build_model(source, 'cuda:0')

    def test_foreground_metrics_and_empty_denominators(self):
        metrics = solar.confusion_metrics(np.array([[8, 2], [1, 3]]))
        self.assertEqual(metrics['solar_IoU'], 0.5)
        self.assertEqual(metrics['solar_Dice_F1'], 2/3)
        self.assertIsNone(solar.confusion_metrics(np.zeros((2, 2)))['solar_IoU'])

    def test_head_padding_and_edge_loss(self):
        from mmseg.models.decode_heads.massformer_head import MaSSFormerHead
        from mmengine.structures import InstanceData
        # Exercise the actual methods without constructing a full GPU network.
        head = object.__new__(MaSSFormerHead)
        torch.nn.Module.__init__(head)
        head.norm_eval = True
        head.example_bn = torch.nn.BatchNorm2d(3)
        head.train()
        self.assertFalse(head.example_bn.training)
        self.assertEqual(tuple(head.example_bn(torch.zeros(1, 3, 1, 1)).shape),
                         (1, 3, 1, 1))
        head.forward = lambda *args: ([torch.tensor([[[5., 0., -5.]]])],
                                      [torch.ones(1, 1, 2, 2)], torch.zeros(1))
        result = head.predict([torch.zeros(1, 3, 4, 4)],
                              [dict(img_shape=(3, 5), pad_shape=(4, 8))], {})
        self.assertEqual(tuple(result.shape), (1, 2, 4, 8))
        head._loss_by_feat_single = lambda *args: (torch.tensor(0.),)*3
        logits = torch.tensor([[[[-2., 2., 0.]]]], requires_grad=True)
        target = torch.tensor([[[[0, 1, 255]]]])
        losses = head.loss_by_feat([torch.zeros(1)], [torch.zeros(1)],
            [InstanceData()], [{}], target, logits)
        losses['edge_loss'].backward()
        self.assertGreater(logits.grad[0, 0, 0, 0].item(), 0)
        self.assertLess(logits.grad[0, 0, 0, 1].item(), 0)
        self.assertEqual(logits.grad[0, 0, 0, 2].item(), 0)


if __name__ == '__main__':
    unittest.main()
