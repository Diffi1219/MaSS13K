"""Check metric calculation and TensorBoard-only comparison image storage."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import uuid

import cv2
import numpy as np
import torch
from mmengine.structures import PixelData
from mmengine.visualization import Visualizer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mmseg.engine.hooks import TestTensorboardHook
from mmseg.structures import SegDataSample


class TestTestTensorboardHook(unittest.TestCase):

    def test_metrics_and_event_images_without_display_png(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            image_path = output / 'sample.tif'
            cv2.imwrite(str(image_path), np.full((2, 2, 3), 80, dtype=np.uint8))
            visualizer = Visualizer(
                name='test_hook_' + uuid.uuid4().hex, save_dir=temporary,
                vis_backends=[dict(type='TensorboardVisBackend')])
            dataset = SimpleNamespace(
                metainfo=dict(classes=['bg', 'pv'], palette=[[0, 0, 0],
                                                           [255, 255, 255]]),
                data_root='test-data', data_prefix=dict(img_path='image'))
            runner = SimpleNamespace(
                test_dataloader=SimpleNamespace(dataset=dataset),
                visualizer=visualizer, work_dir=temporary,
                cfg=SimpleNamespace(load_from='epoch_50.pth'), logger=Mock())
            sample = SegDataSample(metainfo=dict(
                img_path=str(image_path), seg_map_path='sample-label.tif'))
            sample.pred_sem_seg = PixelData(data=torch.tensor([[[0, 1], [0, 1]]]))
            sample.gt_sem_seg = PixelData(data=torch.tensor([[[0, 1], [1, 1]]]))
            hook = TestTensorboardHook()
            hook.before_test(runner)
            hook.after_test_iter(runner, 0, outputs=[sample])
            hook.after_test_epoch(runner, metrics={'mIoU': 58.33, 'aAcc': 75.0})
            report = json.loads((output / 'metrics.json').read_text())
            self.assertEqual(report['confusion_matrix'], [[1, 0], [1, 2]])
            self.assertEqual(report['per_class']['pv']['IoU'], 66.67)
            self.assertEqual(report['per_class']['pv']['Dice'], 80.0)
            self.assertEqual(report['images'], 1)
            self.assertEqual(list(output.rglob('*.png')), [])
            event = next(output.rglob('events.out.tfevents.*'))
            events = EventAccumulator(str(event.parent)).Reload()
            images = events.Images('test_images/0001__sample')
            self.assertEqual(len(images), 1)
            self.assertEqual((images[0].step, images[0].height, images[0].width),
                             (50, 32, 8))
            self.assertEqual(events.Scalars('test/mIoU')[0].step, 50)
            visualizer.get_backend('TensorboardVisBackend').close()


if __name__ == '__main__':
    unittest.main()
