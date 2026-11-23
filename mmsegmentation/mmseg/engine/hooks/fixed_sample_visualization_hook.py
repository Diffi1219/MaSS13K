"""Render the same labeled samples periodically without changing training state."""
import copy
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from mmengine.dataset import Compose, pseudo_collate
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmseg.datasets.solar import LoadSolarAnnotations, LoadSolarImage, read_rgb
from mmseg.datasets.transforms import PackSegInputs
from mmseg.registry import HOOKS


def render_fixed_comparison(image, ground_truth, prediction, alpha=0.5):
    """Original | binary GT | binary prediction | foreground-only overlay."""
    height, width = image.shape[:2]
    if ground_truth.shape != (height, width) or prediction.shape != (height, width):
        raise ValueError('Fixed preview image and masks must have matching dimensions')
    if not np.isin(ground_truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError('Fixed preview masks must use binary classes 0/1')
    gt_image = np.repeat((ground_truth.astype(np.uint8)*255)[..., None], 3, axis=2)
    pred_image = np.repeat((prediction.astype(np.uint8)*255)[..., None], 3, axis=2)
    overlay = image.copy()
    foreground = prediction == 1
    overlay[foreground] = (
        image[foreground]*(1-alpha) + np.array([255, 80, 0])*alpha).astype(np.uint8)
    panels = []
    for title, panel in zip(('Original', 'GT (white = PV)', 'Prediction (white = PV)',
                             'Overlay (prediction)'),
                            (image, gt_image, pred_image, overlay)):
        header = np.full((32, width, 3), 24, dtype=np.uint8)
        cv2.putText(header, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.concatenate((header, panel), axis=0))
    return np.concatenate(panels, axis=1)


@HOOKS.register_module()
class FixedSampleVisualizationHook(Hook):
    """Use native val_step and visualizer backends for four-panel comparisons.

    ``samples`` contains img_path / seg_map_path pairs relative to data_root.
    Runs only on rank zero, after training updates and checkpoint hooks.
    """

    priority = 'LOWEST'

    def __init__(self, data_root, samples, interval=1000, draw_last=True, alpha=0.5):
        if interval < 1:
            raise ValueError('Fixed-sample interval must be positive')
        if not samples:
            raise ValueError('At least one fixed image/label pair is required')
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be between 0 and 1')
        self.data_root = Path(data_root)
        self.samples = copy.deepcopy(samples)
        self.interval = interval
        self.draw_last = draw_last
        self.alpha = alpha
        self._resolved_samples = []
        self._pipeline = None
        self._last_step = None

    @master_only
    def before_train(self, runner):
        self._resolved_samples = []
        for sample in self.samples:
            image = (self.data_root / sample['img_path']).resolve()
            label = (self.data_root / sample['seg_map_path']).resolve()
            for path in (image, label):
                if not path.is_file():
                    raise FileNotFoundError(f'Fixed visualization file does not exist: {path}')
            if any(image == previous[0] for previous in self._resolved_samples):
                raise ValueError(f'Duplicate fixed image: {image}')
            self._resolved_samples.append((image, label))
            runner.logger.info(f'Fixed visualization sample: image={image}; label={label}')
        # No random transforms: every snapshot uses the original image geometry.
        self._pipeline = Compose([LoadSolarImage(), LoadSolarAnnotations(), PackSegInputs()])

    @master_only
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        step = runner.iter + 1
        if step == self._last_step:
            return
        if step % self.interval != 0 and not (self.draw_last and step == runner.max_iters):
            return
        if self._pipeline is None:
            raise RuntimeError('Fixed visualization hook must run before_train first')

        # Avoid distributed-wrapper collectives when only rank zero predicts.
        model = runner.model
        while is_model_wrapper(model):
            model = model.module
        training_flags = [(module, module.training) for module in model.modules()]
        devices = sorted({parameter.device.index for parameter in model.parameters()
                          if parameter.is_cuda})
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        visualizer = runner.visualizer
        try:
            model.eval()
            # Preserve CPU/CUDA RNG as well as model/BN/dropout training flags.
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                for index, (image_path, label_path) in enumerate(self._resolved_samples, 1):
                    packed = self._pipeline(dict(
                        img_path=str(image_path), seg_map_path=str(label_path),
                        label_map=None, reduce_zero_label=False, seg_fields=[]))
                    ground_truth = packed['data_samples'].gt_sem_seg.data[0].cpu().numpy().copy()
                    prediction = model.val_step(pseudo_collate([packed]))[0]
                    pred_mask = prediction.pred_sem_seg.data[0].detach().cpu().numpy()
                    comparison = render_fixed_comparison(
                        read_rgb(image_path), ground_truth, pred_mask, self.alpha)
                    visualizer.add_image(
                        f'fixed_val_{index:02d}_{image_path.stem}',
                        comparison, step=step)
        finally:
            for module, training in training_flags:
                module.training = training
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        self._last_step = step
        runner.logger.info(
            f'Fixed validation previews saved at train iteration {step}: '
            f'{len(self._resolved_samples)} samples: Original | GT | Prediction | Overlay.')
