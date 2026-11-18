"""Compare the same unaugmented training images throughout training."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.dataset import pseudo_collate
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import EpochBasedTrainLoop

from mmseg.registry import DATASETS, HOOKS


@HOOKS.register_module()
class FixedTrainVisualizationHook(Hook):
    """Write fixed training predictions at epochs 0, interval, 2*interval, ...

    ``dataset`` must use the training image/label paths with a deterministic
    inference pipeline (no random augmentation). ``sample_names`` contains
    exact image filenames. If omitted, use the first ``num_samples`` filenames
    in sorted order. Image steps are completed epochs, scalar training steps
    from the standard LoggerHook remain iterations.
    """

    def __init__(self, dataset, interval=5, num_samples=4, sample_names=None,
                 log_initial=True):
        if interval < 1 or num_samples < 1:
            raise ValueError('interval and num_samples must be positive')
        self.dataset_cfg = dataset
        self.interval = interval
        self.num_samples = num_samples
        self.sample_names = sample_names
        self.log_initial = log_initial
        self._dataset = None
        self._indices = None

    def _prepare_samples(self):
        if self._dataset is not None:
            return
        dataset = DATASETS.build(self.dataset_cfg)
        by_name = {}
        for index in range(len(dataset)):
            name = Path(dataset.get_data_info(index)['img_path']).name
            if name in by_name:
                raise ValueError(f'Ambiguous fixed sample filename: {name}')
            by_name[name] = index
        names = (list(self.sample_names) if self.sample_names is not None
                 else sorted(by_name)[:self.num_samples])
        if not names or len(names) != len(set(names)):
            raise ValueError('Fixed sample filenames must be nonempty and unique')
        missing = set(names) - set(by_name)
        if missing:
            raise ValueError(f'Fixed training samples not found: {sorted(missing)}')
        self._indices = [by_name[name] for name in names]
        self._dataset = dataset

    @master_only
    def before_train(self, runner):
        if not isinstance(runner.train_loop, EpochBasedTrainLoop):
            raise ValueError('FixedTrainVisualizationHook requires epoch training')
        self._prepare_samples()
        if self.log_initial and runner.epoch == 0:
            self.visualize(runner.model, runner.visualizer, runner.work_dir, 0)

    @master_only
    def after_train_epoch(self, runner):
        # MMEngine increments runner.epoch AFTER after_train_epoch hooks.
        epoch = runner.epoch + 1
        if epoch % self.interval == 0:
            self.visualize(runner.model, runner.visualizer, runner.work_dir,
                           epoch)

    @master_only
    def visualize(self, model, visualizer, work_dir, epoch):
        """Also usable offline to preview a checkpoint at a specified epoch."""
        self._prepare_samples()
        if is_model_wrapper(model):
            model = model.module
        was_training = model.training
        output_dir = Path(work_dir) / 'fixed_train_samples'
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = [self._dataset.get_data_info(i) for i in self._indices]
        (output_dir / 'samples.json').write_text(json.dumps(
            dict(columns=['image', 'ground_truth', 'prediction', 'overlay'],
                 samples=[dict(image=d['img_path'], label=d['seg_map_path'])
                          for d in manifest]), indent=2), encoding='utf-8')
        palette = np.asarray(self._dataset.metainfo['palette'], dtype=np.uint8)
        try:
            model.eval()
            with torch.inference_mode():
                for index, info in zip(self._indices, manifest):
                    data = pseudo_collate([self._dataset[index]])
                    result = model.test_step(data)[0]
                    pred = result.pred_sem_seg.data[0].cpu().numpy()
                    gt = result.gt_sem_seg.data[0].cpu().numpy()
                    image = cv2.cvtColor(cv2.imread(info['img_path']),
                                         cv2.COLOR_BGR2RGB)
                    gt_color = self._colorize(gt, palette)
                    pred_color = self._colorize(pred, palette)
                    overlay = image.copy()
                    foreground = (pred > 0) & (pred < len(palette))
                    overlay[foreground] = (
                        0.5 * image[foreground] +
                        0.5 * pred_color[foreground]).astype(np.uint8)
                    panels = []
                    for title, panel in zip(
                            ['Image', 'Ground truth', 'Prediction', 'Overlay'],
                            [image, gt_color, pred_color, overlay]):
                        panel = cv2.copyMakeBorder(panel, 30, 0, 0, 0,
                                                   cv2.BORDER_CONSTANT)
                        cv2.putText(panel, title, (8, 21),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                        panels.append(panel)
                    comparison = np.concatenate(panels, axis=1)
                    # Flat names also work with MMEngine's LocalVisBackend.
                    tag = f'fixed_train__{Path(info["img_path"]).stem}'
                    visualizer.add_image(tag, comparison, step=epoch)
        finally:
            model.train(was_training)
        backend = visualizer.get_backend('TensorboardVisBackend')
        if backend is not None:
            backend.experiment.flush()

    @staticmethod
    def _colorize(mask, palette):
        color = np.full((*mask.shape, 3), 128, dtype=np.uint8)
        valid = (mask >= 0) & (mask < len(palette))
        color[valid] = palette[mask[valid].astype(np.int64)]
        return color
