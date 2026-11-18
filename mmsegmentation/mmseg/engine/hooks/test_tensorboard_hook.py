"""Log binary-segmentation test results without a second set of display PNGs."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from mmengine.dist import get_world_size
from mmengine.hooks import Hook

from mmseg.registry import HOOKS


def class_metrics(confusion):
    """Return dataset-level metrics, with undefined ratios represented as zero."""
    tp = np.diag(confusion).astype(np.float64)
    actual = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)

    def ratio(numerator, denominator):
        return np.divide(numerator, denominator,
                         out=np.zeros_like(tp), where=denominator != 0) * 100

    return dict(IoU=ratio(tp, actual + predicted - tp),
                Dice=ratio(2 * tp, actual + predicted),
                Precision=ratio(tp, predicted), Recall=ratio(tp, actual),
                Fscore=ratio(2 * tp, actual + predicted))


@HOOKS.register_module()
class TestTensorboardHook(Hook):
    """Write every test image to TensorBoard, plus JSON/CSV metrics.

    This hook is for a single-process test run. The visualizer should contain
    only TensorboardVisBackend, so comparison images live only in event files.
    The existing IoUMetric can separately export raw 0/1 prediction masks.
    """

    def __init__(self, checkpoint_epoch=50):
        self.checkpoint_epoch = checkpoint_epoch

    def before_test(self, runner):
        if get_world_size() != 1:
            raise ValueError('TestTensorboardHook requires single-process testing')
        self.meta = runner.test_dataloader.dataset.metainfo
        self.classes = list(self.meta['classes'])
        self.palette = np.asarray(self.meta['palette'], dtype=np.uint8)
        self.confusion = np.zeros((len(self.classes), len(self.classes)),
                                  dtype=np.int64)
        self.rows = []

    @staticmethod
    def comparison(image, gt, pred, palette):
        def colorize(mask):
            color = np.full((*mask.shape, 3), 128, dtype=np.uint8)
            valid = (mask >= 0) & (mask < len(palette))
            color[valid] = palette[mask[valid].astype(np.int64)]
            return color

        gt_color, pred_color = colorize(gt), colorize(pred)
        overlay = image.copy()
        foreground = (pred > 0) & (pred < len(palette))
        overlay[foreground] = (
            image[foreground] * 0.5 + pred_color[foreground] * 0.5).astype(np.uint8)
        panels = []
        for panel, title in zip([image, gt_color, pred_color, overlay],
                                ['Image', 'Ground truth', 'Prediction', 'Overlay']):
            panel = cv2.copyMakeBorder(panel, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(panel, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        return np.concatenate(panels, axis=1)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        count = len(self.classes)
        for sample in outputs:
            pred = sample.pred_sem_seg.data[0].detach().cpu().numpy()
            gt = sample.gt_sem_seg.data[0].detach().cpu().numpy()
            assert pred.shape == gt.shape, (pred.shape, gt.shape)
            valid = (gt >= 0) & (gt < count)
            if not np.all((pred[valid] >= 0) & (pred[valid] < count)):
                raise ValueError('Prediction contains an unknown class index')
            confusion = np.bincount(
                count * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
                minlength=count * count).reshape(count, count)
            self.confusion += confusion
            values = class_metrics(confusion)
            index = len(self.rows) + 1
            name = Path(sample.img_path).stem
            tag = f'test_images/{index:04d}__{name}'
            row = dict(sample_index=index, filename=Path(sample.img_path).name,
                       image_path=sample.img_path,
                       label_path=sample.get('seg_map_path', ''), tensorboard_tag=tag)
            for metric in ['IoU', 'Dice', 'Precision', 'Recall']:
                row[f'pv_{metric}'] = round(float(values[metric][1]), 4)
                runner.visualizer.add_scalar(
                    f'test_samples/pv_{metric}', row[f'pv_{metric}'], step=index)
            self.rows.append(row)
            image = cv2.imread(sample.img_path)
            if image is None:
                raise FileNotFoundError(sample.img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            runner.visualizer.add_image(
                tag, self.comparison(image, gt, pred, self.palette),
                step=self.checkpoint_epoch)

    def after_test_epoch(self, runner, metrics=None):
        values = class_metrics(self.confusion)
        per_class = {
            name: {key: round(float(value[i]), 2)
                   for key, value in values.items()}
            for i, name in enumerate(self.classes)
        }
        overall = {key: float(value) for key, value in (metrics or {}).items()}
        for key, value in overall.items():
            runner.visualizer.add_scalar(f'test/{key}', value,
                                         step=self.checkpoint_epoch)
        for name, results in per_class.items():
            for key, value in results.items():
                runner.visualizer.add_scalar(f'test_class/{name}/{key}', value,
                                             step=self.checkpoint_epoch)
        output = Path(runner.work_dir)
        report = dict(checkpoint=runner.cfg.load_from, epoch=self.checkpoint_epoch,
                      dataset=runner.test_dataloader.dataset.data_root,
                      data_prefix=runner.test_dataloader.dataset.data_prefix,
                      images=len(self.rows), metrics_unit='percent', overall=overall,
                      per_class=per_class, confusion_matrix=self.confusion.tolist(),
                      confusion_matrix_axes=['ground_truth', 'prediction'],
                      display_image_storage='TensorBoard event files only',
                      tensorboard_tags='test_images/<sample index>__<filename>')
        (output / 'metrics.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (output / 'image_manifest.json').write_text(
            json.dumps(self.rows, ensure_ascii=False, indent=2), encoding='utf-8')
        with (output / 'per_image_metrics.csv').open('w', newline='',
                                                    encoding='utf-8') as stream:
            if self.rows:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        backend = runner.visualizer.get_backend('TensorboardVisBackend')
        if backend is not None:
            backend.experiment.flush()
        runner.logger.info(f'Test report and per-image metrics saved to {output}')
