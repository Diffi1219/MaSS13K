"""Local RGB imagery and binary photovoltaic masks for the General dataset."""
from pathlib import Path

import numpy as np
from PIL import Image
from mmcv.transforms import BaseTransform

from mmseg.registry import DATASETS, TRANSFORMS
from .basesegdataset import BaseSegDataset

IMAGE_SUFFIXES = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}


def index_images(directory):
    """Index by relative stem, refusing ambiguous image/label pairs."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Image/label directory does not exist: {directory}')
    indexed = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            key = path.relative_to(directory).with_suffix('').as_posix()
            if key in indexed:
                raise ValueError(f'Duplicate relative stem: {indexed[key]} and {path}')
            indexed[key] = path
    if not indexed:
        raise ValueError(f'No supported images found in {directory}')
    return indexed


def read_rgb(path):
    """Require 8-bit RGB; silently stretching multispectral data is unsafe."""
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError(f'{path}: expected uint8 RGB, got mode={image.mode}')
        return np.array(image)


def read_mask(path):
    """0=background; both 1 and 255 denote solar foreground in source files."""
    with Image.open(path) as image:
        mask = np.array(image)
    if mask.ndim != 2:
        raise ValueError(f'{path}: expected a single-channel class mask')
    values = np.unique(mask)
    if not np.isin(values, [0, 1, 255]).all():
        raise ValueError(f'{path}: unsupported labels {values.tolist()}; expected 0/1 or 0/255')
    return (mask != 0).astype(np.uint8)


@DATASETS.register_module()
class SolarGeneralDataset(BaseSegDataset):
    METAINFO = dict(classes=('background', 'solar_panel'),
                    palette=[[0, 0, 0], [255, 80, 0]])

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.tif', seg_map_suffix='.tif',
                         reduce_zero_label=False, **kwargs)

    def load_data_list(self):
        if self.ann_file:
            raise ValueError('SolarGeneralDataset expects image/label directories, not ann_file')
        images = index_images(self.data_prefix['img_path'])
        labels = index_images(self.data_prefix['seg_map_path'])
        missing = images.keys() - labels.keys()
        extra = labels.keys() - images.keys()
        if missing or extra:
            raise ValueError(f'Unpaired files: missing labels={sorted(missing)[:5]}, '
                             f'labels without images={sorted(extra)[:5]}')
        return [dict(img_path=str(path), seg_map_path=str(labels[key]),
                     label_map=None, reduce_zero_label=False, seg_fields=[])
                for key, path in images.items()]


@TRANSFORMS.register_module()
class LoadSolarImage(BaseTransform):
    def transform(self, results):
        # MMSeg's preprocessor converts BGR to RGB to match ImageNet statistics.
        results['img'] = read_rgb(results['img_path'])[:, :, ::-1].copy()
        results['img_shape'] = results['img'].shape[:2]
        results['ori_shape'] = results['img_shape']
        return results


@TRANSFORMS.register_module()
class LoadSolarAnnotations(BaseTransform):
    def transform(self, results):
        mask = read_mask(results['seg_map_path'])
        if mask.shape != results['ori_shape']:
            raise ValueError(f"Image/label size mismatch: {results['seg_map_path']}")
        results['gt_seg_map'] = mask
        results.setdefault('seg_fields', []).append('gt_seg_map')
        return results
