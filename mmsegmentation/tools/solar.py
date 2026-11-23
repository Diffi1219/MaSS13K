"""Photovoltaic data/weight checks and inference; train with tools/train.py."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work_dirs/.cache/matplotlib'))
WORKSPACE = ROOT.parents[3]
DATA = WORKSPACE / 'datasets/dis/General'
PRETRAINED = WORKSPACE / 'weights/cv/MaSS13K/iter_80000.pth'
SOLAR_CONFIG = ROOT / 'configs/massformer/massformer_r50_solar_general.py'
ORIGINAL_CONFIG = ROOT / 'configs/massformer/massformer_r50_8xb2-90k_mass13k-1024x1024.py'
SOLAR_CLASSES = ('background', 'solar_panel')
ORIGINAL_CLASSES = ('background', 'person', 'building', 'tree', 'ground', 'sky', 'water')


def load_checkpoint(path):
    import torch
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {path}')
    checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    key = 'decode_head.cls_embed.weight'
    if key not in state:
        raise ValueError(f'{path}: missing {key}; expected a MaSSFormer checkpoint')
    num_classes = state[key].shape[0] - 1  # final query class is no-object
    return checkpoint, state, num_classes


def inspect_checkpoint(args):
    checkpoint, state, count = load_checkpoint(args.checkpoint)
    meta = checkpoint.get('meta', {})
    print(json.dumps(dict(checkpoint=str(Path(args.checkpoint).resolve()),
                          num_classes=count,
                          classifier_shape=list(state['decode_head.cls_embed.weight'].shape),
                          dataset_meta=meta.get('dataset_meta', {}),
                          transfer_untrained=meta.get('transfer_untrained', False)),
                     ensure_ascii=False, indent=2, default=str))
    if count == 7:
        print('Original MaSS13K classes have NO solar_panel class. Fine-tune before solar inference.')


def prepare_checkpoint(source, destination):
    import torch
    checkpoint, state, count = load_checkpoint(source)
    if count != 7:
        raise ValueError(f'Transfer expects 7-class MaSS13K weights, found {count} classes')
    destination = Path(destination)
    if destination.resolve() == Path(source).resolve() or destination.exists():
        raise ValueError(f'Refusing to overwrite source or existing checkpoint: {destination}')
    removed = [k for k in state if k.startswith('decode_head.cls_embed.')
               or k == 'decode_head.loss_cls.class_weight']
    for key in removed:
        del state[key]
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state, meta=dict(
        transfer_untrained=True, source_checkpoint=str(Path(source).resolve()),
        source_num_classes=count, removed_keys=removed)), str(destination))
    print(f'Transfer initialization saved: {destination}; removed: {removed}')
    return destination


def check_data(args):
    import numpy as np
    from mmseg.datasets.solar import index_images, read_mask, read_rgb
    report = {}
    for split in ('train', 'val', 'test'):
        images = index_images(Path(args.data_root) / split / 'image')
        labels = index_images(Path(args.data_root) / split / 'label')
        missing = sorted(images.keys() - labels.keys())
        extra = sorted(labels.keys() - images.keys())
        if missing or extra:
            raise ValueError(f'{split}: missing labels={missing[:5]}, extra labels={extra[:5]}')
        keys = list(images)
        if args.limit:
            # Cover the whole sorted dataset instead of just its first region.
            keys = [keys[i] for i in np.linspace(0, len(keys)-1,
                    min(args.limit, len(keys)), dtype=int)]
        foreground = pixels = 0
        sizes = set()
        for key in keys:
            image, mask = read_rgb(images[key]), read_mask(labels[key])
            if image.shape[:2] != mask.shape:
                raise ValueError(f'Image/label size mismatch: {images[key]}')
            foreground += int(mask.sum())
            pixels += mask.size
            sizes.add(tuple(mask.shape))
        report[split] = dict(images=len(images), labels=len(labels), checked=len(keys),
                             sizes=sorted(sizes), foreground_fraction=foreground/pixels)
    print(json.dumps(report, indent=2))


def require_cuda(device):
    import torch
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('This MaSSFormer workflow requires an NVIDIA CUDA GPU for MMCV DCNv2.')
    torch.cuda.set_device(device)


def validate_solar_checkpoint(checkpoint, num_classes):
    """Require trained binary weights with verified photovoltaic class semantics."""
    meta = checkpoint.get('meta', {})
    if num_classes != 2 or meta.get('transfer_untrained', False):
        raise ValueError('Solar inference/evaluation needs a TRAINED 2-class checkpoint. '
                         'Train with tools/train.py first; preview can test the original model.')
    classes = tuple(meta.get('dataset_meta', {}).get('classes', ()))
    if classes != SOLAR_CLASSES:
        raise ValueError(f'Checkpoint class semantics are unverified: {classes}; expected {SOLAR_CLASSES}')


def build_model(checkpoint_path, device, preview=False):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    checkpoint, state, count = load_checkpoint(checkpoint_path)
    if preview:
        if count != 7:
            raise ValueError('preview expects the original 7-class checkpoint')
        cfg = Config.fromfile(str(ORIGINAL_CONFIG))
        classes = ORIGINAL_CLASSES
        palette = [[0, 0, 0], [255, 0, 0], [64, 34, 64], [0, 255, 0],
                   [93, 64, 0], [0, 0, 255], [0, 255, 255]]
    else:
        validate_solar_checkpoint(checkpoint, count)
        classes = SOLAR_CLASSES
        cfg = Config.fromfile(str(SOLAR_CONFIG))
        palette = [[0, 0, 0], [255, 80, 0]]
    require_cuda(device)
    cfg.model.backbone.init_cfg = None
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.data_preprocessor.test_cfg = dict(size_divisor=256)
    # No resizing, no annotations: inference must work on unlabeled imagery.
    cfg.test_pipeline = [dict(type='LoadImageFromNDArray'), dict(type='PackSegInputs')]
    register_all_modules(init_default_scope=False)
    init_default_scope('mmseg')
    model = MODELS.build(cfg.model)
    model.load_state_dict(state, strict=True)
    model.cfg = cfg
    model.dataset_meta = dict(classes=classes, palette=palette)
    return model.to(device).eval()


def tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def predict_tiled(model, bgr, tile_size, stride):
    """Average overlapping class scores on CPU, then classify once."""
    import numpy as np
    from mmseg.apis import inference_model
    height, width = bgr.shape[:2]
    sums = np.zeros((model.num_classes, height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    for top in tile_starts(height, tile_size, stride):
        for left in tile_starts(width, tile_size, stride):
            tile = bgr[top:top+tile_size, left:left+tile_size]
            result = inference_model(model, tile)
            scores = result.seg_logits.data.detach().cpu().numpy()
            h, w = tile.shape[:2]
            if scores.shape != (model.num_classes, h, w):
                raise ValueError(f'Unexpected prediction shape: {scores.shape}')
            sums[:, top:top+h, left:left+w] += scores
            counts[top:top+h, left:left+w] += 1
    if (counts == 0).any():
        raise RuntimeError('Sliding-window inference left uncovered pixels')
    return (sums / counts[None]).argmax(0).astype(np.uint8)


def confusion_metrics(confusion):
    tp, fp, fn, tn = (int(confusion[1, 1]), int(confusion[0, 1]),
                       int(confusion[1, 0]), int(confusion[0, 0]))
    def divide(a, b):
        return a/b if b else None
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                solar_IoU=divide(tp, tp+fp+fn),
                solar_Dice_F1=divide(2*tp, 2*tp+fp+fn),
                solar_precision=divide(tp, tp+fp), solar_recall=divide(tp, tp+fn),
                pixel_accuracy=divide(tp+tn, tp+tn+fp+fn))


def infer(args):
    import numpy as np
    from PIL import Image
    from mmseg.datasets.solar import IMAGE_SUFFIXES, index_images, read_mask, read_rgb
    model = build_model(args.checkpoint, args.device, preview=args.command == 'preview')
    source = Path(args.input).resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f'Unsupported image extension: {source}')
        images = {source.stem: source}
    else:
        images = index_images(source)
    if args.limit:
        images = dict(list(images.items())[:args.limit])
    output = Path(args.output).resolve()
    # Prevent predictions from being ingested as input, or overwriting source data.
    if not source.is_file() and (output == source or source in output.parents):
        raise ValueError('Output must be outside the input image directory')
    labels = index_images(args.labels) if args.labels else None
    if labels is not None:
        missing = images.keys() - labels.keys()
        if missing:
            raise ValueError(f'Missing evaluation labels: {sorted(missing)[:5]}')
    products = ['mask', 'color', 'overlay'] + (['binary'] if args.command == 'infer' else [])
    for key in images:
        for product in products:
            path = output / product / f'{key}.png'
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    if (output / 'report.json').exists():
        raise FileExistsError(f'Refusing to overwrite {output / "report.json"}')
    confusion = np.zeros((2, 2), dtype=np.int64)
    for index, (key, path) in enumerate(images.items(), 1):
        rgb = read_rgb(path)
        mask = predict_tiled(model, rgb[:, :, ::-1].copy(), args.tile_size, args.stride)
        color = np.asarray(model.dataset_meta['palette'], dtype=np.uint8)[mask]
        overlay = rgb.copy()
        selected = mask != 0
        overlay[selected] = (rgb[selected]*0.5 + color[selected]*0.5).astype(np.uint8)
        arrays = dict(mask=mask, color=color, overlay=overlay)
        if args.command == 'infer':
            arrays['binary'] = mask*255
        for product, array in arrays.items():
            destination = output / product / f'{key}.png'
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(destination)
        if labels is not None:
            target = read_mask(labels[key])
            if target.shape != mask.shape:
                raise ValueError(f'Evaluation label size mismatch: {labels[key]}')
            confusion += np.bincount((target*2+mask).ravel(), minlength=4).reshape(2, 2)
        print(f'[{index}/{len(images)}] {path.name}', flush=True)
    report = dict(task='mass13k_7class_preview' if args.command == 'preview' else 'solar_segmentation',
                  checkpoint=str(Path(args.checkpoint).resolve()),
                  classes=model.dataset_meta['classes'], images=len(images),
                  tile_size=args.tile_size, stride=args.stride,
                  metrics=confusion_metrics(confusion) if labels is not None else None)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('check', help='validate image/label pairing and sampled data')
    p.add_argument('--data-root', default=str(DATA))
    p.add_argument('--limit', type=int, default=50, help='samples per split; 0 checks all')
    p = sub.add_parser('inspect', help='read checkpoint class count and metadata')
    p.add_argument('--checkpoint', default=str(PRETRAINED))
    p = sub.add_parser('prepare', help='remove the original classifier for transfer initialization')
    p.add_argument('--pretrained', default=str(PRETRAINED))
    p.add_argument('--output', required=True)
    for command in ('infer', 'preview'):
        p = sub.add_parser(command, help='solar predictions' if command == 'infer' else 'original 7-class diagnostic preview')
        p.add_argument('--checkpoint', default=str(PRETRAINED) if command == 'preview' else None,
                       required=command == 'infer')
        p.add_argument('--input', default=str(DATA / 'test/image'))
        p.add_argument('--output', required=True)
        p.add_argument('--device', default='cuda:0')
        p.add_argument('--tile-size', type=int, default=512)
        p.add_argument('--stride', type=int, default=384)
        p.add_argument('--limit', type=int, default=0, help='0 processes all images')
        if command == 'infer':
            p.add_argument('--labels', help='optional label directory for binary evaluation')
        else:
            p.set_defaults(labels=None)
    args = parser.parse_args()
    for name in ('limit',):
        if hasattr(args, name) and getattr(args, name) < 0:
            parser.error(f'--{name} must be >= 0')
    if hasattr(args, 'tile_size') and not (0 < args.stride <= args.tile_size):
        parser.error('Require 0 < stride <= tile-size')
    try:
        if args.command == 'check':
            check_data(args)
        elif args.command == 'inspect':
            inspect_checkpoint(args)
        elif args.command == 'prepare':
            prepare_checkpoint(args.pretrained, args.output)
        else:
            infer(args)
    except (ValueError, FileNotFoundError, FileExistsError, ModuleNotFoundError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
