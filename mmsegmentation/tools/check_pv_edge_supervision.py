"""Audit real edge targets and run FP32 forward/backward for both initializers."""
import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from torch.utils.tensorboard import SummaryWriter

from mmseg.datasets.transforms import GenerateEdge
from mmseg.utils import register_all_modules


def audit_masks(cfg, output_dir):
    root = Path(cfg.data_root) / 'val'
    generator = GenerateEdge(edge_width=2)
    rows = []
    selected = set(cfg.custom_hooks[0].sample_names)
    kernel = np.ones((3, 3), dtype=np.uint8)
    writer = SummaryWriter(str(output_dir / 'vis_data'))
    for path in sorted((root / 'label').glob('*.tif')):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert mask.shape == (512, 512), str(path)
        mask_values = set(np.unique(mask))
        assert mask_values <= {0, 1}, str(path)
        edge = generator(dict(img_shape=mask.shape,
                              gt_seg_map=mask))['gt_edge_map']
        assert edge.shape == mask.shape
        edge_values = set(np.unique(edge))
        if len(mask_values) == 2:
            assert edge_values == {0, 1}, str(path)
        else:
            # A uniform tile has no annotated transition inside the image.
            assert edge_values == {0}, str(path)
        # Independent geometry check: an edge pixel must be near a semantic
        # transition. Erosion/dilation is independent of GenerateEdge's
        # directional-neighbour implementation.
        boundary = cv2.dilate(mask, kernel, borderType=cv2.BORDER_REPLICATE) != \
            cv2.erode(mask, kernel, borderType=cv2.BORDER_REPLICATE)
        band = cv2.dilate(boundary.astype(np.uint8), kernel) != 0
        interior = cv2.erode(mask, np.ones((7, 7), dtype=np.uint8),
                             borderType=cv2.BORDER_REPLICATE) != 0
        far_edges = int(np.count_nonzero((edge == 1) & ~band))
        interior_edges = int(np.count_nonzero((edge == 1) & interior))
        assert far_edges == 0 and interior_edges == 0, str(path)
        assert not np.array_equal(edge, mask), str(path)
        rows.append(dict(name=path.name, edge_pixels=int(edge.sum()),
                         pv_pixels=int(mask.sum()),
                         has_boundary=len(edge_values) == 2,
                         far_edge_pixels=far_edges,
                         interior_edge_pixels=interior_edges))
        if path.name in selected:
            image = cv2.imread(str(root / 'image' / path.name))
            overlay = image.copy()
            overlay[edge == 1] = (0, 0, 255)
            panels = [image, np.repeat((mask * 255)[..., None], 3, axis=2),
                      np.repeat((edge * 255)[..., None], 3, axis=2), overlay]
            labeled = []
            for panel, title in zip(panels, ['Image', 'PV label', 'Edge mask',
                                            'Edge on image (red)']):
                panel = cv2.copyMakeBorder(panel, 30, 0, 0, 0,
                                           cv2.BORDER_CONSTANT)
                cv2.putText(panel, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)
                labeled.append(panel)
            comparison = np.concatenate(labeled, axis=1)
            cv2.imwrite(str(output_dir / f'{path.stem}_edges.png'), comparison)
            writer.add_image(f'edge_audit__{path.stem}',
                             cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB),
                             dataformats='HWC')
    writer.close()
    report = dict(images=len(rows), shape=[512, 512], edge_values=[0, 1],
                  edge_pixels_pct=100 * sum(r['edge_pixels'] for r in rows) /
                  (len(rows) * 512 * 512),
                  far_edge_pixels=sum(r['far_edge_pixels'] for r in rows),
                  interior_edge_pixels=sum(r['interior_edge_pixels'] for r in rows),
                  images_with_both_edge_values=sum(r['has_boundary'] for r in rows),
                  uniform_label_tiles=sum(not r['has_boundary'] for r in rows),
                  samples=rows)
    (output_dir / 'mask_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}),
          flush=True)


def audit_model(config_path, output_dir, steps):
    cfg = Config.fromfile(config_path)
    cfg.work_dir = str(output_dir / Path(config_path).stem)
    cfg.launcher = 'none'
    cfg.log_level = 'WARNING'
    runner = Runner.from_cfg(cfg)
    runner.model.init_weights()
    runner.load_or_resume()
    runner.model.train()
    model = runner.model
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    optimizer = runner.optim_wrapper
    assert optimizer.__class__.__name__ == 'OptimWrapper'
    model_checks = []

    def check_edge_shapes(module, inputs):
        logits, targets = inputs
        assert logits.shape == targets.shape
        assert tuple(targets.shape) == (2, 1, 512, 512)
        assert logits.dtype == torch.float32
        assert set(targets.unique().tolist()) == {0.0, 1.0}
        model_checks.append(dict(aligned_logits=list(logits.shape),
                                 edge_target=list(targets.shape)))

    raw_shapes = []
    raw_handle = model.decode_head.pixel_decoder.register_forward_hook(
        lambda module, inputs, outputs: raw_shapes.append(list(outputs[1].shape)))
    loss_handle = model.decode_head.loss_edge.register_forward_pre_hook(
        check_edge_shapes)
    data_iter = iter(runner.train_dataloader)
    records = []
    try:
        for step in range(steps):
            batch = next(data_iter)
            prepared = model.data_preprocessor(batch, training=True)
            # Check the actual augmented, packed and padded targets.
            for sample in prepared['data_samples']:
                seg = sample.gt_sem_seg.data[0].cpu().numpy().astype(np.uint8)
                edge = sample.gt_edge_map.data[0].cpu().numpy()
                expected = GenerateEdge(edge_width=2)(dict(
                    img_shape=seg.shape, gt_seg_map=seg))['gt_edge_map']
                assert np.array_equal(edge, expected)
            optimizer.zero_grad()
            losses = model(**prepared, mode='loss')
            total, log_vars = model.parse_losses(losses)
            assert torch.isfinite(total), (step, log_vars)
            optimizer.backward(total)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters()
                       if p.grad is not None), f'nonfinite gradients at {step}'
            optimizer.step()
            records.append(dict(step=step + 1, loss=float(total.detach()),
                                edge_loss=float(losses['decode.edge_loss'].detach())))
            print(f'{Path(config_path).stem}: {records[-1]}', flush=True)
    finally:
        raw_handle.remove()
        loss_handle.remove()
    report = dict(config=config_path, raw_edge_shapes=raw_shapes,
                  loss_shapes=model_checks, steps=records,
                  peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20)
    (output_dir / f'{Path(config_path).stem}_model_audit.json').write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--output-dir', default='work_dirs/pv_edge_audit')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--masks-only', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(0)
    torch.set_num_threads(4)
    register_all_modules()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_masks(Config.fromfile(args.configs[0]), output_dir)
    if not args.masks_only:
        assert torch.cuda.is_available()
        for config in args.configs:
            audit_model(config, output_dir, args.steps)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    main()
