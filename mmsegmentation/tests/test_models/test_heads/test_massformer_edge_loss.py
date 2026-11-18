"""Regression checks for binary edge supervision and ignored pixels."""
import math
import unittest
from unittest.mock import Mock

import torch
import numpy as np

from mmseg.models.decode_heads.massformer_head import MaSSFormerHead
from mmseg.datasets.transforms import GenerateEdge


class TestMaSSFormerEdgeLoss(unittest.TestCase):

    def setUp(self):
        # Exercise loss_by_feat without allocating the transformer decoder.
        self.head = MaSSFormerHead.__new__(MaSSFormerHead)
        torch.nn.Module.__init__(self.head)
        self.head.ignore_index = 255
        self.head.loss_edge = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.head._loss_by_feat_single = Mock(
            return_value=(torch.tensor(0.),) * 3)

    def edge_loss(self, logits, targets):
        return self.head.loss_by_feat(
            [torch.zeros(1)], [torch.zeros(1)], [None], [{}],
            targets, logits)['edge_loss']

    def test_binary_targets_and_ignore_gradient(self):
        logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
        targets = torch.tensor([[[[0, 1], [255, 0]]]])
        loss = self.edge_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), 10 * math.log(2), places=5)
        loss.backward()
        self.assertGreater(logits.grad[0, 0, 0, 0].item(), 0)
        self.assertLess(logits.grad[0, 0, 0, 1].item(), 0)
        self.assertEqual(logits.grad[0, 0, 1, 0].item(), 0)

    def test_all_ignored_has_zero_finite_loss(self):
        logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
        loss = self.edge_loss(logits, torch.full((1, 1, 2, 2), 255))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_extreme_logits_have_finite_loss_and_gradient(self):
        logits = torch.tensor([[[[-1000., 1000.]]]], requires_grad=True)
        loss = self.edge_loss(logits, torch.tensor([[[[1, 0]]]]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_generated_edges_follow_region_boundary(self):
        seg = np.zeros((32, 32), dtype=np.uint8)
        seg[8:24, 8:24] = 1
        edge = GenerateEdge(edge_width=2)(
            dict(img_shape=seg.shape, gt_seg_map=seg))['gt_edge_map']
        self.assertEqual(set(np.unique(edge)), {0, 1})
        self.assertEqual(edge.shape, seg.shape)
        self.assertEqual(edge[16, 16], 0)  # PV region interior.
        self.assertEqual(edge[0, 0], 0)  # Distant background.
        self.assertEqual(edge[8, 16], 1)  # Label boundary.
        self.assertFalse(np.array_equal(edge, seg))

    def test_low_resolution_prediction_aligns_before_loss(self):
        logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
        targets = torch.zeros(2, 1, 16, 16, dtype=torch.long)
        targets[:, :, 4:12, 4] = 1
        observed = []
        handle = self.head.loss_edge.register_forward_pre_hook(
            lambda module, inputs: observed.append(
                (tuple(inputs[0].shape), tuple(inputs[1].shape))))
        try:
            loss = self.edge_loss(logits, targets)
            loss.backward()
        finally:
            handle.remove()
        self.assertEqual(observed, [((2, 1, 16, 16), (2, 1, 16, 16))])
        self.assertEqual(tuple(logits.grad.shape), (2, 1, 4, 4))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_uniform_tiles_have_no_internal_boundary(self):
        for value in [0, 1]:
            seg = np.full((32, 32), value, dtype=np.uint8)
            edge = GenerateEdge(edge_width=2)(dict(
                img_shape=seg.shape, gt_seg_map=seg))['gt_edge_map']
            self.assertEqual(set(np.unique(edge)), {0})


if __name__ == '__main__':
    unittest.main()
