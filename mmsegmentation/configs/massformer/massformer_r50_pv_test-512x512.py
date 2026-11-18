"""Evaluate the final PV checkpoint and visualize all test results."""
_base_ = ['./massformer_r50_pv_pretrained-512x512.py']

load_from = '/home/diffi/workspace/project/codes/dis/MaSS13K/mmsegmentation/work_dirs/massformer_pv_pretrained_50ep/epoch_50.pth'
resume = False
work_dir = 'work_dirs/massformer_pv_pretrained_50ep/test'
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', name='test_visualizer',
                  vis_backends=vis_backends)
custom_hooks = [dict(type='TestTensorboardHook', checkpoint_epoch=50)]
test_evaluator = dict(output_dir=work_dir + '/pred_masks')
