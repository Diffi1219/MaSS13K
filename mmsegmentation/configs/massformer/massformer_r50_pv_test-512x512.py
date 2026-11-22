"""Evaluate all test data; visualize 64 evenly spaced samples by default."""
_base_ = ['./massformer_r50_pv_pretrained-512x512.py']

load_from = 'work_dirs/massformer_pv_general_pretrained/epoch_50.pth'
resume = False
work_dir = 'work_dirs/massformer_pv_general_pretrained/test'
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', name='test_visualizer',
                  vis_backends=vis_backends)
custom_hooks = [dict(type='TestTensorboardHook', checkpoint_epoch=None, max_images=64)]
test_evaluator = dict(output_dir=work_dir + '/pred_masks')
