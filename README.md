
<div align="center">
<h2>MaSS13K: A Matting-level Semantic Segmentation Benchmark</h2>



Chenxi Xie<sup>1,2</sup>
| Minghan Li<sup>1,2</sup> | 
Hui Zeng<sup>2</sup> | 
Jun Luo<sup>2</sup> | 
Lei zhang<sup>1,2</sup> 

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute

🚩 Accepted by CVPR2025

<a href='https://arxiv.org/abs/2503.18364'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

</div>

![MaSS13K Dataset](fig/mas13k.png)
## ⏰ TODO
- [x] Create repo.
- [x] Release arxiv paper.
- [x] Release MaSS13K Dataset.
- [x] Release code.




## 🌟 MaSS13K Datasets
The dataset now is available :[`GoogleDrive`](https://drive.google.com/drive/folders/1NooQARLS4epj6GTsf_6hzphl1APUdW7t?usp=sharing)

Download the dataset and extract in mmsegmentation/data


## 🌟 Our Framework

![MaSSFormer](fig/Framework.png)
Architecture of MaSSFormer. The model architecture is shown on the top left corner, including the image encoder, pixel decoder and transformer decoder. The detailed structure of the high-resolution pixel decoder is shown on the right of this figure.

## MaSS13K-Benchmark


### Qualitative Comparison
![Visual Comparison](fig/VisualComp.png)
Qualitative comparisons of MaSSFormer with other three baseline methods.

### Quantitative Comparison

![Mass13K-Benchmark](fig/QuantitativeComp.png)
Quantitative evaluation on MaSS13K validation and test sets.

## 1️⃣Enviroment

``` bash
conda create -n massformer python=3.10
conda activate massformer
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv==2.0.0rc4  mmengine==0.10.3  
cd mmsegmentation
pip install -e .
```
## 2️⃣Training
1. Download the dataset.  
2. Extract the dataset and put them under data/
3. Modify the data path in "configs/massformer/massformer_r50_8xb2-90k_mass-1024x1024.py" and "configs/\_base_/datasets/mass13k.py"
```bash
tools/dist_train.sh configs/massformer/massformer_r50_8xb2-90k_mass-1024x1024.py 4
```
## 3️⃣Inference

We provide a newly pre-trained weights in :[`GoogleDrive`](https://drive.google.com/drive/folders/1NooQARLS4epj6GTsf_6hzphl1APUdW7t?usp=sharing)

## General 光伏数据训练与 TensorBoard

`develop` 分支的配置为 `mmsegmentation/configs/massformer/massformer_r50_pv-512x512.py`。
模型使用 ResNet-50、MaSSFormer 高分辨率像素解码器和 Transformer 解码器，进行背景/光伏二分类。
按本项目的使用要求，**General/val 的 1878 张图用于训练，General/train_old 的 404 张图用于测试**。
训练过程中的评估也使用 `train_old`；日志中仍称为 `val`，TensorBoard 中对应 `mIoU`、`mDice` 等指标。

```bash
conda activate massformer
cd /home/diffi/workspace/project/codes/dis/MaSS13K/mmsegmentation
python -m pip install -r requirements/tensorboard.txt
python tools/train.py configs/massformer/massformer_r50_pv-512x512.py
```

正式对照实验均为 **FP32、batch size 2、50 个 epoch**，每个 epoch 有 939 个训练步。
使用 2 个数据加载进程，主干 BatchNorm 固定统计量。
AdamW 的解码器学习率为 `1e-4`，主干为 `1e-5`；第一个 epoch 线性预热，
后续 49 个 epoch 按 iteration 进行余弦衰减。每 5 个 epoch 评估一次；
A 组每 5 个 epoch 保存 checkpoint，B 组每个 epoch 保存 checkpoint，
保留最近两个 checkpoint（包含优化器和调度器状态），不按测试指标挑选最佳模型。

两组使用相同的数据、增强、seed=42、优化器、边缘损失修正和可视化样本，仅初始化不同：

| 实验 | 配置 | 初始化 | 输出目录 |
| --- | --- | --- | --- |
| A | `massformer_r50_pv-512x512.py` | 缓存的 ImageNet ResNet-50 主干；分割头重新初始化 | `work_dirs/massformer_pv_resnet50_50ep` |
| B | `massformer_r50_pv_pretrained-512x512.py` | `/home/diffi/workspace/project/weights/cv/MaSS13K/iter_80000.pth`；分类层重新初始化 | `work_dirs/massformer_pv_pretrained_50ep` |

B 组原始权重为七分类；658 个张量形状匹配，仅 `cls_embed.weight/bias` 不匹配。
MMEngine 加载匹配的层，分类层保留二分类配置下的新初始化；启动日志中相应的 size mismatch
是预期提示。这里使用 `load_from` 做微调，`resume=False`，不恢复原权重的训练轮次和优化器。

当前 B 组从 `epoch_30.pth` 完整断点续训至第 50 轮，每轮保存模型、优化器和学习率状态，
固定图片仍每 5 轮记录。续训配置保存在输出目录的 `resume_epoch30_config.py`。
以后再次中断时，下面的命令会自动选择最新断点（仅在训练已停止时执行）：

```bash
python tools/train.py work_dirs/massformer_pv_pretrained_50ep/resume_epoch30_config.py \
  --resume --cfg-options load_from=None
```

`load_from=None` 使 `--resume` 使用输出目录中的 `last_checkpoint`，
避免重新从七分类预训练权重或固定的第 30 轮断点开始。

依次执行两组实验：

```bash
python tools/run_pv_comparison.py
```

默认顺序为 A 组后 B 组。先运行完整预训练权重微调时，使用独立的队列目录：

```bash
python tools/run_pv_comparison.py --order pretrained-first \
  --state-dir work_dirs/pv_comparison_pretrained_first_50ep
```

队列开始时冻结两组配置；第一组成功完成后才启动第二组，任一组失败则停止队列。
配置快照和运行状态保存在队列目录（默认 `work_dirs/pv_comparison_50ep/`），其中 `status.json`
记录队列与训练进程 PID、状态和时间（Asia/Shanghai）。正式训练的终端输出分别写入
两组输出目录的 `console.log`，MMEngine 日志及 TensorBoard 事件在各自时间戳目录下。
同一队列目录已有 `status.json` 时，脚本会拒绝重复启动。

另开终端启动 TensorBoard：

```bash
conda activate massformer
cd /home/diffi/workspace/project/codes/dis/MaSS13K/mmsegmentation
tensorboard --logdir work_dirs --host 127.0.0.1 --port 6006 --samples_per_plugin images=1000
```

浏览器打开 <http://localhost:6006>。在 **Scalars** 查看训练 loss、学习率、IoU、Dice 等；
在 **Images** 搜索 `fixed_train__`，选择图片并拖动 Step 滑块，对比第 0、5、10、15…个 epoch。
图片的 Step 表示已完成的 epoch；训练 loss 的 Step 表示 iteration。
每张图从左到右为 **原图 / 真实标签 / 当前预测 / 预测叠加图**，白色表示光伏、黑色表示背景。

固定的 4 张图片均来自训练集 `General/val`，由配置中的
`custom_hooks[0].sample_names` 指定，可替换为该目录下其他 `.tif` 文件名。
可视化使用无随机增强的原图，不影响训练数据增强，推理结束后恢复训练模式。
第 0 个 epoch 是初始权重的对照；此后只在 5 的整数倍 epoch 记录图片。
PNG 备份在 `work_dirs/<运行名>/<时间戳>/vis_data/vis_image/`，
固定图片与标签的路径清单在 `work_dirs/<运行名>/fixed_train_samples/samples.json`。

边缘监督检查使用训练集全部 1878 张标签：1863 张具有内部边界，生成的 Edge Mask
同时包含 0、1；另有 15 张全为光伏的标签，图内没有标注边界，Edge Mask 全为 0。
边缘像素约占 1.297%；检查确认它们位于标注边界附近，没有落在距边界较远的内部区域。
这些边界遵循语义标注的粒度；标注没有分开的单块光伏板，不会凭空生成板间边界。

模型原始边缘输出为 `[2,1,256,256]`，在计算 BCEWithLogitsLoss 前双线性上采样为
`[2,1,512,512]`，与 Edge Mask 完全一致，保留 0/1 目标并忽略 255。
检查报告、4 张边界叠加图和诊断 TensorBoard 图片保存在 `work_dirs/pv_edge_audit/`；
在 TensorBoard Images 搜索 `edge_audit__` 可查看。
两种初始化各用真实增强数据做了 10 步 FP32 前向/反向检查，损失和梯度均为有限值。
完整权重组初始边缘损失较高，记录保留在检查报告中，训练时继续观察其变化。

复现检查（不改变正式训练进度）：

```bash
python tests/test_models/test_heads/test_massformer_edge_loss.py
python tools/check_pv_edge_supervision.py \
  configs/massformer/massformer_r50_pv-512x512.py \
  configs/massformer/massformer_r50_pv_pretrained-512x512.py --steps 10
```

正式训练结束后，可在 `train_old` 上单独测试（替换为实际 checkpoint）：

```bash
python tools/test.py configs/massformer/massformer_r50_pv-512x512.py \
  work_dirs/massformer_pv_resnet50_50ep/epoch_50.pth
```

当前微调模型的独立测试与 TensorBoard 展示使用：

```bash
python tools/test.py configs/massformer/massformer_r50_pv_test-512x512.py \
  work_dirs/massformer_pv_pretrained_50ep/epoch_50.pth \
  --out work_dirs/massformer_pv_pretrained_50ep/test/pred_masks
```

测试使用 `General/train_old` 全部 404 张图，FP32、batch size 1。结果统一位于
`work_dirs/massformer_pv_pretrained_50ep/test/`：

- `metrics.json`：总体指标、背景/光伏各类指标、混淆矩阵，指标单位为百分数。
- `per_image_metrics.csv`：每张图的光伏 IoU、Dice、Precision、Recall。
- `pred_masks/*.png`：可供后续计算使用的单通道 0/1 预测掩码。
- `<时间戳>/vis_data/events.out.tfevents.*`：TensorBoard 指标和全部 404 张对比图。
- `image_manifest.json`：样本序号、源图与标签路径、TensorBoard 图片标签的对应关系。

在同一个 <http://localhost:6006> 中选择 `test/<时间戳>/vis_data`：
Scalars 搜索 `test/` 查看总体指标，`test_class/pv/` 查看光伏类别指标，
`test_samples/` 的横轴为 CSV 中的样本序号。Images 搜索 `test_images/`，
可按文件名查找任意测试图；每张图为原图、标注、预测、叠加图四列，Step=50 对应最终权重。
测试对比图只写入 TensorBoard 事件文件，不另外保存一套展示 PNG。
`pred_masks/` 中保存的是原始类别编号，不是彩色展示图，直接打开时通常接近黑色。

## 🌏 Citation

```bash
@InProceedings{Xie2025mass13k,
    author    = {Chenxi Xie, Minghan Li, Hui Zeng, Jun Luo and Zhang, Lei},
    title     = {MaSS13K: A Matting-level Semantic Segmentation Benchmark},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```


### License
This project is released under the [Apache 2.0 license](LICENSE).

### Acknowledgement

### Contact
If you have any questions, please contact: chenxi.xie@connect.polyu.hk


<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=xiechenxi99.MaSS13K)

</details>
