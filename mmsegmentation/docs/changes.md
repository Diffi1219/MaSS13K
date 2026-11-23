# 修改记录

每次项目源码、配置、测试和文档修改均在此追加记录；原始数据及预训练权重不修改。

## 2026-11-22 · 01：修复迁移训练和任意尺寸推理的基础问题

- `mmseg/models/segmentors/encoder_decoder_multi.py`：移除未使用的 `ipdb` 导入。原因：推理不应依赖交互调试器。
- `mmseg/models/decode_heads/massformer_head.py`：增加默认关闭的 `norm_eval` 选项，在单图微调时固定解码头 BatchNorm 的统计量。原因：全局池化分支为 `B×C×1×1`，batch size 为 1 时原来的训练态 BN 会报错。参数名称及形状不变，原始配置行为不变。
- 同一文件：预测时保留 `pad_shape`，移除无条件覆盖为 `img_shape` 的代码。原因：先插值至补边尺寸再由 MMSeg 裁边，才能保持任意输入尺寸的空间对齐。
- 同一文件：边缘损失使用真实 0/1 目标，并单独屏蔽 255 补边。原因：原代码 `gt_edge != 255` 把所有有效像素都变为正边缘，忽略区域也没有正确排除。
- `docs/changes.md`：建立修改与原因记录。

## 2026-11-22 · 02：增加 General 光伏数据与 Windows 单卡配置

- 新增 `mmseg/datasets/solar.py`：注册二分类数据集和 TIFF/RGB、二值标签读取变换；按相对文件名配对，检测重名、缺失、尺寸错误与非法标签。原因：原数据集只读取 JPG/PNG，七类元数据不适用；这里保留背景 0，源标签 1 或 255 均转为光伏 1（源数据不使用 ignore 类）。
- 新增 `configs/massformer/massformer_r50_solar_general.py`：继承原网络，设置背景/光伏两类、512 裁块、单图训练、固定 BN 统计量、Windows `spawn/gloo`、0 个加载子进程和二分类评估指标；取消作者 Linux 主干权重路径和测试放大到 4096×3072 的步骤。原因：匹配本地数据、Windows 与 8 GB 显卡，避免无意放大遥感小图。
- 同一配置：测试补边倍数为 256。原因：解码头在 1/16 特征上按 16×16 窗口划分，需要输入尺寸可被 256 整除。
- 新增 `requirements/windows-solar.txt`：列出兼容 PyTorch 2.0 / Windows MMCV 官方编译包的依赖版本。原因：仓库根目录 `requirements.txt` 是带 Linux 路径的 `pip list` 输出，不能直接安装，原 README 版本也未覆盖 Ada 显卡适配。

## 2026-11-22 · 03：增加完整命令行工作流

- 新增 `tools/solar.py`，提供 `inspect/check/prepare/train/preview/infer`。原因：原仓库只有 Linux 分布式训练入口，缺少本地权重核验、General 数据检查、光伏迁移和结果导出的一体化流程。
- 迁移初始化只移除七类分类头（以及可能存在的分类损失权重缓冲），不修改源权重；记录 `transfer_untrained`，阻止将初始化文件误用为已训练光伏模型。原因：修改类别数并不等于学习到了光伏语义。
- 推理严格匹配全部参数，要求光伏 checkpoint 有二分类及明确的类别元数据；`preview` 单独展示原始七类结果。原因：防止形状不匹配时静默使用随机分类头，或把建筑物等类别误报为光伏。
- 增加单文件/目录递归推理、重叠滑窗、CPU 分数融合、0/1 mask、0/255 binary、彩色图与叠加图、可选标签评估和 JSON 报告；拒绝覆盖已有结果。原因：支持遥感图像批处理与大图裁块，并保留明确的类别含义。
- 训练入口提供单次训练/验证 `--smoke`、正式微调、AMP 和续训；默认路径按脚本位置解析。原因：先验证实际前向/反向与验证，再运行耗时的正式训练，避免依赖启动目录。

## 2026-11-22 · 04：增加关键行为回归验证

- 新增 `tests/test_solar_workflow.py`：验证 0/1 与 0/255 标签、非法标签、重名/缺失配对、小图与非整倍数滑窗覆盖、拒绝七类权重和未知二分类语义、初始化文件不覆盖、光伏指标、预测补边尺寸及边缘损失梯度。原因：类别语义与像素对齐出错会产生表面可运行但实际无效的结果，边缘梯度验证可防止原错误复发。

## 2026-11-22 · 05：编写中文环境与使用文档

- `docs/use.md`：补充项目结构理解、源权重七类限制、数据布局、Windows/aienv/CUDA 11.8 配置、官方依赖依据、数据核验、原模型预览、训练验证检查、正式微调、续训、批量预测、指标与故障排查。原因：用户需要可直接执行的 Windows 操作说明，并需要明确“预训练迁移”和“已训练光伏模型”的区别。
- 同一文档：注明 PNG 不保存地理参考、分数融合的内存范围、源标签 255 的含义与训练参数的起始性质。原因：限定本流程实际处理的数据类型与结果解释，避免误用。

## 2026-11-22 · 06：覆盖单图全局池化 BatchNorm

- `tests/test_solar_workflow.py`：补充对训练态 `norm_eval` 的行为检查，并实际向 BN 传入 `1×C×1×1` 特征。原因：直接验证 batch size 为 1 的全局池化分支不再触发 BN 异常。

## 2026-11-22 · 07：处理验证时发现的只读用户目录

- `tools/solar.py`：在导入模型依赖前，将未设置的 `MPLCONFIGDIR` 默认指向项目 `work_dirs/.cache/matplotlib`。原因：实际回归验证发现 Matplotlib 尝试在只读用户主目录创建字体锁，导致模型模块导入失败；使用工作区缓存即可解决，且尊重用户显式设置。
- `docs/use.md`：补充缓存位置说明。

## 2026-11-22 · 08：兼容中文 Windows 的 MSVC 信息解码

- `tools/solar.py`：使用本地 `SolarRunner`，在环境日志采集触发 `UnicodeDecodeError` 时记录警告，并补记 Python/PyTorch/CUDA/GPU/运行配置与完整训练配置后继续。原因：真实训练检查定位到 MMEngine 0.10.3 对 `cl.exe` 输出按 GBK 解码失败；本项目使用预编译算子，这段编译器信息不应阻断训练。不修改安装的 MMEngine 包，也不吞掉模型和训练异常。
- `docs/use.md`：说明该 Windows 兼容行为。

## 2026-11-22 · 09：更新实际验证结果与本机复用方法

- `docs/use.md`：将数据抽样状态更新为全量检查通过，记录真实 GPU、回归测试、一次训练/验证、权重回载、导出和非整倍数图像推理结果；增加本次工作区隔离依赖的 `PYTHONPATH` 复用命令及已存在 smoke 目录的重跑方法。原因：让文档准确反映实际运行状态，当前机器可直接复用已下载依赖，且不会把一次迭代检查误认为正式训练。

### 验证证据

- Python 解释器始终为 `C:/Users/23792/.conda/envs/aienv/python.exe`。没有使用 base、BRNet 或 massformer 环境执行 Python，也没有向这些环境安装包。验证依赖写入工作区的 `.mass13k-*-runtime` / `.mass13k-runtime` 隔离目录。
- 权重实读：原始 `dataset_meta.classes` 为七类，分类头 `(8, 256)`，state_dict 共 660 条；`preview` 的 `load_state_dict(strict=True)` 通过。
- 全量数据检查约 403.8 秒：train 17,648 对，val 1,878 对，test 4,802 对，均为 RGB/L、512×512、合法二值标签。光伏像素比例分别约 13.505%、14.473%、13.768%。
- 核心实测版本：Python 3.11.15、torch 2.0.1+cu118、torchvision 0.15.2+cu118、MMCV 2.1.0、MMEngine 0.10.3、MMDetection 3.3.0、NumPy 1.26.4；显卡为 RTX 4060 Laptop GPU。DCNv2 实际完成 CUDA 前向与反向。
- `python -m unittest discover -s tests -p test_solar_workflow.py -v`：6 项全部通过。
- `tools/solar.py train --smoke`：1 次优化迭代 + 2 张验证图完成，保存 `iter_1.pth`、`best_mIoU_iter_1.pth`。读取 checkpoint 的训练标量确认全部 loss 有限；迁移加载仅缺少计划重新初始化的 `decode_head.cls_embed.weight/bias`。
- `tools/solar.py infer ... --limit 1 --labels ...`：用 smoke 二分类权重完成真实测试 TIFF 推理、0/1 mask、0/255 binary、color、overlay 与 JSON 指标导出。该运行只是接口验证，指标不作为正式模型性能。
- 额外真实 GPU 检查：311×407 输入补边后输出保持 311×407；513×777 输入按 512/384 重叠滑窗融合后输出保持 513×777，类别均在 0/1 范围内。
- 正式训练、完整测试集精度和 AMP 性能未运行，不作已完成声明。原始数据及源预训练权重没有修改。

## 2026-11-22 · 10：改用独立 massformer 项目环境

- 用户明确纠正环境约定：`aienv` 只用于临时 Python 操作，项目安装、训练和推理必须使用独立的 `massformer`。之前将项目运行说明写为使用 aienv 的做法不合适，当前配置以本条及后续验证为准，前面的记录保留为历史。
- 已核实旧环境绝对路径为 `C:/Users/23792/.conda/envs/massformer`，按用户要求通过 Conda 删除后重建 Python 3.11 环境；其余 Conda 环境不删除。
- 新增 `environment-windows.yml`：声明项目环境名及 Windows/CUDA 对应的安装来源和依赖。原因：将项目环境与临时工具环境分开，并提供可重建的配置。
- `requirements/windows-solar.txt`：增加固定的 setuptools 与 wheel。原因：MMEngine 0.10.3 使用 `pkg_resources`，新版 setuptools 不再提供该接口，需要显式固定兼容版本。
- aienv 初步核查：项目包不存在，包清单与上一轮开始时一致；其 Conda 历史最后修改仍为 2026-11-07。用户/系统级 `PYTHONPATH`、`PYTHONHOME`、`MPLCONFIGDIR` 均无项目残留。记录包清单及历史、`.pth` 文件 SHA256，后续再次比对。
- 旧 `.mass13k-*-runtime` 及 `.mass13k-runtime` 是之前错误工作流创建的隔离依赖，将在新环境独立验证通过后清除；不盲目卸载 aienv 原有的 Pillow、setuptools 等工具包。

- docs/use.md：将全部项目命令迁移至 massformer；移除旧隔离依赖复用方案，补充独立环境重建和使用说明。原因：按用户要求彻底分离项目环境与临时 Python 工具环境。

## 2026-11-22 · 11：完成 massformer 独立验证与旧依赖清理

- `docs/use.md`：补充本机实际安装状态、独立环境验证结果、新训练/预测产物路径和 aienv 核查结果。原因：使用说明应与本机完成后的环境一致，不再引导用户把临时工具环境用于项目。
- 旧 `massformer` 已通过 Conda 完整删除，新建路径仍为 `C:/Users/23792/.conda/envs/massformer`，实际 Python 为 3.11.18。依赖正式安装在该环境的 `Lib/site-packages`，项目以 editable 方式安装。
- 实测 torch 2.0.1+cu118、torchvision 0.15.2+cu118、MMCV 2.1.0、MMEngine 0.10.3、MMDetection 3.3.0、NumPy 1.26.4、Pillow 10.4.0 均从 massformer 导入。`sys.path` 不含 aienv 或旧隔离目录，`PYTHONPATH` 为空，`pip check` 无依赖错误。
- GPU DCNv2 前向与反向、6 项回归测试均通过；在新环境实际完成 1 次训练迭代和 2 张验证图评估，保存权重至 `work_dirs/massformer_rebuild_validation/smoke/`。
- 验证成功后，核实绝对路径、父目录及非重解析点属性，删除工作区根下 `.mass13k-aux-runtime`、`.mass13k-mmcv-runtime`、`.mass13k-runtime`、`.mass13k-torch-runtime` 四个旧目录；同时移除重建时复制的临时 PyTorch wheel，保留正常 pip 缓存。原因：移除之前错误项目运行方案的残留，不再通过 PYTHONPATH 注入依赖。
- 删除旧目录后，再次使用 massformer 加载新检查权重，完成真实 TIFF 的分割、四类 PNG 导出和指标计算，产物在 `work_dirs/massformer_rebuild_validation/prediction/`。这只是环境及流程验证，不代表正式训练后的精度。
- aienv 清理核查：该环境原本未安装本项目的 torch/torchvision/numpy/mmcv/mmengine/mmdet/mmseg；前后包清单相同，Conda 历史与 `.pth` 文件 SHA256 相同。没有需要卸载的项目包，保留原有工具包；未使用或修改其他项目环境。
- 核查证据位于 `work_dirs/environment_rebuild/`：`aienv_before.json`、`aienv_after.json`、`massformer_environment.json`、`massformer_packages.txt`、`runtime_cleanup.json` 和新环境训练控制台日志。以上为生成的验证产物，不纳入源码版本控制。

## 2026-11-22 · 12：统一先激活环境、再直接执行 Python

- `docs/use.md`：移除安装、检查、训练、续训、测试、推理命令前的 Conda 包装前缀，统一先执行 `conda activate massformer`，随后直接使用 `python ...`；说明每个新终端需要激活一次，并补充重建前退出旧环境、创建后重新激活的顺序。原因：按用户要求简化日常命令，明确一次激活即可覆盖当前终端后续操作。
- 保留激活异常时直接调用 massformer 解释器的备用方式；无需重装环境或修改训练、推理代码。

## 2026-11-22 · 13：目标双卡电脑统一使用第二张显卡

- `docs/use.md`：所有 GPU 运行示例（原模型预览、训练检查、正式训练、续训、AMP/小裁块训练、推理及测试集评估）显式增加 `--device cuda:1`，并说明训练内验证自动沿用同一张卡。原因：用户明确指定目标双 RTX A4000 电脑使用第二张卡，避免误用第一张繁忙显卡。
- 补充 GPU 编号和单卡机器改用 `--device cuda:0` 的说明；区分目标双卡运行方式与原 RTX 4060 单卡验证记录。
- `tools/solar.py` 已支持该参数：训练入口先设置当前 CUDA 设备，推理按指定设备加载模型，因此无需修改代码默认值或环境依赖。保留默认值可继续支持原单卡机器，目标电脑通过文档中的显式参数选择第二张卡。

## 2026-11-22 · 14：补齐安装目录前提并排查同类操作说明问题

- `docs/use.md`：统一为用户实际使用的 Anaconda Prompt（CMD）命令，使用英文双引号、`cd /d`、`dir`、`set` 和 `REM`，修正之前混用 PowerShell 命令与单引号的问题。
- 安装和所有使用相对路径的操作块显式进入 `mmsegmentation`；单独说明 `-r requirements/windows-solar.txt`、`-e .`、环境 YAML、`tools/solar.py`、`tests` 和 `work_dirs` 对当前目录的依赖。原因：仅在前面提一次工作目录不足以支持用户独立复制后续段落。
- 用一次设置的目标电脑项目、数据及权重路径变量替代操作命令中的原电脑硬编码路径；训练、检查、预览、推理明确传递相应参数。原因：目标电脑使用 E 盘，不能照搬原 C 盘路径或原用户解释器路径。
- 区分首次创建、已有环境使用和可选删除重建；修正检查产物与正式训练的目录关系、已有输出的重跑说明、续训目录及示例权重需替换的说明。原因：避免正常使用时误删环境、覆盖现有实验或引用尚未生成的权重。
- 新增 `tools/install_windows.bat`：自动定位项目根目录，核查已激活的 massformer/Python 3.11，顺序执行六步安装，任一步失败立即停止，并在退出时恢复原工作目录。原因：支持用户一次输入后分步执行，避免多条 pip 命令粘成一行，以及在错误环境或目录安装。
- 批处理对缺失的 requirements/windows-solar.txt、setup.py 给出明确文件名；文档说明需将新批处理同步到目标电脑，避免只复制旧代码后入口不存在。
- 验证：在 CMD 中从另一个含空格的目录调用安装器，以临时 pip 拦截器验证六步顺序和真实工作目录；模拟第三步失败后只执行前三步，错误环境在安装前被拒绝，返回原目录。另核查 16 个相对路径操作块都有显式 cd，8 个 GPU 示例均指定 cuda:1。验证不执行真实 pip 安装、不下载依赖、不修改 Conda 环境；生成的模拟记录保存在 work_dirs/install validation/。

## 2026-11-22 · 15：简化为手动逐条 pip 安装

- `docs/use.md`：依用户要求，安装流程仅保留手动逐条执行，移除批处理推荐、调用命令和 YAML 自动安装操作段，重建后也返回手动安装步骤；修正相关章节引用和错误处理说明。原因：用户不希望使用批处理。
- 六条安装命令统一简写为 `pip install`，检查命令改为 `pip check`；保留原版本、下载来源与必要参数。原因：在已正确激活的 massformer 中无需重复写 `python -m`。
- 保留显式激活环境、`cd /d` 进入 mmsegmentation、核对 `pip --version` 及文件存在的步骤；说明安装时保持同一终端和目录，逐条成功后再继续。原因：简化命令不能再次遗漏环境和相对路径前提。

## 2026-11-22 · 16：区分虚拟环境与工作目录的要求

- `docs/use.md`：修正“所有安装都要保持同一终端、同一项目目录”的笼统说法。前四条按包名或下载源安装，可在任意目录执行；只有后两条按相对路径书写时需要进入 mmsegmentation。原因：安装目标由当前环境中的 pip 决定，工作目录只影响相对文件/项目路径的解析。
- 增加后两条使用绝对路径的替代写法，说明二选一；新终端只需重新激活环境并设置需要的路径变量，不强制保持同一终端。
- 同步调整准备步骤、可选重建、依赖检查及错误处理说明，避免继续暗示通用依赖安装或 pip check 依赖项目目录。

## 2026-11-22 · 17：删除目录准备章节，直接展示安装与使用命令

- `docs/use.md`：按用户要求完整删除原第二节的终端、路径变量和目录准备说明，并重排后续章节编号。
- 删除全文的目录变量、自动切目录命令、dir 检查及其引用；由用户自行定位项目目录，数据和权重位置需要调整时直接传入对应参数。原因：用户明确不需要这一层操作包装。
- 简化手动安装段，保留当前已验证的 Windows 依赖组合；进入 mmsegmentation 后按作者 README 使用 `pip install -e .`，移除额外构建控制参数和绝对路径替代安装段，并明确版本组合的来源，避免与作者原始版本混淆。
- 预测示例直接给出权重与图像路径，不再依赖终端变量；保留第二张显卡 `cuda:1` 的设置和既有验证记录。

## 2026-11-23 · 18：训练配置作为唯一默认来源

- `tools/solar.py`：训练参数的命令行默认值改为未指定状态，仅显式参数覆盖配置；新增 `--config`，默认读取光伏配置及其继承项。原因：此前脚本默认值会覆盖用户配置，造成修改配置却不生效。
- 不再自动重写学习率调度结束点、checkpoint 保存周期、launcher、裁块和数据路径；保留用户配置中的独立设置。显式 `--crop-size` 按变换类型查找 RandomCrop，而非依赖固定列表下标。
- `configs/massformer/massformer_r50_solar_general.py`：集中声明 source 权重 `pretrained_checkpoint`、目标 GPU `device`、launcher 和 resume；已有 batch size、训练次数、验证周期、工作目录和优化器设置继续作为训练默认值。显卡选择也会进入保存的配置。
- 保留用户配置的 `load_from`；仅没有可直接加载的 checkpoint 且不续训时，才从 `pretrained_checkpoint` 生成迁移初始化。显式 `--pretrained` 则请求重新准备源权重。原因：不再忽略用户在配置中指定的初始化权重。
- AMP/续训未指定时保留配置，显式 `--amp/--no-amp`、`--resume/--no-resume` 才覆盖。`--smoke` 仍是明确的检查模式，临时设为一次训练、两对样本验证及独立 smoke 目录，不写回源配置。
- `tests/test_solar_workflow.py`：增加训练配置优先级检查，覆盖真实命令行未传参数、显式参数覆盖、独立调度/保存设置不被改写，以及检查模式与续训冲突。原因：避免后续脚本默认值再次覆盖配置，测试不需要目标 GPU。
- `docs/use.md`：正式训练简化为 `python tools/solar.py train`，说明配置项、覆盖顺序、配置中的实际路径及第二张显卡默认值；同步修正检查、续训和相对工作目录说明。原因：操作文档必须与配置优先的实际行为一致。
- 验证：massformer 环境下全部 9 项回归测试通过；本机单卡显式传入 `--device cuda:0 --smoke --work-dir work_dirs/config_priority_validation`，完成一次训练、两张图验证及 iter/best 权重保存。核查保存的配置包含显式 device、检查模式的 max_iters=1/val_interval=1，同时保留配置的 batch_size=1、学习率调度 end=20000 和源权重路径；未改写源配置。产物在 `work_dirs/config_priority_validation/smoke/`，控制台记录在其上级目录 `console.log`；未实测目标电脑 cuda:1，也未执行正式 20000 次训练。

## 2026-11-23 · 19：核查原生 train.py 的单卡训练支持

- 仅补充本项检查记录，未修改 `tools/train.py`、配置文件、模型源码或环境包。原因：用户要求确认原生训练入口是否可用于当前单卡任务。
- 源码核查：`tools/train.py` 的 launcher 默认是 none；MMEngine 0.10.3 在该模式下将 distributed 设为 False，不初始化进程组、不使用 DDP，并将 SyncBN 转为普通 BN。`tools/dist_train.sh` 才显式使用 torch.distributed.launch 和 --launcher pytorch，因此原生 train.py 支持单卡，模型无需因为分布式切换而另写训练循环。
- 实测原入口：指定已准备的二分类迁移初始化权重，并通过 --cfg-options 将训练限制为 1 次更新、验证限制为 2 张图。直接运行遇到既有 MMEngine 环境采集函数的 MSVC GBK 解码错误；在一次性进程内临时替换环境信息采集结果后，原生 train.py 完成训练、验证和 iter/best 权重保存。运行结束确认 current_device=0（本机 RTX 4060 Laptop）、distributed_initialized=False；临时替换不落盘、不修改第三方包。记录及产物位于 `work_dirs/native_train_single_gpu_check/`，其中 `console_retry.log` 保存原生解码错误，`console_probe.log` 保存成功检查记录。
- 当前复用前提：原生入口未处理新增的 device/pretrained_checkpoint 配置字段，且 launcher/resume 会被自身命令行默认值覆盖。使用目标第二张卡需显式选择设备，使用预训练模型需通过 load_from 指定准备好的初始化权重；如迁回原生入口，应同时接入必要的 Windows 兼容及配置优先处理。此次未实测双 RTX A4000 的第二张卡，也未执行正式完整训练。

## 2026-11-23 · 20：统一复用原有 tools/train.py 训练入口

- `tools/train.py`：在原有配置读取、--cfg-options、Runner.from_cfg / RUNNERS.build 和 runner.train 流程中补充配置优先、单卡 device 选择及可选 pretrained_checkpoint 迁移初始化。launcher、resume、AMP 未显式指定时保留配置，提供 --no-resume / --no-amp；专用显式参数优先于 --cfg-options。原因：按用户要求复用原生训练入口，消除两套训练启动逻辑，同时保留此前确认的配置行为。
- 原生分布式与自定义 Runner 分支保留，分布式显卡继续由 LOCAL_RANK 控制；普通配置未设置 pretrained_checkpoint 时沿用原生加载方式。分布式迁移要求预先 prepare 并设置 load_from，避免多进程写同一个初始化文件。续训和显式 load_from 不重新生成迁移初始化；源权重和已有初始化文件禁止覆盖。
- `tools/train.py`：在第三方导入前指定工作区 Matplotlib 缓存，并在 Windows 默认 Runner 中保留环境信息 UnicodeDecodeError 的日志降级处理，复用原 Runner 的全部训练逻辑；不修改第三方环境包。原因：实际单卡检查已复现缓存权限和 MSVC 诊断文本解码问题，迁回原入口需保留兼容处理。
- `tools/solar.py`：移除 train 子命令、配置覆盖、训练 Runner 和检查模式等重复训练代码；保留检查、权重转换、预览、推理，原生 train.py 复用其中权重转换及 CUDA 检查函数；更新推理权重错误提示中的训练入口。原因：训练入口唯一，辅助功能继续复用。
- `tests/test_solar_workflow.py`：将训练优先级检查迁到真实原生入口，覆盖 --cfg-options 与显式参数、配置默认值、迁移初始化/续训/已有权重、分布式及自定义 Runner 分支。原因：确保入口迁移不破坏用户配置或原生扩展能力。
- `docs/use.md`：训练、一次更新检查、续训、AMP 命令统一为 tools/train.py 配置文件路径；移除旧训练子命令及专用参数，临时覆盖统一使用原生 --cfg-options，明确相对路径、配置引用和独立检查输出目录。原因：文档与唯一训练入口保持一致。
- 验证：massformer 环境下全部 10 项回归测试通过；直接执行修改后的 tools/train.py，显式 --cfg-options device=cuda:0，并限制为 1 次更新、2 张验证图。无需临时替换任何第三方函数，成功从原七分类权重生成初始化、执行训练与验证、保存 iter_1.pth / best_mIoU_iter_1.pth；Windows 环境解码错误降级日志正常生效，运行确认未使用分布式。保存的实际配置包含 cuda:0、max_iters=1，而源光伏配置仍为 cuda:1、max_iters=20000；调度 end=20000 保持配置值。记录和产物位于 `work_dirs/native_train_migration_validation/`；未进行正式训练、AMP 或目标第二张 RTX A4000 的实测。

## 2026-11-23 · 21：正式测试复用 tools/test.py，区分推理与代码测试

- `tools/test.py`：继续使用原生配置读取、test_dataloader/test_evaluator 和 runner.test 流程；接入配置优先的 launcher、单卡 device、工作区 Matplotlib 缓存，并复用 tools/train.py 的 WindowsRunner 环境信息解码兼容处理。原因：用户要求复用已有测试入口，避免将带标签推理作为唯一评估方式；原生可视化、TTA、输出参数保留，同时支持配置中的自定义 Runner。
- 测试始终使用位置参数明确指定的 checkpoint，关闭 resume，避免训练配置 resume=True 时意外自动选择其他实验权重；General 光伏数据集额外核查已训练二分类及类别元信息。原因：测试分数必须对应指定模型及正确类别。
- `tools/solar.py`：将既有二分类权重检查提取为 validate_solar_checkpoint，供原生测试和推理共用，保持五个辅助/推理子命令，仍无 train 或 test 子命令。原因：复用类别语义检查，保留无标注图像推理和滑窗导出的功能。
- `tests/test_solar_workflow.py`：新增原生测试入口检查，覆盖指定 checkpoint、禁用续训、调用测试循环及拒绝原七分类权重。原因：确保新接入的评估流程不会误训练或使用错误类别权重。
- `docs/use.md`：正式测试改用 tools/test.py，推理继续使用 solar.py infer；说明 tests/ 是代码测试目录、General/test/ 是测试数据目录，明确测试与推理的指标和输出差异。原因：消除同名 test 概念及入口混淆。
- 验证：massformer 环境下全部 12 项回归测试通过；直接运行 tools/test.py，使用 `work_dirs/native_train_migration_validation/iter_1.pth`，显式 --cfg-options device=cuda:0 test_dataloader.dataset.indices=2。完成 2 张 test 图像评估，输出逐类及平均指标，日志和结果 JSON 位于 `work_dirs/native_test_validation/`；确认指定权重成功加载、Windows 兼容处理生效，未执行训练。使用的是一次更新检查模型及测试子集，不报告为正式模型精度，未实测目标第二张 RTX A4000 或全量测试。

## 2026-11-23 · 22：简化为仅传配置文件的测试命令

- `tools/test.py`：checkpoint 位置参数改为可选；未显式传入时读取 test_checkpoint，再兼容 load_from；全部未设置则明确报错。输出目录优先读取显式 --work-dir，再读取 test_work_dir、work_dir。原因：用户希望测试与训练一样通过配置管理路径，避免每次输入长命令，同时兼容原有显式覆盖形式。
- `configs/massformer/massformer_r50_solar_general.py`：新增 test_checkpoint=None 和 test_work_dir='work_dirs/solar_test'，与训练初始化 load_from 及训练 work_dir 分开。原因：选定测试权重不应改变下次训练的初始化；尚无正式训练权重，默认 None，由用户在训练后填写真实文件，不假定某个 best_mIoU_iter_10000.pth 已存在。
- `tests/test_solar_workflow.py`：新增仅配置命令、显式 checkpoint/目录覆盖、缺少测试权重报错的检查。原因：核实缩短命令后仍使用正确模型和输出目录。
- `docs/use.md`：测试示例改为仅传配置路径，先说明填写 test_checkpoint 的前提；标记检查/原模型预览为可选，说明日常只需训练、测试、推理三个入口，prepare 已由首次训练自动完成。原因：减少不必要的日常操作，保留辅助功能用于排查。
- 验证：massformer 环境全部 15 项回归测试通过。创建工作目录内的检查配置，继承正式光伏配置，仅在该检查配置中指定已有的一次更新权重、独立测试目录、cuda:0 和 2 张测试图；实际执行 `python tools/test.py work_dirs/native_test_config_validation/experiment.py`，不传 checkpoint、--work-dir 或 --cfg-options，成功完成评估并保存结果。记录位于 `work_dirs/native_test_config_validation/`；源配置仍保留 test_checkpoint=None，检查结果不代表正式测试精度，未验证目标 GPU1。

## 2026-11-23 · 23：按公共、train、val、test 分区整理光伏配置

- `configs/massformer/massformer_r50_solar_general.py`：按公共设置、train 训练、val 验证、test 测试四个中文分区重排配置，标明各阶段入口、职责、权重和输出目录；将原来继承的 ValLoop/TestLoop 在对应分区显式声明。原因：用户希望直接识别参数属于哪个阶段，减少在文件中查找和误改。
- 验证新增独立的 val_pipeline，内容保持原来三步流程；test_pipeline 和 val/test 的 evaluator 分别定义，不再相互引用。原因：方便独立调整验证与测试，避免修改其中一方时混淆另一方。明确验证间隔在 train_cfg.val_interval、验证日志随训练保存，以及 model.test_cfg 与顶层 test_cfg 的不同用途。
- 原学习率、更新次数、批量大小、裁块、第二张显卡、路径、类别、初始化/续训及全部实际数据加载和评估设置保持原值。`docs/use.md` 同步补充分区定位和共享设置说明。原因：本次是可读性整理，不改变实验设置。
- 验证：用 massformer 环境读取重排前后继承合并的配置并逐项比较，原有 41 个顶层配置项的值全部一致，无删除或修改；仅新增 val_pipeline，其内容与验证数据加载器实际使用的原流程一致。前后配置及比较结果保存在 `work_dirs/config_layout_validation/`。本次未重新执行 GPU 训练或模型评估。

## 2026-11-23 · 24：清空历史运行目录并记录训练到测试的快速全流程

- 按用户要求清空项目 `work_dirs` 内全部历史产物，清理前为 17 个顶层目录、113 个文件、4,675,576,101 字节；已核对绝对目标在当前 mmsegmentation 项目下、无重解析点，原数据和预训练权重位于目标外。原因：移除历次检查、环境核查、预览及模型快照的混杂产物，重新生成清晰可解释的一次实验。前述第 1～23 项中引用的旧 work_dirs 产物将不再保留，文档中的记录仍作为历史说明。
- 新增 `configs/massformer/massformer_r50_solar_general_quick.py`，继承正式光伏配置，明确只执行 2 次训练更新、2 张 val 和 2 张 test，日志每次更新输出；训练及测试分到 quick_run/train 和 quick_run/test，测试加载本次 val 最佳权重并导出预测 PNG。原因：用可重复的短命令演示完整流程与日志，避免改动正式训练设置；本机使用 cuda:0，目标第二张卡需改为 cuda:1。
- 新增 `docs/quick_run.md` 记录清理范围、命令、实际结果和产物说明；`docs/use.md` 同步说明训练与测试日志可分开存放、原生日志与人工控制台记录的区别，并更新当前有效产物位置。原因：用户希望知道目录如何产生、每类日志如何阅读。
- 实测：使用 massformer 环境直接执行 quick 配置的 train.py 和 test.py，两条命令均成功；完成 2 次更新、2 张 val、2 张 test，保存 init、iter_1/iter_2、val 最佳权重和两张测试 PNG。日志中的两次 loss 为 193.8101/179.9374（日志含平滑），val mIoU=40.33%、test mIoU=54.93%、test 光伏类 IoU=11.83%；仅验证短流程，不作为正式精度。
- 核查：loss 有限；测试实际加载本次 best_mIoU_iter_2.pth；iter_2 含优化器和调度器，best 权重不含这两项；PNG 为 512×512、0/1；源预训练权重与正式配置清理前后 SHA-256 一致。原生日志、汇总和额外控制台记录均在 `work_dirs/quick_run/`，当前 work_dirs 仅有 quick_run 和自动重建的 .cache。本次未修改训练/测试算法或正式配置，未访问 aienv 安装项目依赖。

## 2026-11-23 · 25：直接修改并使用主光伏配置完成快速全流程

- `configs/massformer/massformer_r50_solar_general.py`：按用户纠正，直接将主配置设为快速检查模式，更新 2 次、val_interval=2、调度 end=2、日志/快照间隔=1，三个数据集均 indices=2；本机 device=cuda:0。原因：用户要求修改并执行该文件，而非另建 quick 配置。
- 主配置中训练和测试分别使用 `work_dirs/massformer_solar_general/train` 与 `test`；test_checkpoint 引用本次 val 最佳权重，test_evaluator.output_dir 由 test_work_dir 生成 masks 子目录。原因：只用两条简短命令即可从训练到测试，目录路径关联明确。
- 删除此前新增的独立 `massformer_r50_solar_general_quick.py`，清理其旧 `work_dirs/quick_run` 产物，重新生成主配置的实验记录。原因：避免混用两种配置和两次运行；原数据、预训练权重及历史 changes.md 记录保留。
- `tests/test_solar_workflow.py`：测试配置显式设置 test_checkpoint=None，使缺失权重和回退行为的检查不依赖主配置当前选定的测试权重。原因：主配置现在包含已指定的快速测试路径，测试用例需自行声明其前提。
- `docs/quick_run.md`、`docs/use.md`：更新为直接执行主配置的命令、当前快速参数、实际产物，并列出恢复正式训练需调整的字段。原因：本次结束后主配置仍保留快速参数，不能继续把它描述为 20000 次正式训练配置。
- 清理上次 quick_run 的 20 个文件、1,205,781,683 字节，保留历史修改记录。实测仅传主配置路径，先调用原 train.py 再调用原 test.py，无额外配置和命令行参数覆盖；成功完成 2 次更新、2 张 val、2 张 test，保存本次最佳权重和两张 512×512、0/1 PNG。日志 loss 为 193.8103、180.0586，val mIoU=40.33%，test mIoU=55.50%、光伏类 IoU=12.96%；属于流程检查，不代表正式精度。
- 全部 15 项代码测试通过；核查训练/测试配置快照使用主配置、两次 loss 有限、测试加载本次最佳二分类权重、原预训练权重 SHA-256 不变。当前产物统一在 `work_dirs/massformer_solar_general/`，训练时间戳 20261123_105810、测试时间戳 20261123_105844，汇总为 summary.json；主配置有意保留本机 cuda:0 和快速参数，未恢复完整训练，未实测目标 GPU1。

## 2026-11-23 · 26：详细核查训练目录并更正短流程的更新说明

- `docs/quick_run.md`：补充实际 checkpoint 内容核查、原始与平滑 loss 对照、学习率钩子执行顺序。原因：用户要求详细分析 train 目录，读取文件后发现 end=2 的 PolyLR 在第一轮后即降为零，之前“两次更新”的表述需明确为两次训练循环，其中仅第一次有效改变模型权重。
- 只读核查确认 iter_1、iter_2、best_mIoU_iter_2 的模型 state_dict 内容哈希相同；iter_1/iter_2 文件仍有不同的优化器/调度器/迭代和日志状态。原始两个批次 loss 为 193.8102875、166.3069305，第二条日志显示二者均值 180.0586090。配置副本与 vis_data/config.py 内容相同，两份数值 JSON 也相同。
- 本次未修改配置、模型权重、原生日志或数据，没有重新训练和生成新的运行目录；只在已有文档中保存核查结论。后续若需两轮均产生参数变化，应延长短流程学习率调度。

## 2026-11-23 · 27：接入 TensorBoard，逐次记录训练指标

- `configs/massformer/massformer_r50_solar_general.py`：在原生 LoggerHook / LogProcessor / Visualizer 流程中增加 TensorboardVisBackend，同时保留 LocalVisBackend；显式覆盖 visualizer.vis_backends，确保继承配置里的可视化器实际使用新后端。原因：用户要求通过 TensorBoard 查看 loss、学习率和训练参数，无需另写训练循环。
- logger.interval=1、log_metric_by_epoch=False，LogProcessor.window_size=1。原因：每次训练循环写入一次原始 loss 和其他训练标量，避免此前默认窗口 10 导致曲线显示平均值；验证仍按配置的 val_interval 执行并记录指标，配置全文由原生后端写入 TensorBoard Text。
- `requirements/windows-solar.txt`：增加 tensorboard==2.14.0、protobuf==4.25.8，沿用 Python 3.11 / Torch 2.0.1 / NumPy 1.26.4 的现有环境组合。原因：massformer 原环境未安装 TensorBoard；固定事件读写依赖，便于在其他电脑复现。环境安装仅针对 massformer。
- 按用户明确回复，只记录常规标量和配置，不增加各层权重/梯度分布 Hook；log_with_hierarchy=True 将新记录分为 train/ 与 val/ 标签，显存等部分通用标签保留框架原名。`docs/use.md` 第 8 节补充安装、启动、标签、Smoothing 和目录说明；`docs/quick_run.md` 标明旧记录属于接入前，避免将历史平滑日志与新原始值混淆。
- 安装及验证：仅在 massformer 中以 --no-user 安装 TensorBoard 及依赖，pip check 通过、原有 15 项回归测试通过。直接使用主配置和新 work-dir `work_dirs/massformer_solar_general/tensorboard_check` 完成两次训练循环与两张 val 图验证，保留原 train/test 目录，不修改旧产物。
- 用 TensorBoard EventAccumulator 读取实际事件文件：train/loss 的 step 为 1、2，值约 193.8103、166.3077，与同次本地 JSON 原始值一致；val/mIoU 在 step=2，约 40.33，另含 config/text_summary。事件在 tensorboard_check/20261123_120332/vis_data/，TensorBoard CLI 版本检查通过。没有安装 TensorFlow，没有修改训练循环、权重/梯度分布记录或原短流程学习率计划，未执行正式训练。

## 2026-11-23 · 28：使用 TensorBoard 配置重新执行快速训练、验证和测试

- `configs/massformer/massformer_r50_solar_general.py`：将输出改为 `work_dirs/massformer_solar_general/tensorboard_run/train`、`test`，测试权重与 masks 随对应配置变量生成。原因：用户要求重新跑当前新配置，使用新目录保留此前记录、避免覆盖已有初始化权重，命令仍只需主配置路径。
- 将短流程 PolyLR.end 从 2 改为 3，训练循环数仍为 2。原因：此前已证实 MMEngine 0.10.3 的 end=2 会使第二轮学习率为 0；本次确保两轮都使用非零学习率并检查模型参数实际变化。三个数据集仍各取两对图像，保持 TensorBoard、本地日志、逐次原始标量和配置文本记录。
- 实测仅传主配置路径依次运行原 train.py/test.py，成功完成两轮训练、2 张 val 和 2 张 test。本次主干学习率约 1e-5、5.3589e-6，loss 约 193.8103、166.0430；逐项比较初始到 iter_1 有 412 个共享条目变化，iter_1 到 iter_2 有 414 个状态条目变化，确认第二轮不再是零更新。
- 读取实际 TensorBoard 事件，核对 train/loss 的 step=1、2、两点学习率非零及配置文本/val 指标；测试加载本次最佳权重，test mIoU=49.82%、光伏 IoU=1.71%，导出两张正确尺寸和取值的 PNG。训练时间戳 20261123_163000，测试时间戳 20261123_163034；pip check 通过，未把子集短训练结果视为正式精度。
- `docs/use.md` 更新当前路径和 end=3，保留历史运行目录解释；`docs/quick_run.md` 文末追加本次执行及核查记录。未覆盖旧模型/日志，没有新增人工汇总或重定向日志文件，记录均写入既有文档及程序原生产物。

## 2026-11-23 · 29：训练期间定期保存固定四张验证图的分割效果

- 用户确认固定使用验证集排序前四张，正式训练默认间隔 1000 次；当前两步快速模式用 interval=1 验证两个阶段。主配置显式记录四个图像名及对应标签路径，提供 custom_hooks.interval、draw_last 和 alpha；新输出目录为 fixed_visualization_run/train 与 test，保留此前实验。
- 新增 `mmseg/engine/hooks/fixed_sample_visualization_hook.py` 并在 hooks/__init__.py 注册导出：按全局训练迭代触发，复用原模型 val_step、无随机增强的数据读取、SegLocalVisualizer.add_datasample 及 Local/TensorBoard 后端，生成固定样本 GT/预测左右对照。原因：原 SegVisualizationHook 按验证批次取图，不能保证每次固定同四张且按训练步数保存。
- Hook 仅主进程额外前向，不使用 test 数据或更新模型；保存并恢复各模块训练状态以及 Python/NumPy/CPU/CUDA 随机状态，避免快照预测影响 BN、dropout 或后续训练随机序列。图像通过原生后端自动保存 PNG 和 TensorBoard Images，不修改训练循环。
- `tests/test_solar_workflow.py`：增加固定图像和 step、间隔与重复回调、正常及异常路径下训练状态恢复、随机状态恢复检查。原因：额外推理不能影响训练状态，图片必须在不同训练阶段保持一致，且每次只保存一组。
- `docs/use.md`：新增第 9 节，说明四张固定图的路径、快速/正式间隔、PNG 和 TensorBoard Images 查看方式；更新当前输出目录及完整训练恢复表中的可视化间隔。原因：避免把快照与 val 指标计算混淆，也避免完整训练时误用每一步保存的快速设置。
- 验证：全部 17 项回归测试通过；直接运行主配置的原 train.py/test.py，完成两轮训练、两张 val 评估和两张 test 评估。训练时间戳 20261123_170130，测试时间戳 20261123_170316；本机 cuda:0 检查成功，未实测目标 GPU1。
- 在 step=1、2 各保存四张固定样本图，共八张 1024×512 RGB PNG。用 TensorBoard EventAccumulator 确认四个 Images 标签均含 step=1、2，本地 PNG 与事件图像像素一致，各图的 GT 面板跨 step 一致；已目视检查渲染。`docs/quick_run.md` 追加本次记录，原数据、历史实验和原始权重保留；未把短流程图像或分数作为正式模型效果。

## 2026-11-23 · 30：修复 TensorBoard 页面未显示 Images 的日志目录问题

- 只读核查确认四个图像标签及各自 step=1、2 已存在于 fixed_visualization_run 的训练事件文件；但当前 6006 端口的 TensorBoard 进程仍以 --logdir .../tensorboard_run 启动，该旧目录只有标量。原因：浏览器连接到此前启动的服务，未切换到新增固定图像后的日志目录。
- 核对监听端口与原进程命令行后，仅重启该 TensorBoard 服务，后台隐藏运行，改用 fixed_visualization_run 的绝对路径并保持 127.0.0.1:6006。重启后查询 /data/logdir 与 /data/plugin/images/tags，确认路径正确、四个 fixed_val 图像标签已由网页接口返回；新服务进程号为 38144。
- `docs/use.md`：补充没有 Images 时检查服务 logdir、停止旧服务并重启、刷新页面的排查说明。原因：切换训练输出目录后，需要同步切换 TensorBoard 的读取目录。未重新训练，未修改权重、事件文件或数据。

## 2026-11-23 · 31：固定样本效果改为原图、GT、预测、叠加四栏

- `mmseg/engine/hooks/fixed_sample_visualization_hook.py`：将原 GT/预测双叠加图改为四栏，依次为原始 RGB、黑白 GT、黑白预测、预测光伏区域与原图叠加；增加标题条，背景区域保持原图像素不变，仅预测前景按 alpha 混合橙色。原因：用户希望同时比较原图、标签、预测与叠加，避免原图被整体压暗或混淆掩码和叠加图。
- 继续复用原模型 val_step 以及 Visualizer.add_image 的 Local/TensorBoard 后端，不修改训练循环；在预处理前复制原始 GT，避免 padding 元信息影响展示。去掉不再需要的可视化器元信息和 alpha 临时修改。
- 主配置将输出移至 fixed_visualization_run/four_panel_run/train 与 test，仍位于当前 TensorBoard 服务监控根目录内；保留旧双栏结果，四张固定样本和保存间隔保持原设置。原因：显示新布局且不覆盖已有事件文件，无需再切换服务器 logdir。
- `tests/test_solar_workflow.py`：适配新的 add_image 输出并增加四栏尺寸、GT/预测黑白语义、原图保真及仅前景叠加检查。原因：确保各栏对应正确、背景不被无意改色。
- `docs/use.md` 第 9 节更新四栏语义、尺寸、当前路径和 TensorBoard 新运行选择；`docs/quick_run.md` 追加新布局的运行记录。原因：旧双栏事件保留，页面需要选择 four_panel_run 下的新运行，避免误看旧结果。
- 验证：18 项回归测试通过；直接使用主配置完成两轮训练、两张 val、两张 test。新训练时间戳 20261123_172349，生成八张 2048×544 四栏图；已目视检查并核对本地 PNG 与事件图像像素一致、原图/GT 跨 step 一致、叠加背景不变。
- 通过当前 6006 服务的图片 tags 与 metadata 接口确认新运行四个标签及 step=1、2 已加载，无需重启服务；仅只读查询接口，没有改写原事件文件或旧运行数据。本机 cuda:0 的短流程验证通过，仍不作为正式模型效果。

## 2026-11-23 · 32：逐项添加配置行尾说明，核查 TensorBoard 空白 loss

- `configs/massformer/massformer_r50_solar_general.py`：为公共环境、设备、数据读取/增强、模型、学习率计划、训练循环、日志、权重保存、验证、固定四张图及测试设置添加对应参数同行末尾的中文说明，保留公共/train/val/test 分区。原因：用户要求在配置参数后直接说明用途，便于自行调整训练，而不是另起说明行。
- 将此前继承的 AdamW、权重衰减、动量、梯度裁剪、参数组学习率/衰减倍率及主要损失系数显式列在主配置中，并说明含义，数值沿用原有效配置。原因：这些是常用训练调整项，原来仅覆盖学习率，其他值需要跳转父配置查找；未复制其余网络结构。
- 使用 massformer 环境解析修改前后的完整继承配置，规范化后逐项比较，全部一致。仍是 max_iters=2、三个 indices=2、PolyLR.end=3、logger.interval=1、window_size=1；未改训练行为、未切换正式训练或设备、未重跑训练、未修改已有权重和事件文件。
- 对当前 127.0.0.1:6006 服务进行只读核查：logdir 为 fixed_visualization_run，Scalars 插件已启用；最新 four_panel_run/train/20261123_172349/vis_data 有 39 个标量标签，train/loss 返回 step=1 的 193.8116455 和 step=2 的 166.3241882。原因：用户反馈页面没有 loss，需要先区分写入失败与页面显示问题；当前证据确认服务已加载数据，无法仅靠服务接口确定浏览器具体筛选状态。
- `docs/use.md`：补充 Scalars 直接链接、当前训练运行、train/loss 标签、清空筛选/刷新/展开分组步骤及测试运行不产生训练 loss 的说明，同时说明当前只有两次更新和配置注释不改变快速模式。原因：避免选中图片页、测试运行或残留筛选时将空白误判为未记录。

## 2026-11-23 · 33：复现并定位 Scalars 只有坐标轴、没有 loss 曲线

- 用户补充截图并确认 Smoothing=0，说明已进入图表页面；此前仅凭后端有数据给出运行/筛选建议不足以解释截图。本次继续定位实际图表渲染，而非修改训练记录频率。
- 只读检查当前 6006 服务的 loss、grad_norm 接口，数值正常。读取 massformer 已安装的 TensorBoard 2.14.0 前端实现，确认 Ignore outliers in chart scaling 根据 5%～95% 分位区间计算纵轴范围，两点数据会被同时排除。
- 用独立隐藏 Chrome 会话读取相同训练事件，实际验证：仅选 four_panel_run/train/20261123_172349/vis_data、Smoothing=0，开启异常值过滤时，train/loss 和 train/grad_norm 的计算范围均为空、纵轴均为 [0, 1]，与用户截图一致；关闭该选项后，loss 纵轴恢复 [160, 200]、grad_norm 恢复 [1000, 9000]，实际曲线恢复可见。已核对浏览器渲染截图，而非只检查接口是否有数据。
- `docs/use.md`：将取消勾选 Ignore outliers in chart scaling 放在空白曲线排查首项，说明两点数据与默认纵轴的关系、保留 Smoothing=0 以及必要时重置坐标。原因：这是本次已复现的显示问题，不需要重跑训练、改 loss 或升级依赖。
- 诊断截图保存在 work_dirs/.cache/tensorboard_loss_outliers_on.png、tensorboard_loss_outliers_off.png，独立浏览器缓存使用同级 tensorboard_browser_debug；其他探索截图也仅放在 .cache。调试浏览器已关闭，未操作用户正在使用的浏览器，也未改训练配置、权重或事件文件；用户当前页面仍需取消该复选框。

## 2026-11-23 · 34：主配置切换为第二张 RTX A4000 的完整数据训练

- `configs/massformer/massformer_r50_solar_general.py`：保留用户已修改的 device='cuda:1'、单卡 batch_size=4 和 exp1/train 输出目录，修正设备和 BN 的行尾说明，文件说明改为 Windows A4000 完整数据方案。原因：使用目标电脑第二张可见显卡，继续沿用用户当前参数，而非恢复本机单卡快速检查设置。
- 三个数据集 indices 从 2 改为 None，训练 max_iters=20000、val_interval=1000；PolyLR.end=20001，与 max_iters+1 配套。原因：退出仅两张图、两次更新的快速模式，使用完整 train/val/test；延续此前确定的调度器末次非零学习率处理，20000 次是需根据 val 收敛情况调整的起始方案。
- 权重快照及固定四图保存间隔设为 1000，保留 logger.interval=1、window_size=1、四栏布局和最后一次保存。原因：完整训练避免每次更新都保存模型及图片，同时继续逐次记录原始训练标量。保留 FP32、AdamW 基础学习率 1e-4、512×512 裁块、num_workers=0/persistent_workers=False、Windows spawn/gloo 和单进程模式，未引入未经目标机器验证的 AMP 或并行加载设置。
- test_work_dir 改为 exp1/test，掩码目录随之改为 exp1/test/masks；test_checkpoint 默认由 work_dir 与 train_cfg.max_iters 生成最终迭代权重路径，当前为 iter_20000.pth。原因：原测试仍指向历史快速检查目录，且实际最佳迭代号无法提前知道；默认最终权重确定会保存，测试验证最佳模型时需填实际 best_mIoU_iter_N.pth，训练中的最佳模型保存机制保留。
- 数据与源权重的用户现有 C 盘路径保持原值，目标电脑路径不同时在主配置中修改；未修改源权重、既有实验或 Conda 环境，也未启动目标正式训练。cuda:1 按进程可见设备编号选择，文档说明不能同时将可见设备限制为只剩一张后仍使用编号 1。
- `docs/use.md`：将当前训练、测试、参数表、推理示例和固定图片间隔统一为 A4000/exp1 完整数据方案；历史两步结果保留并标明历史用途。原因：避免文档仍声称主配置只训练两次或测试历史快速权重。TensorBoard 当前实验启动目录改为 exp1，并说明旧 fixed_visualization_run 服务不会读取 exp1、新图片在第 1000 次更新首次保存；保留每次迭代日志及两点曲线异常值过滤处理办法。
- 验证：massformer 环境中 18 项现有代码检查全部通过；解析完整继承配置，核对原 train.py/test.py 入口默认选择 cuda:1、20000 次、调度结束点 20001、1000 次各项间隔、exp1 路径和最终测试权重。实际构建三个完整数据加载器，样本数分别为 train=17648、val=1878、test=4802；读取训练 batch=4、验证/测试 batch=1，检查输入均为 3×512×512、标签值属于 0/1，并确认四张固定原图及 GT 路径存在。git diff --check 通过；本次未执行 GPU 训练，A4000 上实际显存峰值及运行速度未实测。
