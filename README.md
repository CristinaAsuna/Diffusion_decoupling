# Diffusion decoupling

从 CBCT 生成物理一致的 Full/Left/Right 投影，并训练只预测左右差异的 diffusion、flow matching、ResUNet 和 Pix2Pix 模型。支持无配对目标的合成投影和真实侧位片推理。

## 安装

需要 Python 3.10+。先按自己的 CUDA 环境安装 PyTorch，再安装本项目：

```bash
git clone -b migration/core-pipeline https://github.com/CristinaAsuna/Diffusion_decoupling.git
cd Diffusion_decoupling
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[preprocess]'
decouple --help
cbct-preprocess --help
```

仅训练/推理可安装 `pip install -e .`，无需 SimpleITK 或 nnU-Net。自动分割另外安装 `pip install -e '.[segmentation]'`；监控工具使用可选的 `monitoring` extra，默认关闭。

## 1. CBCT → F/L/R

输入为平铺目录中的 `.nii`/`.nii.gz`，可带 `_0000` 后缀；每个文件对应一个患者。首版不读取 DICOM series。

### 自动 nnU-Net 分割

代码不包含训练权重。需要自行提供与 Dataset015 匹配的完整模型目录：

```text
<nnUNet_results>/Dataset015_CTDT/
  nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/
    dataset.json
    plans.json
    fold_all/checkpoint_final.pth
```

适配器按 `nnunetv2==2.6.4` 整理，使用 `nnUNetTrainerNoMirroring`、`3d_fullres`、fold `all`。缺少模型时请向权重持有者获取以上文件；仓库不提供未经确认的下载地址。不要替换权重附带的 plans 或 normalization；当前模型的 plans 使用 ZScoreNormalization。检查命令会输出实际文件 SHA256。

```bash
export nnUNet_results=/path/to/nnUNet_results
cbct-preprocess check-model
cbct-preprocess dataset --images-dir /path/to/cbct \
  --output artifacts/processed
```

需要调整 GPU、射线步长或视野时，先导出配置再编辑：

```bash
cbct-preprocess export-config --recipe automatic --output artifacts/preprocess.yaml
cbct-preprocess --config artifacts/preprocess.yaml dataset \
  --images-dir /path/to/cbct --output artifacts/processed
```

### 已有分割

分割必须与 CBCT 网格、spacing、origin、direction 一致，label 8/9 分别属于患者解剖左/右侧。仅有任意二值牙齿 mask 不足以替代该分割。

```bash
cbct-preprocess export-config --recipe supplied_segmentation \
  --output artifacts/preprocess.yaml
cbct-preprocess --config artifacts/preprocess.yaml dataset \
  --images-dir /path/to/cbct --segmentations-dir /path/to/segmentations \
  --output artifacts/processed
```

单例使用 `case --image IMAGE.nii.gz --segmentation SEG.nii.gz --output artifacts/processed`。批处理逐例记录失败，存在失败时返回非零退出码。

流程包括物理中面对齐、互补左右分割、HU→衰减转换、共用射线 DRR，以及共用翻转/裁剪/缩放。默认生成 512×512 std 和可复现的 aug 投影：

```text
artifacts/processed/
  dataset_manifest.json
  PATIENT/
    manifest.json
    projections/{std,aug}_{full,left,right}.npy
    volumes/...
```

NPY 是 float32 衰减线积分，`F≈L+R`。患者左侧固定为 DICOM LPS `+X`、label 8、channel 0；右侧为 label 9、channel 1。显示时鼻尖朝右不改变解剖侧别。此仓库不包含 landmark/FCSV/CeLDA 功能。

## 2. 固定患者划分与训练统计

```bash
decouple prepare-data --data-root artifacts/processed --output artifacts/prepared
```

默认先按患者作 90/10 划分，seed 1234，再展开 std/aug。输出 `prepared.json`、`split.json`、`d_normalization.json` 和 `calibration.json`。四种模型共享这一准备目录。每个投影都会校验形状、有限数值和可加性；生成源文件哈希，训练前检查是否改动。

默认训练尺寸为 256×256。`--image-size` 可以修改；只生成 std 时使用 `--variants std`。每例需要预处理生成的完整 manifest 与正确的解剖侧别版本，不能仅凭文件名确认左右。显式划分使用 `--split-file FILE.json`，内容为非空 `train`、`val` ID 列表及可选 `test` 列表，必须无重叠地覆盖所有患者。准备步骤不计算 test 重建指标或用其拟合统计。

```text
D = (L - R) / 2
s_D = 训练患者 |D| 的固定 95% 分位数（确定性空间采样）
网络输出 = D / s_D
L_pred = F_raw / 2 + D_pred
R_pred = F_raw / 2 - D_pred
```

D 在原始线积分域计算，不中心化、不按图缩放、不裁剪；condition 单独按固定 `[0,6]→[-1,1]` 输入网络。解码使用原始 F，因此即使 condition 超过输入范围也保持输出和等于原始 F。准备报告记录 condition 饱和比例；训练前检查该数值。

## 3. 训练、恢复与验证

```bash
decouple train --family diffusion --prepared artifacts/prepared \
  --run-id 20261031__donly__diffusion__v1 --device cuda
decouple train --family flow_matching --prepared artifacts/prepared \
  --run-id 20261031__donly__flow__v1 --device cuda
decouple train --family resunet --prepared artifacts/prepared \
  --run-id 20261031__donly__resunet__v1 --device cuda
decouple train --family pix2pix --prepared artifacts/prepared \
  --run-id 20261031__donly__pix2pix__v1 --device cuda
```

| Family | 默认配方 |
|---|---|
| diffusion | 单通道 v-prediction，cosine 1000、zero-terminal-SNR、Min-SNR γ5、EMA、0.05×x0-L1；DPM-Solver++ 30 步/order 2 |
| flow_matching | 同类条件 U-Net、直线路径 MSE、EMA；Heun 30 步 |
| resunet | ResUNet-96，normalized-D L1 |
| pix2pix | ResUNet-64 generator、多尺度 PatchGAN，hinge cGAN + 100×normalized-D L1 |

正式默认配方在 `consistent_cdm/recipes/`，均不包含数据路径或队列统计。`--config MY.yaml` 可替换训练超参数，数据 split、尺寸和 D scale 仍由 prepared 目录注入。`--max-steps`、`--batch-size` 可覆盖常用参数。较小患者队列需要适当减小 batch size。

Diffusion 的 YAML 是 `consistent_cdm/recipes/diffusion.yaml`，Flow Matching 的 YAML 是 `consistent_cdm/recipes/flow_matching.yaml`。每次启动训练还会在对应 run 中保存填入数据路径、患者划分和实际 D scale 的完整 `recipe.yaml`。

结果统一保存到 `runs/<family>/<run_id>/`：resolved recipe、准备元数据、`checkpoints/`、模型支持的 `samples/` 及 `telemetry/train.log`；监控依赖关闭时仍保存训练/验证日志。可通过 `CONSISTENT_CDM_RUNS_ROOT` 指定整个 runs 根目录。run ID 不覆盖已有结果；继续训练需要明确指定 checkpoint：

```bash
decouple train --family resunet --prepared artifacts/prepared \
  --run-id 20261031__donly__resunet__v1 --device cuda \
  --resume runs/resunet/20261031__donly__resunet__v1/checkpoints/last.pt
```

每种模型保存 `last.pt` 和验证选择的 best checkpoint。保存模型/EMA、优化器、配置、训练步数、scale 及随机状态；baseline 恢复会重新创建数据迭代器，不承诺与不中断训练逐 batch 完全相同。

```bash
decouple evaluate --prepared artifacts/prepared \
  --checkpoint runs/resunet/20261031__donly__resunet__v1/checkpoints/best.pt \
  --output artifacts/evaluation/resunet_v1 --device cuda
```

evaluate 仅使用已固定的 val 患者。保存原始预测、逐视图 CSV 和先对患者内部视图平均的患者平均汇总，含 D-NMAE、identity/gain、L/R MAE、PSNR、SSIM、sum error，以及 Full/2 对照；不自动交换左右，不进行额外后处理。

Slurm 模板不指定集群分区、账户或环境。先激活环境、创建日志父目录，再传本站资源参数：

```bash
mkdir -p runs
sbatch --chdir="$PWD" --output=runs/slurm-%j.out --gres=gpu:1 \
  jobs/train.slurm diffusion artifacts/prepared 20261031__donly__diffusion__v1 --device cuda
```

## 4. 合成与真实侧位片推理

四种模型共用入口，从 checkpoint 恢复配置、EMA 和 D scale，不需要目标 L/R：

```bash
decouple infer --checkpoint /path/to/checkpoint.pt \
  --input /path/to/full.npy --output artifacts/inference/synthetic --device cuda
decouple infer --checkpoint /path/to/checkpoint.pt \
  --input /path/to/real_images --output artifacts/inference/real --device cuda
```

支持 NPY、PNG、JPEG、TIFF 的单文件或平铺目录。NPY 默认按 raw 线积分读取；普通图像保留位深并将每图 p1/p50/p99 匹配训练集 Full 参考分位数。这只是显式强度适配，不是物理标定。外置参考可使用 `--calibration artifacts/prepared/calibration.json`；需要自定输入域时明确传 `--mapping raw|quantile`。

默认假定鼻尖朝右，不反相、不裁剪，只 resize 到 checkpoint 尺寸。可使用 `--nose-direction left`、`--invert`、`--crop X0 Y0 X1 Y1`；变换顺序为 crop→水平翻转→反相→resize→强度适配。没有隐式自动朝向检测或实验性 FOV 变换。

每张图保存 `full.npy`、`d.npy`、`left.npy`、`right.npy`、`preview.png`。根 manifest 记录输入/checkpoint 哈希、强度映射、方向、seed、solver、scale 和解码方式。原始 L/R 不裁剪、不平滑；仅预览使用固定显示窗，L/R 共用 `[0,3]`，F 使用 `[0,6]`。输出尺寸为模型尺寸，不伪造原始分辨率细节。无配对真实 L/R 时不输出重建准确率。

## 验证与维护

原生 D-only diffusion/flow checkpoint 即使没有新接口的 `config.family`，也会依据内嵌配置自动识别，保留原始网络参数及采样设置。旧 checkpoint 的存储通道顺序不自动升级为已验证的解剖侧别，manifest 明确标为 `historical_stored_order_unverified`。不兼容旧双输出 checkpoint。

可用真实 checkpoint 运行独立的功能验证：

```bash
python scripts/validate_pretrained.py --checkpoint /path/to/original/checkpoint.pt \
  --input /path/to/development_full.npy --output artifacts/validation/pretrained \
  --real-input /path/to/real.jpg --calibration /path/to/train_only_calibration.json \
  --device cpu --repeat --check-loss --check-resume
```

验证使用原配置的完整采样步数，保存 EMA 严格加载、checkpoint 步数和哈希、固定 seed 重现、训练损失前向、F/D/L/R 及硬解码恒等式的检查结果；不更新权重、不运行冻结测试集或生成准确率结论。`--check-resume` 另外调用原生训练恢复入口，验证 model、EMA、优化器状态及 step 的读取；不继续优化，也不声明跨设备或批次级完全重现。

统一 `decouple train --resume` 还会校验新流程的 `prepared_id`。旧 checkpoint 没有该 ID，因此上述状态恢复检查不等同于将旧权重自动接续到新的患者划分。

已有模型推理结果可保存在本地 `artifacts/imported/`，复制清单与校验结果在 `artifacts/migration/`。这些数据、报告及新增验证输出均被 Git 忽略，不随 clone 下载；保留原文件和原结论，不用新的采样器覆写旧结果。迁移后的 D sampler 不再隐式裁剪标准化 D，因此不承诺与旧采样代码逐像素等价。

本次原权重检查采用 diffusion 的 step 65,000 `best_ema.pt` 和 flow matching 的 step 100,000 `last.pt`。Diffusion 的旧 `last.pt` 仅到 60,000 步，Flow 的主要 `best_ema.pt` 为 99,000 步，因此不以文件名代替训练步数判断最新状态。功能验证记录在本地 `artifacts/validation/pretrained_20261031/`；历史结果已按哈希迁入本地，包含四类 D-only 模型、历史 baseline、真实侧位片和锁定最终报告，未重新评估冻结测试集。

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s Data_process/tests -v
DECOUPLE_TRAINING_SMOKE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_end_to_end_pipeline.py -v
```

集成 smoke 使用临时非对称 CBCT phantom 和已有分割，覆盖预处理→数据准备→短训练→推理。`--tiny` 仅用于小网络接口验证，不用于报告性能。数据、权重和生成产物不进入 Git。

当前代码已通过安装后脱离源码目录的 47 项 CPU 测试（含完整 phantom 流程），并验证四类模型的短训练、恢复至第 2 步及配对验证。真实 nnU-Net 资源预检查已通过；无可用 GPU，未验证真实 nnU-Net GPU 分割或完整规模训练。
