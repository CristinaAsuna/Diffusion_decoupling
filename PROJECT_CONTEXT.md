# 项目交接：背景、数据和实现

更新：2026-11-01。本文是迁移后项目的永久知识摘要；[README](README.md) 提供运行命令，
[AGENTS](AGENTS.md) 约束后续修改。保留方法来路、数据契约和必要的失败教训，
不搬入旧项目的实验日记、结果表、作业记录或待执行实验计划。

## 先读这段

目标是从一张头颅侧位 Full 投影恢复患者左右侧的投影。此前已尝试直接双输出、
sum/difference 参数化，以及只预测差异的 D-only 模型。历史主线在合成 DRR 上以
256×256 训练；合成源投影与用于真实推理的 JPG 均为 512×512，进入模型前缩到
256×256。真实侧位片推理效果不理想（用户观察），尚未得到可靠的真实左右分解。
这项迁移保留可复用代码和上述背景，不代表已经解决合成到真实的泛化问题。

## 数据从哪里来

以下是历史数据的来路，**不是仓库自带的数据，也不是新运行默认使用的固定队列**。
服务器上的精确源路径见本地 `artifacts/migration/documentation_20261101/source_map.json`；
该文件随 artifacts 被 Git 忽略，clone 后需由数据持有者另行提供。

| 数据 | 内容与用途 | 不能混淆的边界 |
|---|---|---|
| 历史 Bone 合成投影 | 1,175 名患者，每人 std/aug 两个视图；既往按患者分为 1,058 训练、117 验证；F/L/R 为 512×512 float32 NPY | 生成模型的开发数据；历史存储通道到解剖侧别的映射尚未完成验证 |
| 带 landmark 的 CBCT 来源队列 | 28 个 CBCT NIfTI，22 个 FCSV，其中 21 例各有 26 个有效 3D 点 | 与 Bone 开发队列不同；其 21 例曾作为冻结外部测试，不能重新用于调参 |
| 修正侧别的 sidefix-v2 投影 | 上述 28 例的完整 std/aug F/L/R，采用 `anatomical_lps_label8_left_v2` | 旧错误侧别投影已删除，不得按旧文件名解释侧别 |
| 真实侧位片 | 50 张 512×512 JPG，无配对真实 L/R 参考 | 只支持定性观察和分布诊断，不能计算真实分解准确率 |

这些名称是原项目内部数据标识，不是已确认的公开数据集名称。Bone 上游采集/标注来源、
原 nnU-Net 标签表及真实 JPG 的原始 DICOM/spacing 来路仍不完整；本摘要不补造来源。
数据、分割权重和生成模型权重需要另外获取，不会随 Git clone 下载。

## 预处理代码怎样工作

当前实现位于 [Data_process/src/cbct_preprocess](Data_process/src/cbct_preprocess)，
由 [pipeline.py](Data_process/src/cbct_preprocess/pipeline.py) 串起以下步骤：

1. 读 CBCT NIfTI；使用已有的同网格分割，或由 nnU-Net Dataset015、
   `nnUNetTrainerNoMirroring`、`3d_fullres`、fold `all` 生成分割。
   权重配套的 plans/normalization 必须保留；当前配套 plans 使用 ZScoreNormalization。
2. 在物理 LPS 空间由 label 8/9 的质心建立中垂面，刚性对齐图像和分割。
   这是可复现的参考平面，不是已验证的全颅对称面。图像线性重采样，标签最近邻重采样。
3. 用同一个平面得到互补左右 mask，去掉的另一侧填空气 `−1000 HU`。
   [split.py](Data_process/src/cbct_preprocess/split.py) 使用
   `mu=max((HU/1000+1)*0.022, 0)`（单位 mm⁻¹）；这是历史衰减近似，未做真实设备标定。
4. [drr_geometry.py](Data_process/src/cbct_preprocess/drr_geometry.py) 与
   [drr_renderer.py](Data_process/src/cbct_preprocess/drr_renderer.py) 对 F/L/R 共用射线，
   积分 `p=∫mu dx`。Full 决定共同的翻转、内容裁剪和缩放，默认输出 512×512。
   保存的是可加的线积分，满足 `F≈L+R`，不是指数化后的探测器亮度。
5. 默认生成标准视角 std 和按患者/seed 可复现的 aug；写 NPY、体积和 manifest，
   记录物理变换、相机、像素变换、侧别及可加性质控。

患者左固定为 **LPS +X、label 8、channel 0**；右为 **LPS −X、label 9、channel 1**。
显示时鼻尖向右是另一个约定，不能据显示左右交换解剖通道。
以上描述是迁移版统一预处理；历史 Bone 文件不是本次迁移重新生成的，不能据此宣称其
生成细节或侧别已全部升级到新契约。

原项目还支持 FCSV 的 RAS→LPS 转换、共享 3D→2D landmark 投影和有效侧掩码。
这部分及 CeLDA 检测器未迁入；现有 CeLDA 匿名点也没有与 26 个语义点建立可靠映射。
本仓库不能据模型输出给出有效的 landmark 定位结论。

## 生成模型代码怎样工作

旧 Branch-UNet 用 Full 条件和带噪目标拼接输入，共享编码器/中间层，可用独立解码头。
sum/difference 版本预测总和相对 Full 的残差以及有符号左右差异。
它能保持较好的总和，但总和相等不能证明左右拆分正确；原项目因此进一步尝试 D-only。
迁移版四类配方统一使用 D-only，不能直接加载旧双输出 checkpoint。

```text
diff = L − R                 # 报告中的完整有符号差图
D = (L − R) / 2              # D-only 目标，注意二倍关系
s_D = 仅训练患者 |D| 的固定 95% 分位数
模型预测 D/s_D；恢复 D_pred 后：
L_pred = F_raw/2 + D_pred
R_pred = F_raw/2 − D_pred
```

该硬解码在浮点精度内保证 `L_pred+R_pred=F_raw`。D 不中心化、不逐图缩放、不裁剪。
condition 单独裁剪并映射 `[0,6]→[-1,1]`；解码仍用原始 Full。
按患者划分后才展开 std/aug；scale 和强度参考仅由训练患者估计。
历史 s_D 不应硬编码到新的队列，使用 `decouple prepare-data` 重新生成。

| 代码入口 | 职责 |
|---|---|
| [runtime.py](consistent_cdm/runtime.py)、[cli.py](consistent_cdm/cli.py) | prepare-data/train/evaluate/infer，共享输入适配、checkpoint 和导出 |
| [contracts/targets.py](consistent_cdm/contracts/targets.py)、[factories](consistent_cdm/factories) | 目标编码、硬解码和四类模型构建 |
| [model/diffusion.py](consistent_cdm/model/diffusion.py)、[apps/train_research.py](consistent_cdm/apps/train_research.py) | 条件扩散：单通道 v-prediction，cosine/zero-terminal-SNR、Min-SNR、EMA、0.05×x0-L1；默认 DPM-Solver++ 30 步/order 2 |
| [model/flow_matching.py](consistent_cdm/model/flow_matching.py) | 条件直线路径 flow matching、速度 MSE、EMA；默认 Heun 30 步 |
| [baseline](consistent_cdm/baseline) | ResUNet-96 的 D-L1；Pix2Pix-64 的条件 PatchGAN、hinge GAN + 100×D-L1 |
| [recipes](consistent_cdm/recipes)、[evaluation/data_metrics.py](consistent_cdm/evaluation/data_metrics.py) | 默认配方与统一 D/分支指标；保留 Full/2 基线，不自动交换左右 |

## 必须保留的历史结论和限制

- **合成域有学习信号，跨域尚未解决。** D-only 在 Bone 留出患者上优于 `Full/2`；
  锁定的外部 CBCT 投影直接配对结果未证明优于该基线。后者仍是合成投影，不能当作真实 JPG 结果。
  历史权重的存储侧别语义尚未验证，也不能自动改称已验证的解剖左/右。
- **真实 512→256 后效果差，但原因未定。** 历史 Bone loader 使用 Torch bilinear、
  不抗锯齿，先固定归一化再 resize；旧真实 JPG adapter 使用 Pillow bilinear resize 后
  p1/p50/p99 分位数适配。真实强度适配不是物理标定；FOV、尺度、噪声、响应和缩放均可能影响泛化。
  已有缩放诊断不足以把失败单独归因于分辨率；小规模 512 微调也未建立明确优势。
- **迁移实现与旧流程有差异。** 当前 `runtime.resize_raw` 和 dataset 统一先在 raw 域
  Torch bilinear resize，再做 condition 映射；真实输入同样使用该 resize。
  D sampler 不再隐式裁剪标准化 D。因此接口/权重可运行不等于复现旧输出或提高真实效果。
- **可运行与科学有效分开。** 已完成 CPU 集成、四类短训练/恢复及原 D-only diffusion/flow
  权重功能检查；迁移版真实 GPU nnU-Net 分割和完整规模训练尚未验证。
  旧探索计划、诊断脚本、512 pilot、预训练大模型适配和完整实验结果不构成迁移版默认功能。

后续首先需要核实数据/权重侧别来路，并在独立的真实参考或合格标注下验证泛化。
真实图像没有配对 L/R，不能以输出和等于 Full、视觉更清晰或生成谱相似替代准确率。

## 文档维护

本文维护项目背景、来路和方法边界；README 维护可运行接口；AGENTS 维护协作规则。
后续已验证的接口/契约变化应同步这三处相关内容；新假设和探索结果不能覆盖已验证事实。
原项目的永久文档与实验原件保留在原位置，本次只提炼摘要；本地来源索引记录其路径。
