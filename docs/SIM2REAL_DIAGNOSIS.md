# 真实域塌缩：诊断与改进执行指南（给 coding agent）

> 状态：**计划 / 开放问题**，不是已验证结论。执行前先读 `AGENTS.md`、`PROJECT_CONTEXT.md`，以及 README 的第 2–5 节。
> 本文只规定步骤和判据。得到的结论以诊断结果为准；经核实后，把结论精简写入 `PROJECT_CONTEXT.md`，不要在这里堆实验日志。

## 现象与假设

在真实侧位片（厂商处理后的 JPG/PNG，没有 PixelSpacing）上推理时，L/R 和 F/2 几乎一样，只有局部亮度差，也就是 D 的能量很低。合成 val 上的指标看起来好，但合成 val 的 PSNR 主要由 F/2 这部分决定。

候选原因，按先验可能性排序：
1. **强度域。** 训练输入是线性 μ 线积分；真实片经过 exp/log、散射、PSF、噪声和厂商多尺度增强。推理时只做 p1/p50/p99 映射，撤不掉这些处理。
2. **几何。** DRR 用 SAD 1000 / SID 1500，两侧放大率比约 1.16；头颅定位仪约 1.10。靠近探测器的是哪一侧也还没核对。
3. **FOV 与尺度。** DRR 是紧裁剪后拉伸到 512；真实片带颈椎、标尺、耳杆等，mm/px 也不同。
4. **训练没有强度或退化增广。**
5. 分辨率 512→256 是次要因素，历史 512 微调没有明显收益。

## 前置条件（在有数据的环境里）

- 环境：`pip install -e .[test]`。不需要分割 extra。
- 本地路径写在被忽略的 `artifacts/migration/` 里，不要写进被跟踪的文件。
- 需要的东西：
  - `decouple prepare-data` 产出的 `artifacts/prepared`。必须和 checkpoint 的 `prepared_id` 一致；如果是历史权重，用对应的 prepared 版本。
  - 待诊断的 D-only checkpoint（diffusion 用 `best_ema`，flow 用 `last`）。
  - 真实片目录。
- 回归测试：`python -m unittest discover -s tests -v`。

## 阶段 1：诊断（不训练，只用 val，不碰锁定的外部测试集）

1. **基线确认。** 运行 `decouple evaluate`，记录模型和 F/2 的 `asymmetry.energy_ratio`、`identity`、`nmae`。如果在合成 val 上 energy_ratio 就明显小于 1，说明模型本身就偏保守，要先在报告里写明。
2. **退化压力测试。**
   ```bash
   decouple stress-test --prepared artifacts/prepared --checkpoint <ckpt> \
     --output artifacts/stress/<run> --max-cases 20 --device cuda
   ```
   再用 `--mapping raw` 跑一遍，区分退化本身的影响和分位数映射的影响。
3. **域统计与测试时扫描。**
   ```bash
   python scripts/domain_stats.py --prepared artifacts/prepared --real <real_dir> \
     --checkpoint <ckpt> --output artifacts/domain_stats/<run> --max-cases 20 --sweep
   ```
4. **几何核对（手工加脚本）。**
   - 在 5–10 张真实片上量下颌下缘的双轮廓间距，按头颅宽度归一化，再和 DRR 对比。
   - 根据采集资料或耳杆、放大标尺，确认靠近探测器的是哪一侧。
   - 用 `cbct-preprocess` 把 SAD 和 SID 改成约 1520 / 1670，另行渲染 val 子集，再跑 `evaluate`，看 energy_ratio 怎么变。重渲染需要用户授权使用 CBCT 数据。

### 判据
- **定位主因：** 在 stress-test 里，找出把 energy_ratio 从约 1 压到 ≤0.3 的算子和强度。把它在 `conditions/` 下的条件图、它的 `d_rms_over_scale`，分别和真实片的图像与统计对比。最接近真实片塌缩程度的那一项，就是主因候选。
- **预处理或几何问题：** 如果 `--sweep` 中某个反相、鼻尖朝向或缩放组合让真实片的 `d_rms_over_scale` 接近 DRR 的值，说明问题主要在预处理或几何。
- **几何因素：** 如果改用头颅定位仪几何后，在 DRR 上 energy_ratio 也明显下降，说明几何是因素之一。
- 产出一份简短报告：主因排序、支撑它的数字、`conditions/` 和真实片的对照图。报告给用户确认后，才进入阶段 2。

## 阶段 2：改进（按阶段 1 结论选择；训练和提交作业须用户授权）

1. **物理化渲染**（`Data_process/src/cbct_preprocess/`）：
   - 几何改成头颅定位仪参数并加随机化（±10%）；偏航和滚转放宽到 ±5–8°；靠近探测器的一侧与真实采集一致。
   - 按固定物理 mm/px 重采样。
   - L/R/F 仍留在线积分域，F=L+R 和 D 的契约不变。
2. **训练时退化增广：**
   - 在 `consistent_cdm/data/dataset.py` 里，对条件 F 随机组合使用 `consistent_cdm/evaluation/degradations.py` 中的算子；几何类算子要同时作用于 L/R。
   - 用 recipe 开关控制，默认关闭，并在 checkpoint 中保留配置。
   - 验收：在 stress-test 上，主因算子对应的 energy_ratio 曲线明显变平，同时干净 val 上的 nmae 不变差。
3. **两阶段线性化（若强度域是主因）：**
   - 先训练确定性网络 G：退化或厂商风格图像 → 线积分 F̂。训练用合成退化；再用公开数据集（如 ISBI 2015 ceph）和私有无配对片做分布对齐。
   - 然后 D 模型在 F̂ 上预测 D，解码改为 L=F̂/2+D、R=F̂/2−D。
   - 这是契约变更，必须同步更新 `PROJECT_CONTEXT.md`、`README.md` 和测试。
4. **无配对真实片约束：** L,R≥0；测试时增广（TTA）下 D 保持一致；用单侧投影先验或判别器要求 L、R 各自像单侧投影。
5. **监控：**
   - 训练验证时记录 `asymmetry.energy_ratio`、`identity`、`asymmetry_rich.*`，并用复合指标选 best，避免选到塌缩模型。
   - 推理 manifest 记录 `d_rms_over_scale`。
6. 等以上前提成立后，再考虑 512 或 patch 训练；resize 一律带抗混叠。

## 护栏
- 不提交作业、不下载权重、不 push，除非用户明确授权。不用锁定的外部测试集调参。
- 不自动交换左右。保留硬加性解码器。s_D 和划分只来自 `prepare-data`（只用 train）。
- 不把患者 ID、主机路径、数据和权重提交到 git。真实片没有配对 L/R，不能以视觉效果代替准确率。
