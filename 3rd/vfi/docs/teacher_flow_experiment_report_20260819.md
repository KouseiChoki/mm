# VFI Teacher Flow 实验阶段报告

**报告日期：** 2026-08-19

**实验目标：** 验证 Unreal 原生 MV 的正确性，判断 ground-truth MV 是否能改善 VFI 光流与最终插帧质量，并确定可复用的 teacher-flow 训练策略。

**基线模型：** S2V3 official-style、`0729_lc_v3s2`

**Teacher 数据：** Unreal、Spring、Sintel、FlyingThings3D

> 说明：实验期间验证清单和数据规模有过调整，因此本文只对同一初始化、同一数据清单和同一训练步数的成对消融做数值比较。不同批次实验的绝对 PSNR 不用于横向排名。

## 1. 结论摘要

本阶段得到五项主要结论：

1. **Unreal MV 的方向、通道顺序、单位和缓存均已验证正确。** 当前约定为 `mv1: 中间帧 -> 前帧`、`mv0: 中间帧 -> 后帧`，缓存顺序为 `mv1_x, mv1_y, mv0_x, mv0_y`。在 501 组确定性样本上，使用当前 MV 的重建误差显著低于零流、反号、交换方向和错误尺度等对照，因此 teacher 失败不能归因于 MV 符号或单位错误。
2. **混合 teacher 数据直接监督无效。** 在相同官方 checkpoint 上，raw MV 和 hard-cycle MV 都没有降低运动区域 EPE，反而使 Hard PSNR 分别下降 0.130 dB 和 0.160 dB。混合数据上的整体 EPE 下降主要来自静止背景，不代表运动估计变好。
3. **FlyingThings3D + hard-cycle 是目前唯一得到正向量化证据的 teacher 方案。** 150 轮受控实验中，FlyingThings moving EPE 相对 noflow 降低 7.10%，Hard PSNR 提升 0.047 dB；raw flow 只有 0.88% 的 EPE 改善，基本可以视为无效。
4. **只训练局部 IFBlock 是错误方向。** 与全模型联合适配相比，IFBlock-only 的 FlyingThings moving EPE 恶化 22.60%，Hard PSNR 下降 0.549 dB。GT MV 不是只修改末端 flow head 就能吸收的监督，粗尺度运动、上下文特征和融合模块需要联合适配。
5. **Teacher EPE 下降不等于实际插帧质量提升。** 在 `0729_lc_v3s2` 上迁移 hard-cycle 后，FlyingThings moving EPE 在第 30 轮下降 3.55%，但生产素材实测出现局部运动边界退化，cycle-140 的实际效果很差。因此 teacher flow 不能作为成熟模型的后期 repair，也不能按 EPE 单独选 checkpoint。

最终建议是：**保留 FlyingThings3D + hard-cycle，定位为新模型前 20%–30% 训练阶段的低权重运动辅助；不再用于 0729 成熟模型的后期修补。**

## 2. Teacher Flow 在当前代码中做了什么

Teacher 样本由前帧、GT 中间帧和后帧组成，时间点固定为 `t=0.5`。模型仍以中间帧重建损失为主，同时在多级 flow 输出上增加直接监督：

- GT 双向流：`mv1` 对应中间帧到前帧，`mv0` 对应中间帧到后帧；
- 监督对象：模型各级双向 flow，后级权重更高；
- 有效区域：raw 模式使用原始有效像素，cycle 模式使用前后向一致性过滤后的像素；
- 区域平衡：按 `flow_motion_threshold=1.0 px` 区分运动区与静止区，并提高运动区采样/损失占比；
- 总损失：图像重建损失仍是主项，flow loss 仅以小权重叠加；
- 评估：同时记录整体 EPE、moving EPE、static EPE、插帧 PSNR，以及最终真实素材的边界与闪烁表现。

hard-cycle 使用前后向一致性检查。对流 `f` 和反向流 `b`，在 `f` 的终点采样 `b`，当残差满足下式时才保留监督：

```text
residual² = ||f + warp(b, f)||²
threshold = 0.05 × (||f||² + ||warp(b, f)||²) + 1.0
valid = residual² <= threshold
```

它主要用于排除遮挡、越界、尾帧无对应流和错误对应，而不是修改 MV 数值本身。

## 3. 源头验证：MV 是否正确

### 3.1 方向、符号与单位

对 501 组确定性 triplet 做 warp 重建，并与交换 `mv0/mv1`、XY 反号、0.5 倍、2 倍及零流对照。当前顺序、符号和尺度的误差最低：

| 图像域 | 零流 moving-region MAE | 当前 MV moving-region MAE | 结果 |
|---|---:|---:|---|
| clean | 0.043442 | **0.011944** | 当前 MV 明显更好 |
| final | 0.039019 | **0.012554** | 当前 MV 明显更好 |

由此确认：

- `mv1 = middle/current -> previous input`；
- `mv0 = middle/current -> next input`；
- EXR 中存储的是按宽高归一化的位移，Dataset 读取后恢复为 pixel flow；
- 当前 XY 符号均为正向约定，不需要额外反号；
- cache 通道顺序为 `mv1_x, mv1_y, mv0_x, mv0_y`。

### 3.2 clean MV 复制到 final 是否可用

同一组 Unreal dump 中，clean 与 final 的几何和相机运动一致，复制后的 MV 文件抽检为字节一致。final 尾部存在 5 个无配对帧，这是旧 teacher 中明显异常值的来源之一；hard-cycle 会将这些帧标记为零有效监督，避免错误梯度。

### 3.3 缓存是否改变数值

EXR 与 float16 训练缓存抽样比较一致，未发现通道交换、单位变化或精度导致的可见误差。缓存读取约为 3.59 ms/sample；在线计算 cycle 约为 13.06 ms/sample，因此正式训练采用预生成 cache 和 cycle confidence。

**源头结论：** MV 生成与读取链路成立。后续问题主要来自数据域、遮挡有效性、teacher 占比、监督位置和 VFI 目标之间的冲突，而不是符号反向或单位错误。

## 4. 实验时间线

### 4.1 初期强 teacher 混训：暴露目标冲突

#### 0807 `s2v3_all_teacher_tuesday`

- fresh S2V3 official-style curriculum；
- teacher 占 microbatch 20%；
- `flow_loss_weight=0.01`；
- 运动/静止平衡实际偏向静止区域；
- 全模型训练，计划 240 + 80 轮，实际约训练到第 70 轮。

典型验证结果：

| Epoch | All PSNR | Hard PSNR | Teacher flow EPE | Teacher moving EPE |
|---:|---:|---:|---:|---:|
| 20 | 24.8425 | 23.7158 | 0.6474 | 0.8547 |
| 60 | 25.0108 | 23.5087 | 0.6163 | 0.9726 |

整体 flow EPE 略降，但 moving EPE 从 0.8547 恶化到 0.9726，Hard PSNR 同时下降。该实验说明：高 teacher 占比、高 flow 权重和偏静态的损失平衡会让优化目标偏离真实运动区域。由于它不是严格 A/B，只作为问题定位，不作为最终因果证据。

#### 0810 `s2v3_all_teacher_flow_fixed`

将 teacher 占比降到 10%、flow 权重降到 0.002、运动区权重提高到 0.8，并启用 MV cache。训练损失由约 0.1506 降到 0.0936，但 batch flow EPE 在 1.7–71.9 间剧烈波动，且训练未完整结束。

该轮主要完成了工程修正：确认 EXR I/O 是主要训练停顿来源，推动 float16 MV cache、cycle cache 和分数据源统计；它不能作为 teacher 有效性的结论。

#### 0811 `0729_lc_teacher_repair`

- 从 `0729_lc_v3s2` 初始化；
- teacher 占比 25%，`flow_loss_weight=0.002`；
- 全模型训练 100 轮；
- 同时将 `refine_res_scale` 从 0.25 改为 0.05，存在额外变量。

| 指标 | Epoch 10 | 最好值 | Epoch 100 |
|---|---:|---:|---:|
| All PSNR | 35.9867 | 36.0827 | 36.0809 |
| Hard PSNR | **30.2285** | **30.2285** | 30.1309 |
| Teacher flow EPE | 0.3538 | 0.2855 | 0.2864 |
| Teacher moving EPE | 0.2966 | 0.2803 | 0.2807 |

EPE 下降，但 Hard PSNR 下降，实际 teacher-repair 结果也明显变差。由于 refiner scale、teacher 比例和 flow supervision 同时变化，本轮不能精确归因，但足以否定“高比例 teacher 对成熟模型直接 repair”的方案。

### 4.2 混合 Teacher 三组受控消融：raw/cycle 均无收益

0813 三组实验从同一 `0807_s2v3_official_tuesday_320.pkl` 开始，训练 20 轮 × 802 optimizer steps，图像样本、crop、seed、学习率和训练范围保持一致：

| 组别 | Flow supervision | Flow weight |
|---|---|---:|
| noflow | 无；保留相同 teacher 图像样本 | 0 |
| raw | 原始 GT MV | 0.0005 |
| cycle | hard-cycle 过滤后的 GT MV | 0.0005 |

第 20 轮结果：

| 指标 | noflow | raw | cycle |
|---|---:|---:|---:|
| All PSNR | **37.1295** | 37.1230 | 37.1211 |
| Hard PSNR | **38.8206** | 38.6904 | 38.6605 |
| Teacher flow EPE | 0.2272 | **0.2149** | 0.2172 |
| Teacher moving EPE | **0.2892** | 0.2928 | 0.2946 |
| Teacher static EPE | 0.2052 | **0.1871** | 0.1897 |

关键观察：

- raw 的 Hard PSNR 相对 noflow 下降 0.130 dB，cycle 下降 0.160 dB；
- raw moving EPE 反而恶化 1.25%，cycle 恶化 1.86%；
- aggregate EPE 的改善几乎全部来自 static EPE。

因此，**混合 teacher 池的“整体 EPE 下降”是误导性指标**。模型更会拟合背景零运动，但没有提升真正需要解决的运动物体。

### 4.3 数据源拆分：FlyingThings3D 是唯一有效来源

为了排除域和数据质量差异，teacher 数据拆为 Unreal、Spring、Sintel、FlyingThings3D 四个 tier，并按 scene/group 划分 train/val，避免 clean/final、左右视图或同一序列泄漏。

- Unreal/Spring：高分辨率、镜头/场景占比高，与当前 256/384 crop 和小运动物体目标不完全匹配；
- Sintel：数据量较小，反向流需要保守反演，遮挡与 disocclusion 区域被排除；
- FlyingThings3D：大量独立运动物体，原生提供 `into_past` 和 `into_future`，与双向 teacher 目标最匹配。

已完成的来源消融中，Unreal、Spring 和 Sintel 没有得到与 FlyingThings 相当的正向证据；后续正式验证聚焦 FlyingThings3D。

### 4.4 FlyingThings 四组消融：找到唯一正向组合

0814 四组实验均从同一 official-style checkpoint 开始，teacher 图像统一限制为 FlyingThings3D，占比 10%，训练 150 轮 × 802 steps。前 20 轮完成主要 LR 曲线，之后低学习率巩固。

| 组别 | Flow 监督 | 可训练范围 |
|---|---|---|
| noflow | 无 | 全模型 |
| raw | 原始 GT MV | 全模型 |
| cycle | hard-cycle GT MV | 全模型 |
| cycle_ifblock | hard-cycle GT MV | 仅 local IFBlock ×2 |

第 150 轮结果：

| 指标 | noflow | raw | cycle | cycle_ifblock |
|---|---:|---:|---:|---:|
| All PSNR | 35.7333 | 35.7249 | **35.7479** | 35.6573 |
| Hard PSNR | 38.8854 | 38.8615 | **38.9326** | 38.3833 |
| FlyingThings PSNR | 23.3982 | 23.3916 | **23.4805** | 22.6711 |
| FlyingThings flow EPE | 12.7506 | 12.6355 | **11.8487** | 14.5235 |
| FlyingThings moving EPE | 13.4181 | 13.3004 | **12.4658** | 15.2831 |

相对 noflow：

- raw：moving EPE 仅下降 0.88%，All/Hard PSNR 均略降，收益不成立；
- hard-cycle：flow EPE 下降 7.07%，moving EPE 下降 7.10%，All PSNR +0.015 dB，Hard PSNR +0.047 dB，FlyingThings PSNR +0.082 dB；
- cycle_ifblock：相对全模型 cycle，moving EPE 恶化 22.60%，Hard PSNR 下降 0.549 dB。

这组实验给出了本阶段最强的因果证据：**有效性来自“FlyingThings 数据质量 + cycle 有效性过滤 + 全系统联合适配”的组合，三项缺一不可。**

### 4.5 迁移到 0729 LC：数值改善没有通过实拍验收

0817 从同一 `0729_lc_v3s2_800.pkl` 初始化，保持 LC 纹理重建设置，只训练 flow heads 与 local IFBlocks。noflow 运行 30 轮，cycle 继续到 150 轮；teacher 占比为 1/6，约 16.7%。

第 30 轮成对结果：

| 指标 | noflow | cycle | 变化 |
|---|---:|---:|---:|
| All PSNR | 35.3089 | 35.3116 | +0.0027 dB |
| Hard PSNR | 39.8256 | 39.8320 | +0.0065 dB |
| FlyingThings PSNR | 23.4381 | 23.4927 | +0.0546 dB |
| FlyingThings flow EPE | 12.9586 | 12.4973 | -3.56% |
| FlyingThings moving EPE | 13.6403 | 13.1560 | -3.55% |

cycle 在第 140 轮达到最好 FlyingThings flow：flow EPE 12.3919、moving EPE 13.0440。但实际素材检查显示，cycle-140 虽然全局画面接近原 0729，却在局部运动边界产生可见退化，用户验收结论为“实测结果很差”。

这说明：

- 合成数据上的 EPE 下降可以只反映 FlyingThings 域内适配；
- 在成熟 LC 模型上，仅修改 flow heads 会破坏原有特征、warp、mask 与 refiner 的协同关系；
- 遮挡区的单一 GT flow 不一定等价于最佳插帧解释；
- teacher 占比 16.7% 且持续 150 轮，对后期 repair 过强；
- aggregate PSNR 对小面积边界裂缝、闪烁和局部纹理退化不够敏感。

因此，**`0817_0729_lc_flowheads_cycle_140.pkl` 不应作为生产 checkpoint，也不建议继续延长该路线。**

## 5. 为什么 EPE 下降但最终 VFI 可能变差

Teacher flow 和最终插帧质量不是同一个优化目标：

1. **遮挡多解性。** GT MV 描述中间帧表面到两侧帧的几何对应，但被遮挡像素无法从两侧图像同时正确 warp 得到；最终质量还依赖 mask 和纹理补全。
2. **数据域差异。** FlyingThings 的物体运动适合训练匹配，但材质、模糊、噪声和真实摄影运动与生产素材不同。长期训练会把模型拉向合成域。
3. **模块耦合。** flow、warp、融合 mask、上下文和 refiner 共同决定输出。只让 flow head 追 GT，可能破坏原模型已经形成的补偿关系。
4. **指标被静态区域主导。** 背景像素远多于运动边界，overall EPE 和整图 PSNR 都可能掩盖小物体、细边缘和局部裂缝。
5. **时序质量未被覆盖。** 单 triplet PSNR/EPE 无法测量连续视频中的高频纹理闪烁和边界跳动。

## 6. 已证实、未证实与否定项

### 已证实

- Unreal MV 的方向、顺序、符号、尺度及 cache 正确；
- hard-cycle 能有效排除无配对尾帧和前后向不一致区域；
- FlyingThings3D hard-cycle 在受控全模型训练中能显著降低 moving EPE；
- 全系统联合适配优于只训练 local IFBlock；
- EPE 不能单独作为 VFI checkpoint 选择依据。

### 尚未证实

- FlyingThings teacher 能否提升一个从头训练的新主模型的最终生产质量；
- 只在早期使用 teacher、后期完全关闭后，正向运动先验能否保留且不伤害 LC 稳定性；
- 加入遮挡感知 mask、ROI 边界指标和时序 flicker loss 后，EPE 与生产质量是否能重新对齐。

### 已否定或不建议继续

- raw GT flow 不做 cycle 过滤；
- 混合 Unreal/Spring/Sintel/FlyingThings 后直接长期监督；
- teacher 占比高于 10% 的成熟模型 repair；
- 只训练 local IFBlock ×2；
- 按 teacher EPE 选择最终 checkpoint；
- 继续使用 0729 cycle-140 做生产或延长训练。

## 7. 后续可复用策略

如果下一轮是从头或大范围联合训练的新模型，可将 teacher flow 作为短时辅助：

```yaml
teacher_sources: [flyingthings]
teacher_sample_ratio: 0.05-0.10
flow_loss_weight: 0.0005
flow_loss_warmup_steps: 2000
flow_motion_balance: 0.8
mv_cycle_confidence: hard
mv_cycle_cache_required: true
mv_cycle_on_the_fly: false
```

训练策略：

1. 只在总训练进度前 20%–30% 注入 5%–10% FlyingThings teacher；
2. flow、backbone/context、mask 与 refiner 联合训练，不做 IFBlock-only；
3. 后 70%–80% 将 teacher 采样率和 direct flow loss 都置零，让真实/目标域重建重新主导收敛；
4. checkpoint 选择以真实测试片段为最终门槛，同时观察 ROI moving EPE、边缘梯度保持、局部 LPIPS 和连续帧 flicker；
5. 必须保留 noflow 同图像样本对照，避免把 FlyingThings 图像重建收益误认为 MV 监督收益。

对于当前 `0729_lc_v3s2`，建议冻结 teacher-repair 路线，把它保留为稳定生产基线；下一步应从模型根因改进匹配、运动表达或融合质量，再用上述 early-teacher 方案做辅助消融。



## 8. 结论

**我们已经确认 GT MV 本身是正确的，也证明 FlyingThings3D 经 hard-cycle 过滤后能够降低运动区光流误差；但后期用它修补成熟 VFI 模型会出现“EPE 更好、实拍更差”，所以后续只把 teacher flow 作为新模型早期、低比例、短周期的运动辅助，最终仍以真实视频的边界与时序稳定性验收。**
