# P1 图像管线功能：原型实现、实测与 PocoPaw 集成

- 日期：2026-11-01
- 设备：Redmi Note 12 Turbo / Poco F5（marble），Snapdragon 7+ Gen 2，所有耗时都是手机 8 线程纯 CPU 实测（Mac 负载重，计时不可信）
- 前置报告：[`PIPELINE_REPORT.md`](PIPELINE_REPORT.md) 第 8 节 P1 表、[`ALGORITHM_UPGRADE.md`](ALGORITHM_UPGRADE.md) 第 6 节路线图
- 原型：[`pipeline/`](../../../pipeline/)（`burstpipe` 命令行，C++17）与 `pipeline/tools/`（Python 实验脚本）；逐项原始记录在 [`pipeline/out/p1/RESULTS.md`](../../../pipeline/out/p1/RESULTS.md)
- App：[`app/src/main/cpp/burstpipe/`](../../app/src/main/cpp/burstpipe/)（原生管线）与 [`app/src/main/java/com/izum1k/pocopaw/pipeline/`](../../app/src/main/java/com/izum1k/pocopaw/pipeline/)（曝光规划、阶段、设置）

> 本文复制自研究工作区 `Poco-F5-RAW-Report/`。文中的 `pipeline/`、`data/`、`bin/` 都指该工作区下的原型、工具和素材，不在 PocoPaw 仓库内；图片已复制到 `images/p1/`。

## 0. 结论摘要

P1 两份路线图里的全部条目都已在原型中实现并测量，纯 CPU 的部分择优接入了 PocoPaw；学习式模块（C5、FFCC、PMRID）按约定只做原型。

| 条目 | 原型 | App | 关键结果 |
|---|---|---|---|
| 常驻线程池 + 核回归 NEON | 完成 | 已接入 | 超广角 8MP 核回归 1x：合并 1600 → 683 ms，总计 3.05 → 1.49 s |
| RCD / AMaZE 去马赛克 | 完成 | 已接入，默认 RCD | Kodak 8 张：MHC 29.7 dB、RCD 32.8 dB、AMaZE 33.7 dB；16MP 耗时 49 / 160 / 681 ms |
| 频域合并 | 完成 | 已接入（夜景模式） | 高噪声 16 帧：残余噪声 σ 0.00305（空间域）→ 0.00089（频域 + 空间收缩 2） |
| 亚像素多样性 / 自动 2x | 完成 | 未接入（2x 内存约 600 MB） | 静止连拍得分 0.000，手持连拍 0.750，阈值 0.3 能干净区分 |
| Pixel 风格 | 完成 | 已接入（设置“风格”） | 亮度与 ISP 成片一致（L* 54.8 对 55.6），局部对比 +25–30% |
| Ultra HDR 增益图 | 完成 | 已接入（设置开关） | 主摄最大增益 2.3–2.6×，增益图 20–45 KB，手机端约 0.3 s |
| 手持与运动 | 完成 | 已接入 | 运动掩膜膨胀 1：鬼影 0.453 → 0.374，残余噪声只增 3% |
| 包围曝光 | 完成 | 已接入（场景“包围曝光”） | 合成包围：高光 259.1 对 259.1 DN；实拍 App 端 16MP×8 帧 3.0 s |
| 夜景模式 | 完成 | 已接入（场景“夜景”） | 陀螺仪运动测光；桌面放置时选 175 ms / ISO 50；App 端 16MP×8 帧 3.0 s |
| 横向色差（LCA） | 完成 | 已接入（超广角，核回归） | 超广角中径残余色差 0.2–0.35 px → 0.0–0.1 px，合并耗时 +5% |
| 学习式 AWB（C5、FFCC） | 完成 | 不接入（约定） | 与 HAL 差距：C5 中位 2.1°（高噪声素材 10°）；FFCC 受 Cube++ 日光先验影响偏蓝 5°，去先验 2.4°；C5 手机推理 5 ms；需灰卡素材定结论 |
| PMRID 本机微调 | 完成 | 不接入（约定） | 合成训练对微调：真实高噪声帧 53.83 → 54.50 dB（估计对干净真值 58.9 → 61.6 dB） |

还缺的是**实拍验证**：运动、逆光包围、夜景、灰卡白平衡和三脚架 64 帧这几类素材都还没拍，目前的结论来自已有连拍和合成数据。拍摄清单在 [`pipeline/CAPTURE_P1.md`](../../../pipeline/CAPTURE_P1.md)，第 14 节列出每组素材要验证什么。

---

## 1. 常驻线程池与核回归 NEON

- `parallel_for` 原来每个并行区都新建线程，手机上每个阶段要多花几毫秒；改为常驻线程池（调用线程也参与，嵌套区内联执行）。
- 核回归合并的累加改为 NEON：四个 Bayer 相位放在 SIMD 通道里，核权重与流场按块缓存；LK 细化共享参考帧梯度。

| 超广角 uw8，核回归 1x | 之前 | 之后 |
|---|---|---|
| LK 细化 | 282 ms | 51 ms |
| 合并 | 1600 ms | 683 ms |
| 总计 | 3052 ms | 1494 ms |

主摄手持 hh_main8 核回归 1x 总计 5.8 → 2.9 s；超广角 2x 8.8 → 4.5 s；经典路径主摄 2.4 → 1.6 s。NEON 与标量结果的 JPEG 最大差 5 级，0.014% 像素不同（浮点求和顺序）。

## 2. 去马赛克：RCD / AMaZE

从 librtprocess（GPL-3，与本仓库许可证一致）引入 RCD 与 AMaZE，按水平条带并行。

| | MHC | RCD | AMaZE |
|---|---|---|---|
| Kodak 8 张 PSNR / 伪色 | 29.67 dB / 38.8 | 32.76 dB / 26.5 | 33.68 dB / 23.6 |
| 对 ISP JPEG（2x 缩小）PSNR / 伪色 | 37.89 / 13.8 | 37.68 / 13.3 | 38.41 / 12.1 |
| 手机 16MP 耗时 | 49 ms | 160 ms | 681 ms |

MHC 在高频纹理上有明显拉链和伪色（`pipeline/out/p1/cmp_demosaic_kodim08.jpg`）。**决定**：App 经典路径默认 RCD（多 110 ms），AMaZE 作为可选项保留在原生接口里。

![去马赛克对比](images/p1/cmp_demosaic_kodim08.jpg)

## 3. 频域合并

按 HDR+ 论文第 5 节实现成对 Wiener 合并（`merge_freq.cpp`）：16×16 半重叠正弦窗，四个 Bayer 平面放在 SIMD 通道里，可选对合并后频谱做空间收缩。

- 正确性：`c → ∞` 时与普通均值一致（σ 0.00303 对空间域 0.00305）。
- 空间收缩要四个平面共用增益，否则会出现迷宫纹。
- 高噪声 uwnoisy16（超广角 16 帧，2 ms 快门，ISO 800）：空间域 σ 0.00305；频域 s=1 0.00135；s=2 0.00089；s=4 0.00054（开始抹细节）。
- 普通光照下频域合并让文字略软，所以只在夜景模式使用（`c=64`、`s=2`）。手机耗时：超广角 8MP 合并 0.32 s，主摄 16MP 0.62 s。

![频域合并](images/p1/cmp_freq_noisy.jpg)

## 4. 亚像素多样性与自动 2x

原来的“相位覆盖率”在静止连拍上也有 0.7 以上（LK 噪声本身就会填满相位格），区分不了手持和静止。新指标 `subpixel_diversity`：

1. 由 LK 的 Hessian 最小特征值和噪声模型得到每块位移的标准误差；只用所有帧误差都小于 0.1 灰度像素的块。
2. 每块计算各帧位移的稳健离散度（中位绝对偏差），减去 3 倍估计误差后归一化，再乘以半像素相位格的覆盖比例。

| 连拍 | 得分 | 可信块比例 | 离散度（raw px） |
|---|---|---|---|
| uw8（静止） | 0.000 | 0.14 | 0.26 |
| main8（静止） | 0.000 | 0.12 | 0.42 |
| t_night（放在桌上） | 0.000 | 0.37 | 0.35 |
| hh_uw8（手持） | 0.750 | 0.09 | 7.84 |
| hh_main8（手持） | 0.750 | 0.09 | 7.36 |
| uwnoisy16（手持） | 0.750 | 0.15 | 8.89 |

`--scale auto` 在得分 ≥ 0.3 时用 2x。App 暂不接入 2x：主摄 2x 的浮点 RGB 约 600 MB。

## 5. Pixel 风格

`--look pixel`：目标亮度 0.22、压缩比 6、对比 0.26、饱和度 1.24、锐化 0.9、局部对比 0.6（1/4 分辨率快速引导滤波）、白平衡略偏冷 0.025、天空蓝饱和度 ×1.18 / 色相 −4°、肤色饱和度 ×0.92 / 色相 +3°（不用三角函数的 CbCr 窗口）。

| 1024 px 宽，L* 均值 / 局部对比 / 色度 / b* | ISP | 默认 | Pixel |
|---|---|---|---|
| main8 | 55.6 / 1.16 / 11.2 / 8.3 | 52.3 / 1.02 / 11.0 / 7.9 | 54.8 / 1.25 / 11.9 / 7.8 |
| hh_uw8 | 56.9 / 2.69 / 7.9 / 3.4 | 52.0 / 1.30 / 6.2 / 3.6 | 57.1 / 1.71 / 6.3 / 2.9 |

亮度与 ISP 一致、色调更冷、局部对比比默认高 25–30%（ISP 的锐化仍更重）。风格阶段手机耗时从最初 458 ms 优化到 209 ms。

![Pixel 风格](images/p1/cmp_look_main8.jpg)

## 6. Ultra HDR

- HDR 版本：CCM 之后的场景线性亮度 × 色调映射的长曝光增益（1.0 = SDR 白），1.0 以上软肩部压到 headroom（默认 4）。
- 增益图自己算：1/4 分辨率单通道，`(Y_hdr + 1/64) / (Y_sdr + 1/64)`，只作用于 SDR 显示亮度 0.35–0.75 以上（smoothstep），对数域 3×3 平滑，灰度 JPEG q90。
- 容器用 libultrahdr API-4（压缩底图 + 增益图 + 元数据），同时写 XMP（hdrgm）和 ISO 21496-1。最大增益低于 1.05× 时不写增益图，退回普通 JPEG（室内无高光的场景）。
- 用 `ultrahdr_app` 解码验证：main8 HDR 亮度最大 2.67、SDR 1.00，提亮只落在高光上（`pipeline/out/p1/uhdr_main8_boost.png`）。
- 最大增益：main8 2.3–2.6×、uw8 1.83×、hh_uw8 1.68×。正常曝光的连拍在 RAW 里就已经剪切，headroom 受限；−2 EV 包围能提高。
- App 端：系统 `ExifInterface` 只重写主图的 EXIF 段，增益图原样保留；实拍文件的 MPF 偏移校验正确。

![Ultra HDR 提亮区域](images/p1/uhdr_main8_boost.png)

## 7. 手持与运动

- 空间域合并改为逐帧“鲁棒性图 → 膨胀 → 累加”，与核回归共用 `MergeOpts` 和 `frame_weight`（两帧噪声相加的 z 检验）；`--motion-dilate R` 是鲁棒性图上的最小值滤波。
- 合成运动：`tools/synth_motion.py` 把一块纹理（对比度 ×1.6 / ×1.15）以每帧 40–56 px 的速度贴进真实连拍；`tools/ghost_eval.py` 在轨迹上测鬼影。

| hh_main8 空间域，高对比运动块 | 鬼影均值 | p99 | 合并权重 | 残余 σ |
|---|---|---|---|---|
| 阈值 3–6，膨胀 0 | 0.509 | 1.75 | | |
| 阈值 2–4.5，膨胀 0 | 0.453 | 1.25 | 0.971 | 0.00103 |
| **阈值 2–4.5，膨胀 1** | **0.374** | **1.0** | 0.888 | 0.00106 |
| 阈值 1.5–3，膨胀 1 | 0.320 | 1.0 | 0.675 | 0.00120 |

调高阈值鬼影反而变多；三脚架素材上所有设置的鬼影都不高于静止基线。**决定**：保持阈值 2 / 4.5，两条合并路径都默认膨胀 1。

## 8. 包围曝光

- 参考帧取最短曝光（保高光）；其余帧在黑电平以上按 `e_max / e_k` 放大到 uint16，参考帧白电平相应提高；噪声模型用与参考帧同曝光的帧估计。
- 每帧权重：z 检验用 `var_ref + (g·a·s + g²·b)`，再乘曝光的逆方差权重；该帧中达到 0.98 白电平的样本权重为 0，再做膨胀。空间域和核回归都支持；频域合并不支持曝光加权，包围时自动退回空间域。
- t_br（桌面，66 ms × 4 + 27.5 ms × 2，比 4.03）：增益 0.248，白电平 1023 → 3931；可见噪声 σ：包围空间域 0.0068、包围核回归 0.0053、以长帧为参考 0.0073。
- 合成包围（uw8 中 2/3 帧 +2 EV 并剪切）：高光均值 259.1 对 259.1 DN，相对误差 1.9%（中间调噪声 2.3%），没有洋红或发灰的高光。
- App：`BurstExposure.bracket` 每 3 帧中 2 帧 −2 EV、1 帧 0 EV（`ExposurePlan.PerFrame`）；原生会话自动选最短曝光为参考。实拍主摄 8 帧：增益 1, 1, 0.25, …，总计 3.0 s。

![包围曝光](images/p1/cmp_bracket_tbr.jpg)

## 9. 夜景模式

- 运动测光：取景时读陀螺仪 RMS 角速度，乘焦距（像素），在模糊预算 1.5 px（有 OIS 时 ×4）内取最长快门，限制在 33–333 ms（probe 为 500 ms），在测光曝光（0 EV）下换算 ISO。
- 夜景风格：目标亮度 0.11、压缩比 4、对比 0.18、饱和度 1.05、锐化 0.4、局部对比 0.15、色度降噪 ×1.6；并选择频域合并 + 空间收缩 2。
- uwnoisy16 可见噪声 σ：默认 0.0253；夜景风格 + 空间域 0.0155；+ 频域 s1 0.0064；**s2 0.0042**；s4 0.0026（抹细节）；核回归 0.0092（2.8 s）。
- App 实拍（手机放在桌上）：选了 175 ms / ISO 50，频域合并，16MP × 8 帧总计 3.0 s。

![夜景](images/p1/cmp_night_uwnoisy16.jpg)

## 10. 横向色差（LCA）

- 模型：通道 X 在 `p + (p − c)(k1 + k2 (r/rmax)²)` 处采样；核回归累加时按输出像素平移 R / B 的查询点（NEON 按相位设通道偏移）。
- 标定 `tools/lca_calib.py`：取强径向 G 边缘，对局部归一化的 R / B 与 G 做径向 LK，修剪加权最小二乘；r/rmax > 0.7 的块不足 8 个时不拟合三次项。
- 超广角（uw8 + hh_uw8 合并拟合）：kr +0.000471 / −0.000348，kb +0.000096 / +0.000372（中径 R +0.39 px、B +0.19 px，角上 B +0.96 px）。主摄中径 R +0.08、B +0.15 px，低于每块噪声（拟合 rms 0.24–0.31 px）。
- 校正后中径残余：hh_uw8 +0.25 / +0.18 → +0.01 / +0.03 px；uw8 +0.28 / +0.36 → −0.10 / −0.01；uwnoisy16（不在标定集内）+0.16 / +0.18 → +0.01 / +0.10。合并耗时 +4–5%。
- **决定**：超广角自动启用（App 中 `LensCalibration`）；主摄要等 `lca_main` 标定板素材。

![LCA](images/p1/cmp_lca_uw8.jpg)

## 11. 学习式 AWB：C5 与 FFCC

两个模型都只做原型（Python），App 仍用 HAL 的 AsShotNeutral。

- **C5**（`tools/awb_c5.py`）：官方跨相机权重（NUS + Gehler-Shi 的单反数据训练，对 Poco 的两颗传感器都是没见过的相机），m = 7：查询图加上同一摄像头的其它连拍作为无标签附加直方图。参考代码依赖已移除的 `torch.rfft`，FFT 卷积和 softmax 改在 numpy 中做，ONNX 只含编码器 / 解码器（2.0 MB）。手机端 `ort_run`：1 线程 9 ms，4 线程 5 ms。
- **FFCC**（`tools/ffcc.py`）：用 Python 重写（对数色度直方图 + 局部偏差直方图，FFT 卷积，偏置 B 即光源先验），在 SimpleCube++（Canon 550D，训练 1772 / 测试 462 张）上训练。要用到 Poco 上，先把 Cube++ 的原始 RGB 映射到 Poco 相机空间：D65 的 raw-to-raw 矩阵，再按每张图光源的相关色温（McCamy），在 colorTransform1（D65）和 colorTransform2（A）之间插值，做一次对角校正，让每个光源落在 Poco 看到它的位置。

**Cube++ 测试集**（角误差，度）：

| 模型 | 均值 | 中位数 |
|---|---|---|
| FFCC，Canon 原生空间 | 2.49 | 0.89 |
| FFCC，映射到主摄（只用 D65 矩阵） | 1.32 | 0.70 |
| FFCC，映射到主摄（按光源插值，采用） | 1.91 | 0.64 |
| 灰度世界（映射后） | 2.66 | 1.69 |
| Shades of gray（映射后） | 2.83 | 1.10 |

**Poco 素材与 HAL 估计的一致性**（还没有灰卡素材，下表不是精度，只是和 HAL 的差距；HAL 的估计都落在 Poco 白点轨迹上约 4100–4400 K，偏离 < 1°）：

| 方法 | 均值 | 中位数 | 最差 |
|---|---|---|---|
| 灰度世界 | 1.02 | 0.98 | 1.98 |
| C5 | 3.27 | 2.14 | 10.10（uwnoisy16） |
| Shades of gray | 3.80 | 2.96 | 7.48 |
| FFCC | 5.18 | 5.60 | 5.82 |
| FFCC，去掉光源先验 B | 2.38 | 2.39 | 3.66 |

- FFCC 在每段素材上都比 HAL 偏蓝约 5°。Cube++ 几乎都是户外日光，学到的先验 B 把室内暖光（约 4300 K）拉向日光；去掉 B 后差距降到 2.4°。映射本身是一致的：插值后的日光白点与 HAL 在日光下差 0.36°。
- C5 在大多数素材上与 HAL 相差约 2°；在 uwnoisy16（ISO 800、2 ms，噪声很大）上差 10°，跨相机模型对高噪声直方图不稳。
- 这些素材都是室内光、场景单一，灰度世界与 HAL 吻合得最好，并不能说明它更准。

**决定**：两个学习式模型都没有比 HAL 更可信的证据，App 不接入。等 `awb_*` 灰卡素材拍完，`awb_eval.py` 会自动改用灰卡真值（`<burst>/awb_gt.json`）；届时再判断 C5 是否值得用，以及 FFCC 是否要用 Poco 自己的数据重新学习先验 B。

## 12. PMRID 本机微调

计划用三脚架 64 帧平均作真值，`tp_*` 素材还没拍。所以先用 SimpleCube++ 的干净原始图作信号源，用 Poco 的噪声模型合成训练对（`tools/pmrid_finetune.py`）。

- **数据**：Cube++ 训练集映射到超广角相机空间（`camera_map`），按 Poco 的 RGGB 重新马赛克，随机缩放亮度。噪声按 `var = k·x + b`（DN）合成：k 在 [0.05, 12] 内取对数均匀分布（覆盖超广角 ISO 50–3200，k ≈ 0.0026–0.0035·ISO），b = 0.3 + 0.5k²。输入用 PMRID 原版的 k-sigma 变换，统一到它的锚点噪声。
- **训练**：从官方权重开始，Adam lr 1e-4（余弦衰减）、L1、batch 16 × 256²，2000 次迭代，Mac MPS 约 20 分钟。只训练 20 次的试跑在真实数据上反而变差（51.26 dB 对 53.83 dB），完整训练后才有收益。
- **合成测试**（Cube++ 测试集，同一噪声模型，PSNR dB）：

| k（DN） | 带噪 | 预训练 | 微调 |
|---|---|---|---|
| 0.3 | 43.68 | 43.68 | **45.87** |
| 1 | 38.18 | 41.20 | **42.00** |
| 3 | 33.33 | 38.73 | 38.78 |
| 8 | 29.36 | **36.45** | 36.22 |

- **真实测试**（uwnoisy16，ISO 800 / 2 ms，噪声估计 k = 2.78、b = 1.17）：以 16 帧合并（有效帧数 14.3）为伪真值评估第 0 帧。伪真值本身也有残余噪声，"估计"一列按 `var_gt = mean(k·ref + b) / neff` 扣掉这部分，换算成相对干净真值的 PSNR。

| | PSNR | 估计（对干净真值） |
|---|---|---|
| 带噪第 0 帧 | 43.16 | 43.43 |
| 预训练 | 53.83 | 58.93 |
| **微调** | **54.50** | **61.62** |

- 低噪声端提升最大：预训练模型在 k = 0.3 时几乎不起作用，与输入相同。高噪声端（k = 8）略差 0.2 dB，说明训练分布在高 ISO 端偏少。成片上，微调版暗部的大块色斑和"油画感"比预训练少，平坦区域更干净；与 16 帧合并相比仍有涂抹。

![PMRID](images/p1/cmp_pmrid_uwnoisy16.jpg)

**决定**：微调权重 `pipeline/calib/pmrid_poco_uw.ckp` 优于官方权重，`pmrid_denoise.py --weights` 可直接使用。按约定只留在原型，App 不引入 ONNX Runtime。拍到 `tp_*` 后，用真实 64 帧真值复核，并补主摄的权重。

## 13. PocoPaw 集成

**移植方式**：原型的 `align`、`merge`、`merge_freq`、`kr`、`demosaic`、`finish`、`ultrahdr` 与 App 中的文件**逐字节相同**。差异只在：

- `common.h`：App 的 `Raw` 是相机缓冲区的只读视图；包围曝光需要改写样本时 `make_writable()` 先复制一份。
- `burst.*`、JNI、`pool.cpp`：会话接口与内存池。
- 第三方：librtprocess（源码）、libultrahdr + libjpeg-turbo（arm64 预编译静态库）、stb_image_write。

`pipeline/tools/session_check.cpp` 用 App 的源码处理 probe 连拍。与原型命令行对比，5 种模式的 JPEG 最大差 ≤ 1 级（抖动），即结果一致：经典 RCD、包围曝光、夜景频域合并、核回归 + LCA + Pixel 风格、Ultra HDR。

**设置与模式**：

| 设置 | 选项 | 作用 |
|---|---|---|
| 场景 | 自动 / 包围曝光 HDR / 夜景 | 曝光规划（`BurstExposure.plan / bracket / night`）；夜景还会切到频域合并和夜景风格 |
| 风格 | 标准 / Pixel 风格 | `LookParams` 预设 |
| Ultra HDR | 开 / 关 | 输出带增益图的 JPEG |
| 处理方式 | HDR+ 合并 / 核回归 | 经典路径默认 RCD；核回归在超广角上自动校正 LCA |

**手机端耗时**（`session_check-android`，8 线程，同一套原生代码）：

| 素材 | 模式 | 总计 |
|---|---|---|
| main8（16MP × 8） | 经典 RCD | 1.46 s |
| main8 | 经典 RCD + Ultra HDR | 1.75 s |
| main8 | 经典 MHC | 1.37 s |
| t_br（8MP × 6，包围） | 经典 RCD | 0.69 s |
| uwnoisy16（8MP × 16） | 频域 + 夜景风格 | 1.56 s |
| uw8（8MP × 8） | 核回归 + LCA + Pixel 风格 | 1.40 s |

**App 实拍**（主摄 16MP × 8 帧，含快速预览、DNG 写出和 JPEG 写出）：

| 设置 | 管线日志 | 总计 |
|---|---|---|
| 自动，Pixel 风格，Ultra HDR | classic/spatial/rcd，最大增益 1.50×，增益图 22 KB | 2.82 s |
| 包围曝光，Pixel 风格，Ultra HDR | 增益 1, 1, 0.25…，最大增益 1.64× | 2.97 s |
| 夜景，Ultra HDR | classic/freq/rcd，175 ms / ISO 50 | 2.96 s |
| 核回归，16 帧，标准风格 | kr | 6.37 s |

没有接入 App 的：核回归 2x 与自动 2x（内存）、AMaZE 的设置项（接口已支持）、学习式模块（C5、FFCC、PMRID，按约定只做原型）。

## 14. 待实拍验证

按 [`pipeline/CAPTURE_P1.md`](../../../pipeline/CAPTURE_P1.md) 拍摄后需要确认：

| 素材 | 验证内容 |
|---|---|
| `mo_main_wave`、`mo_main_walk`、`mo_uw_street` | 真实运动下阈值 2 / 4.5 + 膨胀 1 是否足够，核回归与空间域的鬼影对比 |
| `br_main_backlit`、`br_uw_backlit` | 真逆光下高光保留与暗部噪声，Ultra HDR headroom 是否提高 |
| `nt_main_hand`、`nt_main_brace`、`nt_uw_hand` | 手持时运动测光的快门与模糊，频域 s2 在真实夜景的细节 |
| `awb_*` | 灰卡真值下 HAL、灰度世界、C5、FFCC 的角误差 |
| `tp_main_iso*` | 64 帧平均真值：PMRID 原版与微调版的真实 PSNR；按 ISO 的噪声标定 |
| `lca_main` | 主摄 LCA 是否值得校正 |

## 15. 许可证

| 组件 | 许可证 | 位置 |
|---|---|---|
| librtprocess（RCD、AMaZE） | GPL-3.0 | 原型与 App |
| libultrahdr、libjpeg-turbo | Apache-2.0 / MIT；IJG / BSD-3 / zlib | 原型与 App（静态库） |
| hdr-plus-swift | GPL-3.0 | 只作频域合并参考，未复制代码 |
| C5（官方权重） | Apache-2.0 | 仅原型 `pipeline/tools/third_party/C5` |
| PMRID（官方权重） | Apache-2.0 | 仅原型 |
| SimpleCube++ 数据集 | CC BY 4.0 | 仅原型训练（不随仓库分发） |

## 16. 复现

```bash
cd ~/Documents/Poco-F5-RAW-Report/pipeline
./build.sh                                               # burstpipe-mac / burstpipe-android
bin/burstpipe-mac data/main8 --demosaic rcd --look pixel --ultrahdr -o out.jpg
bin/burstpipe-mac data/t_br --demosaic rcd               # 包围：自动选最短曝光为参考
bin/burstpipe-mac data/uwnoisy16 --look night            # 夜景：频域合并 s2
bin/burstpipe-mac data/uw8 --kr --lca auto --scale auto  # 核回归 + LCA + 自动 2x
python3 tools/awb_eval.py data/main8 data/uw8 ...        # AWB 对比
python3 tools/ffcc.py train-cube /path/SimpleCube++ calib/ffcc_cam0.npz --to-camera data/main8
python3 tools/pmrid_finetune.py train /path/SimpleCube++ calib/pmrid_poco_uw.ckp --camera data/uw8
python3 tools/pmrid_finetune.py eval-real data/uwnoisy16 /tmp/PMRID/models/torch_pretrained.ckp calib/pmrid_poco_uw.ckp --merged <dir>
```
