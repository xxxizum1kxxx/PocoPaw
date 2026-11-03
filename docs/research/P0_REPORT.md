# P0 工程化与噪声：原型实现、实测与 PocoPaw 集成

- 日期：2026-11-03
- 设备：Redmi Note 12 Turbo / Poco F5（marble），Snapdragon 7+ Gen 2，Adreno 725；耗时均为手机实测（`session_check-android` 或 App logcat），另注明的除外
- 前置报告：[`PIPELINE_REPORT.md`](PIPELINE_REPORT.md) 第 8 节 P0 表、[`ALGORITHM_UPGRADE.md`](ALGORITHM_UPGRADE.md) 第 6 节第 1–2 条、[`P1_REPORT.md`](P1_REPORT.md) 第 14 节（实拍素材）
- 原型：[`pipeline/`](../../../pipeline/)（`burstpipe`、`session_check`、`pipeline/tools/`）；逐项原始记录在 [`pipeline/out/p1/RESULTS.md`](../../../pipeline/out/p1/RESULTS.md) 的 `p0` 各节
- App：[`app/src/main/cpp/burstpipe/`](../../app/src/main/cpp/burstpipe/)（原生管线）与 [`app/src/main/java/com/izum1k/pocopaw/pipeline/`](../../app/src/main/java/com/izum1k/pocopaw/pipeline/)（阶段、边拍边合并、合成 DNG），`dev-exp` 分支每项一个提交（见各节）

> 本文复制自研究工作区 `Poco-F5-RAW-Report/`。文中的 `pipeline/`、`data/`、`bin/` 都指该工作区下的原型和素材，不在 PocoPaw 仓库内。

## 0. 结论摘要

两份路线图的 P0 全部完成：工程化 7 项已在原型实现并接入 PocoPaw，GPU 与 CPU 输出的差距都在 70 dB 以上；噪声标定已接入 App；PMRID 门控按约定只做原型。

| 条目 | App | 关键结果 |
|---|---|---|
| 噪声标定 | 已接入（主摄） | HAL 不报 `SENSOR_NOISE_PROFILE`，主摄改用标定曲线（ISO 50–876）；鲁棒性阈值改为 1.75 / 3.9，手持包围权重 .911 → .934 |
| 参考帧选择 | 已接入 | 前 3 帧取梯度能量最高者；手持 `hh_main8` 选中第 1 帧，成片梯度能量 +3%；DNG 写实际参考帧 |
| 噪声自适应 | 已接入 | 锐化阈值随可见噪声，高噪声限制目标亮度；ISO 876 平坦区噪声 10.8 → 8.7（8 bit），低 ISO 不变 |
| 低分辨率色调映射 | 已接入 | 16MP 色调映射 CPU 260 → 95 ms、GPU 138 → 72 ms；与全分辨率相比 52.9–63.6 dB |
| NEON | 已接入 | 主摄 8 帧 CPU 对齐 454 → 347 ms、合并 349 → 234 ms；与标量版 JPEG 最大差 1–2 |
| libjpeg-turbo | 已接入 | 16MP 编码 26–28 ms（stb 132 ms，`Bitmap.compress` 147–217 ms）；App JPEG 阶段 200 → 69 ms |
| 边拍边合并 | 已接入（默认） | 16 帧从对焦到 JPEG 3.19 → 2.29 s，峰值内存 +1064 → +181 MB，内存与帧数无关；32 帧 1.76 s 合并完 |
| 合成 DNG | 已接入（默认） | 16 位马赛克 + GainMap + 双光源颜色矩阵；与原型逐像素相同，GPU 版 139 dB；Mac 预览 / rawpy 正确解码 |
| PMRID 门控 | 仅原型（约定） | 64 帧真值上 MSE 最优混合系数 0.89–0.97；K-Sigma 域保留负值反而掉分（ISO 876 单帧 −7 dB）；**偏粉来自后处理**，不是 PMRID 的截断 |

App 内整体（主摄 16MP，GPU，标定噪声，室内静态，从对焦锁定到 JPEG 写完）：4 帧 1.41 s、8 帧 1.68 s、16 帧 2.29 s；同一版本关掉边拍边合并（批处理）为 1.51 / 2.03 / 3.19 s。只看管线（不含采集），8 帧从 P0 之前的 1501 ms（`PIPELINE_REPORT.md` 第 7.5 节）降到 916 ms。

---

## 1. 噪声标定接入 App（提交 `b4bdfa7`）

- **来源**：本机 HAL 不报 `SENSOR_NOISE_PROFILE`，无法与标定表对比，按计划的备选：主摄用三脚架标定曲线 `calib/noise_curve_cam0.json`（`tools/noise_calib.py curve`，ISO 50/100/400/800/876，a 0.06–1.25、b 0.30–5.36 DN，按 ISO 插值），超广角继续帧间估计，补拍 `dk_uw`、`tp_uw_*` 已加进 [`pipeline/CAPTURE_P1.md`](../../../pipeline/CAPTURE_P1.md)。包围曝光的参考帧放大了 f 倍，模型随之变为 a' = f·a、b' = f²·b。
- **鲁棒性**：P1 发现直接换标定值后手持包围权重从 .91 掉到 .47。原因是帧间估计偏低 10–25%，并吸收了对齐误差，阈值是按它调的。三个候选里选了 (c)：标定模型单独一组阈值 1.75 / 3.9（`kRobustCalibrated`），估计模型保持 2 / 4.5。
- **验证**：
  - 鬼影（`ghost_eval`，越低越好）：标定 / 估计 = 空间域 0.373 / 0.374，核回归 0.389 / 0.393。
  - 64 帧真值 PSNR：ISO 800 54.07 / 54.07，ISO 876 59.47 / 59.40，ISO 100 56.21 / 56.32。
  - `br_main` 权重 .934、可见噪声 0.0047（估计 .911 / 0.0051）；`nt_main_hand` 频域可见噪声 0.0032（0.0047）。
  - GPU 对 CPU：`main8` 95.06 dB，包围 93.15 dB。App 实拍 ISO 788：`var 0.1445 S + 0.591 calibrated robust 1.75-3.90`。

## 2. 参考帧选择（提交 `04deccf`）

- `align.cpp` 的 `sharpest_frame`：前 3 帧中心区域 4×4 合并后的梯度能量，后一帧要锐利 3% 以上才替换；包围曝光仍用最短曝光帧。
- 手持组 `hh_main8` 三帧锐度 866 / 1018 / 1040，选第 1 帧（第 2 帧差距不到 3%）；三脚架和支撑组差距在 0.5% 以内，保持第 0 帧。
- 成片梯度能量（越高越清晰）：`hh_main8` 5.47 对固定第 0 帧 5.30（+3%），`syn_hhmain` 7.55 对 7.47；合并权重 .896 对 .891。
- 耗时每帧 1–3 ms；App 的参考帧阶段 7 ms（4 帧）。App 的 `RawBurst.referenceIndex` 由原生层回传，参考帧 DNG 写的是实际选中的帧。

## 3. 噪声自适应（提交 `70a04f8`）

- 锐化阈值 `core = max(0.004, 0.6·visibleSigma)`；目标亮度在可见噪声超过 0.015 时乘以 √(0.015 / visible)，最低 0.5 倍。CPU 与 GPU 的 finish 同步。
- JPEG 平坦区噪声（8 bit，32 px 块高通标准差的第 20 百分位）与平均亮度，修改前 → 后：

| 素材 | 平坦区噪声 | 平均亮度 |
|---|---|---|
| `uwnoisy16` | 7.48 → 5.97 | 123.5 → 108.1 |
| `t_dark` | 7.22 → 5.89 | 122.8 → 110.3 |
| `tp_main_iso3200`（ISO 876） | 10.81 → 8.71 | 112.7 → 100.7 |
| `tp_main_iso800` | 4.55 → 3.96 | 110.4 → 108.9 |
| `main8` / `uw8` / `hh_main8` / `t_br` | 不变 | 不变 |

## 4. 低分辨率色调映射（提交 `9e773dd`）

- Mertens 融合改在 4×4 合并的亮度上做，融合增益（log2）对亮度（log2）用快速引导滤波（r 2，eps 0.02）上采样回全分辨率；`--tone-full` 保留旧路径对照。
- 与全分辨率版 JPEG 的 PSNR：`main8` 62.98、`uw8` 63.57、`t_br` 60.52、`br_main_backlit` 52.93、`br_uw_backlit` 54.76、夜景 58.72 dB。差异集中在灯泡和画框边缘（`br_main_backlit` 第 99.9 百分位 6 级），全分辨率裁切上没有光晕。
- 16MP 耗时：CPU 260 → 93–97 ms（log2 / exp2 多项式近似、按行插值系数、不再分配全分辨率 Y），GPU 138 → 72 ms。App 实拍 GPU 色调映射 62 ms，后处理 703 → 632 ms。
- 近似函数让 GPU 对 CPU 从 89–96 dB 降到 70–81 dB，仍在 50 dB 要求以上。

## 5. NEON（提交 `676ed4e`）

- `align.cpp` 的 `tile_cost`、`merge.cpp` 的块读取 / 鲁棒性（4 块并行）/ 累加改为 NEON；`min_filter` 改成可自动向量化；`frame_weight` 改用 float（与 GPU 一致）。`-DBURSTPIPE_NO_NEON` 保留标量版本供对拍。
- 手机 CPU，标量 → NEON（ms）：

| 素材 | 对齐 | 合并 |
|---|---|---|
| `main8` | 454 → 347 | 349 → 234 |
| `uwnoisy16` | 609 → 518 | 392 → 275 |
| `uw8` | 250 → 213 | 187 → 132 |
| `t_br`（包围，剪切检查保留标量第 1 遍） | 203 → 184 | 179 → 176 |

- NEON 对标量 JPEG：`main8` / `t_br` 最大差 1，`uw8` / `uwnoisy16` 最大差 2（浮点求和顺序），差值超过 8 的样本为 0。
- RCD：随仓库的 librtprocess `rcd.cc` 没有 `vfloat` / SSE2 分支（只有 `amaze.cc` 有），所以没有可映射的 SIMD 路径，保持 `-O3 -ffast-math` 自动向量化。

## 6. libjpeg-turbo（提交 `631c3f3`）

- 原型与 App 都直接调用 libjpeg-turbo 3.1.0（来自 libultrahdr 的构建）：按 MCU 行切成条带并行编码，每条是独立的 baseline JPEG，用 DRI + RSTn 拼接；Ultra HDR 的基图和增益图走同一个编码器。`encode_jpeg_stb` 保留作对照。
- 手机 16MP q95：4:2:0 条带并行 26–28 ms（2.12 MB），单条 55 ms，4:4:4 33–36 ms；stb 132 ms；App 原来的 `Bitmap.compress` 147–217 ms。
- 拼接文件通过 `djpeg -strict`，解码结果与单条编码逐像素相同（restart marker 只重置 DC 预测）。
- App 的 JPEG 阶段（编码 + EXIF + MediaStore）195–209 → 69 ms，4 帧总计 1135 → 1000 ms。

## 7. 边拍边合并（提交 `e2ce88c`）

- **原生层**：`merge.cpp` 的 `SpatialMerge` 把 `merge_frames` 拆成逐帧接口（批处理也改用它，输出不变）；`BurstSession` 增加 `begin_stream` / `add_frame` / `end_frames`：先缓存前 3 帧选参考帧，之后每来一帧就对齐、算鲁棒性、累加，然后释放。GPU 端 `GpuBurst::create_stream` 只开参考帧和当前帧两个槽，复用原有 kernel；GPU 失败时从参考帧重新开始走 CPU。
- **App**：`BurstStream` 在单独的串行线程上接收帧；`CameraController` 的 `SteadyGate` 在线剔除失焦或移动中的帧（前 3 帧的中位对焦距离作基准）；参考帧确定后其余帧立即归还相机。只用于单一曝光的空间域合并，包围、夜景（频域）和核回归仍走批处理。
- **一致性**：流式对批处理 JPEG 最大差 0（`main8`、`uw8`，CPU）；GPU 流式对 CPU 77.5–80.3 dB（含 2x 变焦、32 帧、标定噪声）。
- **耗时**：主摄 16MP 每帧 CPU 72–82 ms、GPU 70–76 ms，帧间隔 66 ms，所以最后一帧之后还积压 2–3 帧；32 帧 CPU 1889 ms、GPU 1758 ms 合并完。
- **App A/B**（主摄，GPU，标定噪声，从对焦锁定到 JPEG，进程 RSS 峰值相对空闲，每 50 ms 采样）：

| 帧数 | 批处理 | 边拍边合并 |
|---|---|---|
| 4 | 1.51 s，+305 MB | 1.41 s，+161 MB |
| 8 | 2.03 s，+595 MB | 1.68 s，+212 MB |
| 16 | 3.19 s，+1064 MB | 2.29 s，+181 MB |

## 8. 合成 DNG（提交 `5d1ad29`）

- **写入器**：`pipeline/src/dng.cpp` 的 `encode_dng`，原型（`burstpipe --dng`）与 App 共用同一份 C++ 代码，App 侧的 Kotlin `MergedDngWriter` 只负责从 Camera2 收集标签。计划里写的是纯 Kotlin 写入器，改成共用 C++ 的原因：两边输出可以直接逐字节对拍，GPU 路径的马赛克也不用回传到 Java 堆。
- **内容**：
  - 无压缩 16 位 CFA，DNG 1.4。数值 ×16（白电平 ×16 超过 65535 时取更小的倍数），保留合并带来的 4 位小数精度，黑白电平同步放大。
  - 镜头阴影写成 4 个 GainMap opcode（OpcodeList2，标记为可选），2x 变焦时按裁切区域定位。
  - ColorMatrix / ForwardMatrix / CalibrationIlluminant 1–2 取自 `CameraCharacteristics`，AsShotNeutral 取自参考帧的 `CaptureResult`；方向、厂商、型号和 256 px 缩略图沿用原来的写法。
  - 包围曝光合并后白电平扩大了 f 倍，写 BaselineExposure = log2 f 抵消，否则渲染器会暗约 2 档（`t_br`：Mac 预览平均亮度 41 → 104）。
- **文件布局**：macOS ImageIO 拒绝“像素条带在 IFD 之前且带 OpcodeList2”的文件，改为 IFD 在前、数据在后（与 `DngCreator` 相同）后正常解码。
- **App**：
  - 设置新增“DNG 内容：合成 / 参考帧”，默认合成；合并完成后由 `MergedDngStage` 写出。
  - 核回归输出 RGB，没有马赛克，仍在合并前写参考帧 DNG。
- **验证**：
  - 一致性：`main8`（标定噪声）上，原型 `--dng` 与 App 会话（批处理和流式）的原始数据最大差 0；手机 CPU 版也完全相同，GPU 版和 GPU 流式最大差 1/65535（139 dB）。`t_br` 原型对会话最大差 0。
  - rawpy 读取：黑电平 1024、白电平 16368，CFA 排列与相机白平衡正确；|dng/16 − merged.f32| ≤ 0.031 DN。
  - 解码：macOS 预览（`sips`）能正确解码 `main8`、2x 变焦（2312×1736）和 `t_br`，颜色正确。
  - 体积与耗时：文件 32.1 MB；编码在 Mac 上 12 ms，手机 CPU 22 ms，GPU 路径含下载 112 ms。
  - App 实拍（4 帧流式，DNG + JPEG）：`merged_dng` 阶段 123 ms（编码 + 写入 32.3 MB），总计 1211 ms；92% 的采样用到了 4 位小数；厂商、型号、方向、光源 21 / 17 正确。

## 9. PMRID 门控与混合（仅原型）

工具：`tools/pmrid_gate.py`（主摄三脚架 `tp_main_iso100/800/3200`，完整表格在 `pipeline/out/p1/pmrid_gate.txt`）。输入为第 0 帧或前 n 帧（n = 2/4/8/16）的空间域合并（标定噪声），PMRID 用官方权重、按 k/neff、b/neff 降噪。真值从 64 帧对齐平均里**扣掉输入用到的帧**，保证真值的残余噪声与输入无关，否则 8 帧合并与真值共享 1/7 的噪声，会让混合系数偏向保留输入。

**截断位置**（对干净真值估计的 PSNR，dB）：

| 输入下界（K-Sigma 域） | ISO 876 单帧 | ISO 800 单帧 | ISO 876 8 帧 |
|---|---|---|---|
| 线性 0 DN 的映射（原做法，默认） | **64.25** | **55.02** | **64.92** |
| 0 | 64.20 | 54.92 | 64.90 |
| −1 | 63.40 | 54.90 | 64.90 |
| −2 | 58.68 | 54.89 | 64.90 |
| −S/K（方差为零处，负值全保留） | 57.23 | 54.88 | 64.90 |

官方模型是用截断后的输入训练的，下界放低就超出了它的分布，高噪声单帧最多掉 7 dB。所以默认保持“线性 0 DN 的映射”这个下界（`to_ksigma(floor=None)`），`--ks-floor` 留作实验。

**偏粉**：计划的假设是“线性域截断让暗部均值上偏”。实测不成立：

- PMRID 输出在暗部（真值 G 第 10–30 百分位）的偏色（白平衡后 R−G、B−G 误差）：各合并输入都在 ±0.07 DN 以内；高噪声单帧略偏绿（ISO 800 −0.11 / −0.49，ISO 876 −0.33 / −0.39 DN），没有偏粉。
- `uwnoisy16` 第 0 帧对它自己的 16 帧合并：默认 −0.35 / −0.16，−S/K −0.50 / −0.25。若在 K-Sigma 域截到 0 才会偏粉（+5.1 / +3.0 DN）。
- 偏粉出现在**后处理**：成片暗部（最暗 15%）的 R−G / B−G（8 bit），`uwnoisy16` 单帧 +26.6 / +17.2，16 帧合并 +7.3 / +2.6，PMRID 单帧 +0.9 / +1.5。原因是 `finish.cpp` 的归一化在乘白平衡增益（R 1.85、B 1.64）之前把每个通道截到 0，高噪声暗部的均值被抬高；只去掉这一处截断，单帧降到 +19.5 / +12.8，去马赛克和 CCM 之后还有截断。这部分涉及 App 后处理和 GPU kernel，**本次未改**，列为后续工作（第 10 节）。

**混合曲线**：MSE 最优 α* = ⟨p − x, t − x⟩ / ‖p − x‖²。

| k_eff（DN） | ISO 800 | ISO 876 |
|---|---|---|
| 约 1（单帧） | 0.975 | 0.969 |
| 0.5–0.6（2 帧） | 0.965 | 0.958 |
| 0.26–0.32（4 帧） | 0.945 | 0.951 |
| 0.13–0.16（8 帧） | 0.911 | 0.929 |
| 0.07–0.08（16 帧） | 0.897 | 0.891 |

- ISO 100（k_eff 0.12 → 0.018）各档都是 1.0；混合带来的提升最多 0.11 dB，说明在这个范围内 PMRID 基本应全量使用。
- `pmrid_denoise.ALPHA_KNOTS` 设为 (log10 k_eff, α) = (−1.1, 0.90)、(−0.5, 0.95)、(0, 0.97)，区间外取端点值。
- PMRID 对输入的提升：ISO 876 单帧 50.47 → 64.25 dB、8 帧 59.76 → 64.92 dB、16 帧 61.92 → 65.15 dB；ISO 100 8 帧（k_eff 0.018）52.54 → 52.97 dB，增益很小但仍为正。
- ALGORITHM_UPGRADE 第 3.3 节在手持超广角（k_eff 0.0325）上看到的纹理涂抹，在三脚架真值的 MSE 上没有体现。低噪声端的门控需要手持真值（`mo_*`、`tp_uw_*`）或感知指标来定。

## 10. 后续

- **后处理偏粉**：归一化、去马赛克、CCM 之后都截掉了负值。可以改为保留负值到色度降噪之后再截断，或在归一化时按噪声模型做均值校正。需要同步 GPU kernel，并重新对拍。
- 补拍：`dk_uw`、`tp_uw_iso100/800`（超广角标定与 PMRID 低噪声端），`mo_*`（运动组，复核两组鲁棒性阈值）。见 `pipeline/CAPTURE_P1.md`。
- 边拍边合并目前只覆盖单一曝光的空间域合并。夜景的频域合并逐帧累加在数学上可行（每帧独立的维纳收缩），可作为下一步。
