# Redmi Note 12 Turbo / Poco F5（marble）Camera2 RAW 性能边界报告

- 测试日期：2026-11-01
- 设备：23049RAD8C（marble），LineageOS 24.0-20261027-NIGHTLY，Android 17
- 相机 HAL：Qualcomm CamX + 小米 `camera.xiaomi.so` 扩展，HIDL `device@3.7`
- 测试对象：主摄（Camera ID 0）、超广角（ID 2）、前置（ID 1）。微距（ID 3）不在本报告范围内。
- 测试工具：自制 Camera2 探针 App，源码见研究工作区的 `probe/`

> 本文复制自研究工作区 `Poco-F5-RAW-Report/`。文中的 `pipeline/`、`probe/`、`data/`、`bin/` 都指该工作区下的原型、采集工具和素材，不在 PocoPaw 仓库内。PocoPaw 中的对应实现在 [`app/src/main/cpp/burstpipe/`](../../app/src/main/cpp/burstpipe/)。

## PocoPaw 实现情况（2026-11-01）

本报告第 6 节的 App 设计建议在 PocoPaw 中的落实情况，已逐条标在第 6 节。概括如下：

- 预览只出 YUV，按快门才发 RAW 连拍；RAW 额度（65）不足时先重开设备：**[App 已实现]**（`CameraController`、`RawRequestBudget`）。
- 测光用 YUV 预览，RAW 用手动曝光，并换算成低 ISO + 长快门：**[App 已实现]**（`BurstExposure`）。
- 逻辑摄像头 4、5、6 不打开：**[App 已实现]**（`CameraCatalog.DENYLIST`）。
- 从 `ERROR_CAMERA_DEVICE` 恢复：**[App 已实现]**（关闭后重开，最多 3 次）。
- 连拍先放内存再处理：**[App 已实现]**。RAW 拷贝放在原生内存中，避免 256 MB 的 Java 堆上限。
- 长时间分段采集、RAW10：**[未实现]**（当前没有需要它们的功能）。

## 1. 结论摘要

1. 三颗摄像头都能通过 Camera2 拿到**全尺寸 RAW（`RAW_SENSOR` / `RAW10`）**，**30 fps 满帧率连拍**，帧间隔恒定 33.3 ms，零丢帧，没有 stall。
2. **HAL 有一个硬性上限：每次打开 `CameraDevice`，最多能发 65 个带 RAW 输出的请求。** 第 66 个请求进入 HAL 后，设备会报 `ERROR_CAMERA_DEVICE`（error=4），这时 App 只收到 58 帧（管线深度是 8）。三颗摄像头的上限完全相同。
3. 这个上限按**请求数**算，跟时间、格式、模板、是否带 YUV 流都无关。**重建会话不能清零，只有关闭再重新打开 `CameraDevice` 才能清零。**
4. 按“每段 65 帧，段间重开设备”的方式，可以长时间连续拿 RAW，平均约 **25 fps**，但每段之间有约 0.5 到 0.6 秒的断档。
5. 自动曝光和自动白平衡在带 RAW 流时都正常工作，但快门会被 50 Hz 防频闪锁在 1/100 s，AE 只靠调 ISO。做 HDR+ 式的欠曝连拍时，更好的办法是先让 AE 收敛，再换算成“低 ISO + 更长快门”的手动曝光（见 4.2 节，本条已复核更正）。
6. 拿不到传感器原生全分辨率：主摄（64MP）和前置（16MP）只输出四合一像素合并后的尺寸，HAL 没有声明 `MAXIMUM_RESOLUTION` 模式。

## 2. 各摄像头 RAW 规格

| 项目 | 主摄（ID 0） | 超广角（ID 2） | 前置（ID 1） |
|---|---|---|---|
| 传感器 | OmniVision OV64B40（64MP） | Sony IMX355（8MP） | Samsung S5K3P9SP（16MP） |
| RAW 输出尺寸 | 4624×3472（像素合并） | 3280×2464（原生） | 2304×1728（像素合并） |
| 拜耳排列 | BGGR | BGGR | GBRG |
| 有效位深 / 白电平 / 黑电平 | 10-bit / 1023 / 64 | 10-bit / 1023 / 64 | 10-bit / 1023 / 64 |
| 焦距 / 光圈 | 4.68 mm / f1.8 | 1.65 mm / f2.2 | 3.26 mm / f2.4 |
| 光学防抖 | 有 | 无 | 无 |
| 曝光时间范围 | 73 µs 到 1 s | 12.8 µs 到 1 s | 38.6 µs 到 843 ms |
| ISO 范围（模拟增益上限） | 50 到 877（796） | 50 到 4000（800） | 50 到 800（800） |
| 每帧大小（RAW16 / RAW10） | 32.1 MB / 20.1 MB | 16.2 MB / 10.1 MB | 8.0 MB / 5.0 MB |
| 30 fps 数据率（RAW16） | 约 963 MB/s | 约 485 MB/s | 约 239 MB/s |

三颗摄像头声明的共同属性：

- 硬件级别 `LEVEL_3`，能力包含 `RAW`、`MANUAL_SENSOR`、`MANUAL_POST_PROCESSING`、`READ_SENSOR_SETTINGS`、`BURST_CAPTURE`。
- RAW 最短帧间隔约 33.3 ms（30 fps），stall 时长为 0。
- `pipelineMaxDepth` = 8；`sync.maxLatency` = `PER_FRAME_CONTROL`（声明支持逐帧改参数）。
- 时间戳源为 `REALTIME`，可以和 `SystemClock.elapsedRealtimeNanos()` 直接比较。
- 每颗摄像头只提供一种 RAW 尺寸，没有降分辨率换高帧率的 RAW 模式。
- DNG 所需的色彩矩阵、前向矩阵、标定矩阵、D65 参考光源、镜头阴影图都有。缺 `android.sensor.noiseProfile`。

## 3. 实测性能

### 3.1 单次连拍（一次 `captureBurst`，未超过上限）

| 指标 | 主摄 | 超广角 | 前置 |
|---|---|---|---|
| 帧率（自动曝光，30 帧） | 30.07 fps | 29.70 fps | 30.03 fps |
| 帧间隔（中位数 / 最大值） | 33.3 / 33.3 ms | 33.3 / 43.3 ms | 33.3 / 33.3 ms |
| 单次最多帧数 | 65 帧（2.48 s） | 65 帧（2.45 s） | 65 帧（2.47 s） |
| 从发出请求到第一帧 | 343 到 369 ms | 302 到 321 ms | 290 到 331 ms |
| 从曝光到 App 收到帧（中位数） | 约 45 ms | 约 44 ms | 约 36 ms |
| 50 帧 + 每帧拷进 Java 内存 | 30.07 fps，零丢帧 | 29.82 fps，零丢帧 | 30.03 fps，零丢帧 |
| 50 帧 + 每帧同步写盘 | 30.07 fps，零丢帧 | 29.82 fps，零丢帧 | 30.03 fps，零丢帧 |

超广角偶尔会出现一次 43 ms 的帧间隔（在曝光收敛阶段），不算丢帧。

`RAW10` 的帧率和 `RAW_SENSOR` 完全相同，只是体积小 37.5%。

主摄 50 帧写盘共 1.4 GB，在约 2 秒内完成，大部分被页缓存吸收了。这个结果不代表闪存能持续写 700 MB/s 以上；更长的连拍需要另外测持续写入速度。

### 3.2 长曝光

手动曝光下，帧间隔严格等于曝光时间，两次之间没有额外开销：

| 摄像头 | 曝光时间 | 实测帧率 | 帧间隔 |
|---|---|---|---|
| 主摄 | 100 ms | 9.98 fps | 100.2 ms |
| 主摄 | 500 ms | 2.00 fps | 500.2 ms |
| 超广角 | 100 ms | 9.99 fps | 100.1 ms |
| 前置 | 50 ms | 19.98 fps | 50.0 ms |

### 3.3 长时间连续 RAW（分段，段间重开设备）

每段发 65 个请求，段间关闭再重开 `CameraDevice`，共 5 段 325 帧：

| 指标 | 主摄 | 超广角 | 前置 |
|---|---|---|---|
| 收到帧数 | 325 / 325 | 325 / 325 | 325 / 325 |
| 平均帧率 | 25.00 fps | 25.64 fps | 25.41 fps |
| 重开设备耗时 | 160 到 206 ms | 113 到 141 ms | 121 到 168 ms |
| 段间断档（最大帧间隔） | 约 583 到 614 ms | 约 503 ms | 约 524 到 574 ms |
| 段内帧率 | 30 fps | 30 fps | 30 fps |

## 4. HAL 限制详情

### 4.1 每次打开设备最多 65 个 RAW 请求

**现象**

- 单次 `captureBurst` 发 65 帧成功，发 66 帧失败。三颗摄像头、RAW16 和 RAW10 结果都一样。
- 用 repeating 请求连续出 RAW，约 2.2 秒、交付 58 帧后失败。
- 失败时 App 收到 `CameraDevice.StateCallback.onError(ERROR_CAMERA_DEVICE)`，此后这个设备不能再用，必须关闭重开。HAL 进程本身不会崩。

**HAL 日志**

```
CHIUSECASE: chitargetbuffermanager.cpp:441 SetupTargetBuffer()
  [PortTargetBuffer_RealTime:Raw_Out_RealTime_0] Max number of nodes reached 64!
  m_pProducerList size=7, m_pConsumerList size=58
CHIUSECASE: chifeature2base.cpp:415 GetOutputBufferHandle() Returning null buffer handle for port Raw_Callback
```

高通 CHI 层 `Raw_Out_RealTime_0` 端口的 target buffer 节点，每交付一帧 RAW 就漏掉一个，从不回收，上限是 64。

**排除测试**

| 假设 | 测试 | 结果 |
|---|---|---|
| 与时间有关 | 前置用 50 ms 曝光发 66 帧（耗时 3.2 s） | 照样失败，说明按请求数算 |
| App 没有及时释放缓冲 | 检查 `onCaptureBufferLost` 和丢帧数 | 都是 0，App 释放正常 |
| 只出 RAW 不出 YUV 导致 | 加一路 1080p YUV 流 | 照样失败 |
| 和请求模板有关 | 分别用 PREVIEW、STILL_CAPTURE、ZERO_SHUTTER_LAG | 都失败 |
| 和格式有关 | 分别用 RAW_SENSOR、RAW10 | 都失败 |
| 只要配置了 RAW 流就会出问题 | 配置了 RAW 流，但请求只输出 YUV，跑 8 s / 231 帧 | 正常，说明只有输出 RAW 的请求才计数 |
| 重建会话能清零 | 同一设备上重新 `createCaptureSession`（只需 6 ms） | 不能，计数继续累加 |
| 重开设备能清零 | `CameraDevice.close()` 后重新 `openCamera` | 能，测了 5 段共 325 帧 |

**为什么报错时只交付了 58 帧**：`pipelineMaxDepth` = 8，第 66 个请求提交进 HAL 时，前面还有约 8 个请求在管线里没出来。所以“已交付帧数 + 管线里的请求数”约等于 66。

### 4.2 自动曝光：快门被防频闪锁在 1/100 s（已复核更正）

> 更正：本报告初版写的是“带 RAW 流时自动曝光不工作”，这个结论是错的，已在 2026-11-01 复核后改正。

只要请求里有 RAW 输出，CamX 日志里会反复出现下面这类报错：

```
CamX: [ERROR][STATS_AEC] caecxcore.cpp:192: setStats CID:[0] Did not receive any grid stats
CamX: [ERROR][STATS_AFD] afdalgorithminterface.cpp:942: ... StatsBayerGrid does not have channel data
```

但 AE 实际上在工作。主摄用 RAW + YUV 的 repeating 请求、开着 AE，分别设置不同的曝光补偿，收敛后的结果如下：

| 曝光补偿 | AE 状态 | 曝光时间 | ISO |
|---|---|---|---|
| −2 EV | CONVERGED | 10 ms | 74 |
| 0 EV | CONVERGED | 10 ms | 333 |
| +2 EV | CONVERGED | 10 ms | 876（到达上限） |

曝光时间一直停在 10 ms，是因为 AE 处在 50 Hz 防频闪模式，把快门锁在 1/100 s，靠调 ISO 控制亮度。自动白平衡同样正常，室内灯光下给出的增益约为 R 1.58、B 2.03。

手动曝光（`CONTROL_AE_MODE_OFF` 加上 `SENSOR_EXPOSURE_TIME` 和 `SENSOR_SENSITIVITY`）也完全生效，回传值和设定值一致。

### 4.3 没有原生全分辨率 RAW

主摄 OV64B 是 64MP（约 9248×6944），前置 S5K3P9SP 是 16MP（约 4608×3456），但 HAL 只输出四合一像素合并后的 4624×3472 和 2304×1728。HAL 没有声明 `ULTRA_HIGH_RESOLUTION_SENSOR` 能力，也没有 `pixelArraySizeMaximumResolution`，第三方 App 拿不到原生全分辨率。超广角 IMX355 本来就是 8MP，输出的就是原生全尺寸。

### 4.4 逻辑摄像头和辅助 ID 不可用

| ID | 属性 | 打开 RAW 会话的结果 |
|---|---|---|
| 4 | 逻辑多摄，由物理 ID 0、2、3 组成 | 请求卡住直到超时 |
| 5 | 逻辑多摄，由物理 ID 2、0 组成 | HAL 段错误崩溃 |
| 6 | 前置（特殊用途，API1 设备 2） | HAL 段错误崩溃 |

两次崩溃的位置相同，都是空指针读取：

```
#00 com.qti.chi.override.so (UsecaseVTCam::ExecuteCaptureRequest(camera3_capture_request*)+380)
#05 camera.xiaomi.so (Xiaomi::MiCamExtensionModule::submitRequest(...))
```

崩溃后相机 provider 约 10 秒内自动重启，这段时间所有摄像头都打不开。即使加一路 YUV 流也一样会崩。

### 4.5 其他

- 只要任何一个请求输出 RAW，就算进 65 个的额度。预览阶段只输出 YUV 就不会消耗额度。
- `noiseProfile` 缺失，DNG 里不会写入噪声模型。
- `PER_FRAME_CONTROL` 是 HAL 声明的能力；逐帧切换曝光（比如包围曝光）本次没有实测。

## 5. 性能边界汇总

| 边界 | 主摄 | 超广角 | 前置 |
|---|---|---|---|
| 最高 RAW 帧率 | 30.07 fps | 30.0 fps | 30.03 fps |
| RAW 分辨率（唯一档位） | 4624×3472 | 3280×2464 | 2304×1728 |
| 每次打开设备最多 RAW 帧数 | 65 | 65 | 65 |
| 一次不中断的满速连拍最长时间 | 约 2.13 s | 约 2.13 s | 约 2.13 s |
| 长时间连续拍（分段）平均帧率 | 约 25 fps | 约 25.6 fps | 约 25.4 fps |
| 每段之间断档 | 约 0.6 s | 约 0.5 s | 约 0.55 s |
| 最长曝光 | 1 s | 1 s | 0.84 s |
| 65 帧 RAW16 占用内存 | 约 2.09 GB | 约 1.05 GB | 约 0.52 GB |

## 6. App 设计建议

1. **预览只出 YUV。** 按下快门时再发一次 `captureBurst`（不超过 65 帧），拍完立刻 `close()` 掉 `CameraDevice`，重新打开后再恢复预览。这样每次连拍都能拿到完整的 65 帧额度。 PocoPaw：**[App 已实现]**。预览只出 YUV；额度不足时才重开设备，不是每次拍完都关闭。
2. **测光用 YUV 预览，RAW 用手动曝光。** 预览阶段让 AE 收敛，读出曝光时间和 ISO 的乘积，再按需要的欠曝量换算成手动参数设进 RAW 连拍请求。优先压低 ISO、延长快门（受手抖限制），不要沿用 AE 被防频闪锁死的 1/100 s。 PocoPaw：**[App 已实现]**（`BurstExposure`，−1 EV，OIS 66 ms / 无 OIS 33 ms）。
3. **长时间采集要分段。** 每段最多 65 个请求，段间重开设备。可以接受约 0.6 秒的断档，换来约 25 fps 的平均帧率。不要指望用这台机器录连续的 RAW 视频。 PocoPaw：**[未实现]**（暂无连续采集功能）。
4. **主摄连拍先存内存。** 主摄 30 fps 的数据率接近 1 GB/s，65 帧约 2 GB，应该先放在内存里，再异步写盘（或转成 DNG）。如果内存吃紧，优先用 `RAW10`（体积小 37.5%，帧率不变）。 PocoPaw：**[App 已实现]**，帧拷贝放在原生内存中；RAW10 **[未实现]**。
5. **只用物理 ID 0、1、2（和 3）。** 不要打开 4、5、6；要变焦就自己在物理摄像头之间切换。 PocoPaw：**[App 已实现]**。
6. **代码要能从 `ERROR_CAMERA_DEVICE` 里恢复。** 收到这个错误后应关闭设备并重新打开，而不是退出。 PocoPaw：**[App 已实现]**。
7. **要彻底修掉 65 帧上限**，需要修改 `com.qti.chi.override.so`（厂商闭源）或 LineageOS 的设备树/HAL 配置，App 侧绕不过去。建议把 4.1 节的日志反馈给 marble 设备树的维护者。 PocoPaw：App 侧无法处理，**[未实现]**。

## 7. 测试方法

- 探针 App 用 Camera2 + `ImageReader` 接收 RAW 帧。`maxImages` = 8，回调线程拿到帧就立刻释放（拷贝或写盘测试除外）。
- 帧率和帧间隔用图像的传感器时间戳计算。repeating 模式会去掉前 1 秒的预热期。
- 丢帧判断：帧间隔超过中位数 1.5 倍就算有丢帧，同时统计 `onCaptureFailed` 和 `onCaptureBufferLost`。
- 延迟 = App 收到帧时的 `elapsedRealtimeNanos` 减去传感器时间戳（时间戳源是 `REALTIME`）。
- 每轮测试后检查 crash 日志缓冲区，发现 HAL 崩溃就等 provider 恢复后再继续。
- 复现方法：`probe/build.sh` 负责编译、安装并授权（需要 Android SDK build-tools 36 和 platform android-36）。`probe/burst.sh <超时秒数> <am 参数>` 运行单项测试，例如：

```sh
./burst.sh 20 --es id 0 --es mode burst --es n 65                     # 主摄单次 65 帧连拍
./burst.sh 60 --es id 1 --es mode chunk --es n 65 --es reps 5 --ez reopendev true   # 前置分段重开
./burst.sh 30 --es id 2 --es mode burst --es n 20 --es exp 100000000 --es iso 100   # 超广角 100 ms 手动曝光
```

可用参数：`id`、`fmt`（raw16 / raw10）、`mode`（repeat / burst / chunk）、`n`、`reps`、`dur`、`exp`（纳秒，0 表示自动）、`iso`、`tpl`（preview / still / zsl）、`yuv`、`rawtarget`、`copy`、`save`、`reopen`、`reopendev`、`max`。
