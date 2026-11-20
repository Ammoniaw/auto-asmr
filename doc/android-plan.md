# 安卓版技术方案（调研）

代码目录：仓库根目录的 [`auto-asmr-android/`](../auto-asmr-android/)。本文是 2026-11 的调研结论，还没有实现；做完验证后按实际结果更新。

## 目标

- 带一个内嵌浏览器。哪个网页在发声，就把它的声音实时送进 auto-asmr 引擎，输出处理后的双耳声音（类似桌面版的实时模式，只是输入从麦克风换成了网页）。
- 保留 Rust 版的核心选项：三个风格预设、全部效果参数、降噪、手动摆位和位置图、背景雨声、配置导入 / 导出（与桌面版 `.ron` 通用）、效果 / 原声对比。
- 去掉麦克风实时（主播）模式，以及音频设备、缓冲这类桌面设置（由系统决定）。

## 结论

**用系统自带的 Android WebView（它本身就是 Chromium）当浏览器，在 App 自己的进程里用 PLT hook 截住 WebView 的 AAudio 数据回调。** Chromium 每填好一块声音，先在原地交给 Rust 引擎处理，再交还系统播放。Rust 引擎用 NDK 原生编译、通过 JNI 调用，和桌面版是同一份 `dsp` 代码；界面用 Kotlin + Jetpack Compose。

这样不用自己编译 Chromium、不用录屏授权，网页用 `<video>`、MSE、HLS、Web Audio 中的哪一种都能截到。额外延迟只有引擎本身的 2.4 ms。

## 为什么可行：WebView 的声音怎么出来

以下依据 Chromium main 分支源码（2026-11 核对）：

| 事实 | 出处 |
|---|---|
| WebView 的"浏览器进程"就是宿主 App 的进程。网页在单独的沙箱渲染进程里解码、混音，PCM 经共享内存送回浏览器进程，由那里的 audio service 输出 | [WebView 架构](https://chromium.googlesource.com/chromium/src/+/main/android_webview/docs/architecture.md)；`content/public/common/content_features.cc` 中 `kAudioServiceOutOfProcess` 只在 Windows / macOS / Linux 默认开启，安卓上 audio service 在浏览器进程内运行 |
| 输出用 AAudio：`kUseAAudioDriver` 默认开（只有电视 HDMI 棒会关掉，改用 OpenSL ES） | `media/audio/audio_features.cc`；`media/audio/android/audio_manager_android.cc` 的 `MakeLowLatencyOutputStream` / `MakeLinearOutputStream` |
| AAudio 用回调模式、32 位浮点：`setFormat(AAUDIO_FORMAT_PCM_FLOAT)`、`setDataCallback(OnAudioDataRequestedCallback, this)`，打开后 CHECK 格式和声道数；`setFramesPerDataCallback` 固定每次回调的帧数（可变长回调 `kAAudioVariableSizedCallbacks` 默认关） | `media/audio/android/aaudio_stream_wrapper.cc` |
| libaaudio 是直接链接的（`libs += [ "aaudio" ]`），不是 dlopen / dlsym。所以 WebView 的 so（`libmonochrome_64.so` 或 `libwebviewchromium.so`）调用 `AAudioStreamBuilder_setDataCallback` 要经过 PLT / GOT，PLT hook 能截到 | `media/audio/BUILD.gn` |
| `AudioTrackOutputStream` 只用于 AC3 / E-AC3 直通（接 HDMI 功放时），耳机场景不走 | `audio_manager_android.cc` 的 `MakeBitstreamOutputStream` |
| 用系统 MediaPlayer 播 HLS 的旧路径已经删除（`media_player_bridge.cc`、`media_player_renderer.cc` 都不在了），HLS 由内置 demuxer 解码，同样走 AAudio | `media/media_options.gni` 的 `enable_hls_demuxer` |

所以最终调用系统音频接口的就是我们自己的进程，回调里拿到的是 Chromium 已经混好、即将播放的那一块 float PCM。

## 方案对比

| 方案 | 做法 | 主要问题 | 结论 |
|---|---|---|---|
| **A. WebView + 进程内 hook AAudio 回调** | 见上 | 依赖 Chromium 内部实现，WebView 每月自动更新，需要检测失效（见[风险](#风险)） | **推荐** |
| B. WebView + 注入 JS，在页面里用 Web Audio 处理 | `addDocumentStartJavaScript` 注入脚本，把 `<video>` / `<audio>` 用 `createMediaElementSource` 接进 AudioWorklet，引擎编译成 WASM | 跨域且没有 CORS 头的媒体会被判为"污染"，输出静音（Blink `media_element_audio_source_handler.cc` 的 `WouldTaintOrigin`）；跨域 iframe、页面自己建的 AudioContext 要逐个挂；降噪等也要编进 WASM；网站一改版就可能失效 | 作为 A 失效时的备用 |
| C. AudioPlaybackCapture（录本 App 的播放） | MediaProjection + `AudioPlaybackCaptureConfiguration.addMatchingUid` | 每次都要用户同意"录屏"，要前台服务；它只是复制一份声音，原声照样播出，会同时听到原声和处理后的两份；还多一道录制缓冲的延迟 | 不可用 |
| D. 自己编译 Chromium | 基于 Chromium 或 Cromite 这类开源分支做浏览器，直接在 `aaudio_stream_wrapper.cc` 里调用引擎 | 最稳，也能分清是哪个标签页的声音；但要 Linux 构建机（源码上百 GB、编译数小时），APK 大，每个 Chromium 版本都要重打补丁、跟进安全更新；Chromium 没有现成的可嵌入库（WebLayer 已停，[chromium-dev 讨论](https://groups.google.com/a/chromium.org/g/chromium-dev/c/7oa3OoEt-7w)） | A、B 都走不通时再考虑 |
| E. GeckoView（Firefox 内核） | Mozilla 官方的可嵌入库 | 不是 Chromium；没有音频输出接口，截声音同样只能 hook 或自己编译 Gecko | 不推荐 |
| F. 系统音效 AudioEffect | 给 WebView 的音频会话挂效果 | 自定义效果要写进系统配置（需要 root），自带的只有均衡、低音、虚拟环绕、动态处理，做不了双耳空间化 | 不可用 |

## 推荐方案的细节

### 结构

```
网页（沙箱渲染进程）：解码、混音 ──共享内存──┐
                                             ▼
App 进程：WebView 浏览器侧（libmonochrome_64.so）
          └─ AAudio 回调：Chromium 填好一块 float PCM（立体声）
               └─ hook 包的一层：下混成单声道 → Engine::process → 左右声道写回同一块缓冲
                    └─ 交还 AAudio → 耳机
```

### hook

- 用 [ByteHook](https://github.com/bytedance/bhook)（字节跳动开源，MIT，支持 Android 4.1–17，Maven 依赖 `com.bytedance:bytehook`）。只 hook 调用方是 WebView so（文件名匹配 `libmonochrome*.so`、`libwebviewchromium.so`）、被调方是 `libaaudio.so` 的这几个函数：
  - `AAudioStreamBuilder_setDataCallback`：把 Chromium 的 `(回调, userData)` 换成 `(我们的回调, 上下文)`，上下文里记着原回调、原 userData 和分到的引擎。
  - `AAudioStreamBuilder_openStream`：记下流的采样率和声道数。
  - `AAudioStream_close`：把引擎还回池子。
- 我们的回调：先调原回调让 Chromium 填数据，再原地处理。回调跑在 AAudio 的高优先级线程上，所以引擎事先建好（池子，比如 4 个），回调里不分配内存、不加锁；参数更新沿用桌面版 `realtime.rs` 的做法（`Arc` 配置 + 版本号原子量）。
- App 启动时（第一个 WebView 创建之前）装好 hook；ByteHook 会自动处理之后才加载的 so。
- 备用：PLT hook 在某些 WebView 版本上不生效时，改用 [ShadowHook](https://github.com/bytedance/android-inline-hook)（同一团队，MIT，v2.0.1）直接在 `libaaudio.so` 里 inline hook 同一个函数，不管调用方怎么链接都能截到。
- hook 这一层用 C 写（几百行，CMake 编译），只负责转发，处理全在 Rust 里，通过 `extern "C"` 接口调用。

### 声音格式

- **声道**：引擎是单声道输入、双耳输出。立体声流先下混（或像桌面版一样只取一个声道 / 取更响的声道），处理后写回左右声道；不是 2 声道的流直接放过。
- **采样率**：Chromium 按设备原生采样率开流，几乎都是 48 kHz，与引擎一致。不是 48 kHz 时，在回调里用现有的 `resample.rs` 先转到 48 kHz、处理后转回，中间加一小段缓冲；第一版可以先直接放过。
- **延迟**：原地处理，只多引擎算法的 2.4 ms（开降噪再加 20 ms）。Chromium 按 AAudio 时间戳做音画同步，不知道这几毫秒，但差这么一点看不出口型对不上。
- **动作**：网页内容是连续的录音，默认用"试听"的行为（`set_live(false)`，动作严格跟着说话走），不用主播模式的"停顿 1 秒恢复"。
- **多个流**：一个流一个引擎，通常一个页面只有一个流。同时有几个流时它们会各走各的，验证时看要不要共用位置。
- **已经是双耳录音的视频**（很多 ASMR 视频本身就是 KU100 录的）：再处理会变成"双重空间化"，要有一键直通，最好能按网站记住。
- **背景雨声**：网页暂停后 Chromium 会停掉输出流，雨声不能只混在网页的流里。雨声单独开一个自己的 AAudio 输出流（Rust 里用 `ndk` 或 `cpal` 的安卓后端）。

### 只处理"在发声的那个网页"

hook 截到的是整个 App 里所有 WebView 的声音，从 AAudio 这一侧分不出是哪个标签页。做法：

- 注入一段文档开始时运行的脚本（`WebViewCompat.addDocumentStartJavaScript`），监听媒体元素的 play / pause / volumechange 和 AudioContext 的状态，通过 `addWebMessageListener` 报告给 App，标签页上显示"正在发声"。
- 用 [`WebViewCompat.setAudioMuted`](https://developer.android.com/reference/androidx/webkit/WebViewCompat)（androidx.webkit，需要 `WebViewFeature.MUTE_AUDIO`）把其它标签页静音，只让选中的那个出声。默认"跟随当前标签页"，也可以锁定某一个。静音的标签页是否完全不再开 AAudio 流，需要在验证时确认。

### 复用 Rust 代码

- `auto-asmr/src/dsp/` 只依赖 rustfft、realfft、nnnoiseless、serde 和内嵌的 `assets/ku100.bin`（10.9 MB）、`pinna.bin`，可以直接编译到 `aarch64-linux-android`。
- 要先把 `auto-asmr` 改成"库 + 程序"：加 `src/lib.rs` 导出 `dsp`；eframe、cpal、rfd、symphonia 放进默认开启的 `desktop` feature；`dsp` 测试里用到的 `crate::cli::test_voice` 等辅助函数挪到测试模块。桌面版行为不变。
- 安卓的 Rust 部分（`auto-asmr-android/rust/`，cdylib，用 `jni` crate）以 `default-features = false` 依赖它。`Params` 本来就是 serde 结构，配置文件直接用 ron 读写，和桌面版互通。
- 参数的名称（中英）、范围、默认值、说明由 Rust 导出成一张表，Kotlin 按表生成滑条，不在两边各写一份。
- 构建：`cargo-ndk` 编出 `jniLibs/arm64-v8a/`，Gradle 任务里调用。Google Play 要求面向 Android 15 以上的新应用和更新支持 16 KB 内存页，用 NDK r28 以上（默认按 16 KB 对齐），或显式加 `-Wl,-z,max-page-size=16384`。ABI 先只做 `arm64-v8a`，模拟器调试加 `x86_64`。

### 界面和其它

- Kotlin + Jetpack Compose：标签页、地址栏、前进后退、书签；底部抽屉是效果面板，分组和桌面版右侧参数一致；位置图（俯视图 + 手动摆位）用 Compose Canvas 画，引擎状态通过 JNI 轮询。
- 后台播放：`mediaPlayback` 类型的前台服务 + MediaSession（锁屏控制）；切到后台时不调用 WebView 的 `onPause()` / `pauseTimers()`。
- 桌面版功能的取舍：

| 功能 | 安卓版 |
|---|---|
| 风格预设、效果参数、降噪、手动摆位、位置图、背景雨声、配置导入导出、效果 / 原声对比、延迟和处理时间显示 | 第一版就有 |
| 麦克风实时（主播）模式、音频设备和缓冲设置 | 去掉 |
| 文件转换、批量、多音频合成、自动对齐、左右双人、3D 视图 | 以后再说（引擎和 `offline.rs` 本身能在安卓上跑，主要是界面工作量） |

## 风险

| 风险 | 应对 |
|---|---|
| PLT hook 在某些 WebView 构建上不生效（RELRO 共享、链接方式变化等） | 换 ShadowHook inline hook；再不行退到方案 B |
| WebView 每月自动更新，Chromium 改了音频后端（比如服务端开关让某些设备改回 OpenSL ES）后 hook 悄悄失效 | 运行时自检：注入脚本报告"在播放"、但几秒内没有经过 hook 的回调，就提示"未能截获声音"并切到方案 B；hook 失败时一律放过、不改声音，保证不崩溃 |
| 国产 ROM 自带的 WebView 版本旧、实现有差异 | 验证时覆盖 Pixel（或原生系统）和一两台主流国产机；App 里显示 WebView 版本 |
| DRM（Widevine）内容的声音是否也经过这条路径 | 验证时确认；不行就提示不支持 |
| 手机 CPU 比桌面弱 | 桌面上每 128 个样本 22–39 µs（Apple M 单核约 0.8–1.5%），手机大核估计仍在单核几个百分点以内，验证时实测每次回调的处理时间和断流次数 |
| 应用商店审核 | 只改本进程内的调用，不碰其它 App 和系统，常见于大厂 App；但没有官方保证 |

## 开发步骤

本机目前没有 Android SDK / NDK、adb、cargo-ndk，也没装 `aarch64-linux-android` 目标，第 0 步先装好。

0. **环境**：Android Studio（含 SDK、NDK r28+），`rustup target add aarch64-linux-android x86_64-linux-android`，`cargo install cargo-ndk`。
1. **验证 hook（最关键）**：最小 App = 一个 WebView + ByteHook，回调里先只做左右声道互换或压低一边。用 B 站 / YouTube 视频（MSE）、一个 HLS 直播、一个 Web Audio 网页游戏、一个直接链接的 mp4 试，日志打出每个流的采样率、声道数、每次回调帧数、性能模式。通过标准：四种都听得出变化，WebView 升级后仍正常。不通过就试 ShadowHook，再不行转方案 B。
2. **接引擎**：拆出 Rust 库、写 JNI 和 C 接口，回调里跑 `Engine::process`，实测处理时间和断流。
3. **浏览器和效果面板**：标签页、发声检测、静音其它标签页、参数面板、位置图、配置导入导出。
4. **完善**：雨声单独输出、后台播放、直通开关、hook 失效自检、发布（GitHub Actions 打 APK）。

## 目录规划

```
auto-asmr-android/
  README.md
  settings.gradle.kts, build.gradle.kts
  app/                       Kotlin + Compose：浏览器、效果面板
  app/src/main/cpp/          hook 层（C，ByteHook），调用 Rust 的 C 接口
  rust/                      cdylib：JNI + C 接口，依赖 ../../auto-asmr 的 dsp
```

## 参考

- Chromium 源码：[`media/audio/android/`](https://source.chromium.org/chromium/chromium/src/+/main:media/audio/android/)、[`media/audio/audio_features.cc`](https://source.chromium.org/chromium/chromium/src/+/main:media/audio/audio_features.cc)、[`media/audio/BUILD.gn`](https://source.chromium.org/chromium/chromium/src/+/main:media/audio/BUILD.gn)、[`media_element_audio_source_handler.cc`](https://source.chromium.org/chromium/chromium/src/+/main:third_party/blink/renderer/modules/webaudio/media_element_audio_source_handler.cc)
- [WebView 架构](https://chromium.googlesource.com/chromium/src/+/main/android_webview/docs/architecture.md)
- [ByteHook](https://github.com/bytedance/bhook)、[ShadowHook](https://github.com/bytedance/android-inline-hook)
- [WebViewCompat（setAudioMuted、addDocumentStartJavaScript、addWebMessageListener）](https://developer.android.com/reference/androidx/webkit/WebViewCompat)
- [AudioPlaybackCaptureConfiguration](https://developer.android.com/reference/android/media/AudioPlaybackCaptureConfiguration)
- [GeckoView](https://firefox-source-docs.mozilla.org/mobile/android/geckoview/index.html)
