# 构建、发布与开发（Rust 版）

## 构建

需要 Rust 工具链（<https://rustup.rs>）。

```bash
cd auto-asmr
```

```bash
cargo build --release
```

生成的程序在 `target/release/auto-asmr`（Windows 是 `auto-asmr.exe`）。HRIR、房间、耳机补偿数据都内嵌在程序里，单个文件即可运行，约 25 MB。

| 系统 | 说明 |
|---|---|
| Windows | 直接双击 `auto-asmr.exe`。直播时把输出设备选成虚拟声卡（如 VB-Cable）。 |
| macOS | 运行 `tools/bundle_macos.sh` 生成 `target/release/auto-asmr.app`，双击启动。打包后 macOS 才会弹出麦克风授权。 |
| Linux | 编译前需要 ALSA 开发库（Debian / Ubuntu：`sudo apt install libasound2-dev`）。 |

## GitHub 自动构建与发布

推送任意名称的 tag 都会触发 `.github/workflows/release.yml`：使用 Rust 1.95.0 并行测试和构建 Windows x64、Linux x64 发布版，验证程序启动，两个平台全部成功后自动创建或更新该 tag 对应的 GitHub Release。附件包含 `auto-asmr.exe`、`auto-asmr-windows-x64.zip`、`auto-asmr-linux-x64.tar.gz` 和统一的 `SHA256SUMS.txt`；压缩包内含说明文档，Actions 页面也可下载构建产物。

Linux 版在 Ubuntu 22.04 上构建（GNU/glibc），解压后运行 `./auto-asmr`，需要桌面图形环境以及 ALSA 等运行库；打包保留程序的可执行权限。

```bash
git tag v0.1.0
git push origin v0.1.0
```

工作流文件需要先提交并推送，且包含在该 tag 指向的提交中。本地创建 tag 后，需要推送到 GitHub 才会触发构建。

## 测试

```bash
cargo test --release
```

70 个测试，包括：三个风格按钮互相切换不留痕迹、不动降噪 / 耳廓 / 输出等设置，旧配置文件读进来新参数取默认值；入耳深度为 0 时入耳这一组参数怎么设输出都逐样本不变；掏耳预设下声音只在一只耳朵里（另一只低 40 dB 以上）、耳内低频比摩擦声高 6–24 dB；耳道共振峰落在 c / 4L 上、峰高接近设定值、直流增益为 1，共振频率飞快滑动时输出有界，并且在引擎里跟着入耳深度移动；耳内低频集中在 200 Hz 以下、是粉红谱、与输入成正比、没伸进耳道或没有声音时为 0；进出的手势在深浅两头都停、共振每秒滑动不超过 5 个倍频程、换边途中一定已经退出耳道，手动摆位时只有贴在耳朵上才伸进去；随机走位的正面比例；导出的配置文件能原样导入、手动摆位时目标不动位置就纹丝不动、突然换位置时每 1.3 ms 最多挪 2 mm（不跳变）、自动与手动来回切换时位置连续、位置图的拖动换算与画图的刻度互为反函数、降噪各档在稳定底噪上降的分贝数与设定值相差不到 0.6 dB、降噪前后两路信号严格对齐、两级分块房间卷积与直接卷积逐样本一致、限制器不超过上限、BS.1770 响度读数、输入输出时钟相差 ±0.1% 时 60 秒不断流、话少时实时模式照常换耳、从头顶绕时位置连续、上下全关时仰角恒为 0；多音频合成：一左一右时各自那一侧比另一侧响 20 dB 以上（文件转换和试听都测）、响度对齐和音量增减的分贝数、默认位置两两相隔至少 80°、几个引擎并行与顺序处理逐位相同、试听中换位置和跳转立即生效；一起移动时两个音源始终相隔 150° 以上（试听中改移动方式也一样）、四个音源环绕时相邻约 90°、只有一个人在说话时走位与单独转换相同。

## 代码结构

| 路径 | 内容 |
|---|---|
| `src/dsp/` | 信号处理：数据表、直达声渲染、房间、运动、语音分析、人声处理、响度与限制器、引擎；`mix.rs` 是多音频合成（每个音源一个引擎、位置、响度对齐、相加） |
| `src/realtime.rs` | 设备枚举、输入输出流、时钟漂移补偿 |
| `src/offline.rs` | 文件解码、两遍处理、写 WAV |
| `src/gui.rs` | 界面 |
| `assets/ku100.bin` | 内嵌数据（10.9 MB） |
| `tools/export_assets.py` | 从 Python 版的预处理结果重新导出 `assets/ku100.bin` |
| `tools/bundle_macos.sh` | 打包 macOS 应用 |

重新导出数据（一般不需要）：

```bash
uv run tools/export_assets.py
```
