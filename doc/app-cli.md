# 命令行（Rust 版）

不带参数启动是图形界面；带子命令时在终端运行。

```bash
auto-asmr convert 输入.m4a -o 输出.wav --style ku100 --duration 60
```

批量：给多个文件或文件夹，`--out-dir` 指定输出文件夹（不写就放在源文件旁边），`-j` 指定并行数（默认 CPU 逻辑核数）：

```bash
auto-asmr convert a.m4a b.mp3 某文件夹/ --out-dir 输出/ -j 8
```

多音频合成：2–4 个输入加 `--mix`。`--pos` 按输入顺序给位置（`left` / `right` / `front` / `back` / `front-left` …，或"方位角[:到头中心的距离 米]"，如 `45:0.3`），不写就是一左一右；`--mix-gain` 是各自的音量增减：

```bash
auto-asmr convert a.m4a b.m4a --mix -o 合成.wav --pos left,right --mix-gain 0,-3
```

一起移动加 `--mix-follow`，走位方式用 `--mode`（默认穿梭）：

```bash
auto-asmr convert a.m4a b.m4a --mix --mix-follow --mode orbit -o 合成_环绕.wav
```

```bash
auto-asmr --help
```

命令行里降噪是 `--denoise 12`（底噪最多压低 12 dB，范围 0–100）；只写 `--denoise` 是降到最干净。想单独听降噪的效果（不做空间化）：`auto-asmr decode 输入.m4a --denoise 12 -o 降噪后.wav`。

其他命令：`auto-asmr devices` 列出音频设备，`auto-asmr bench` 测处理速度，`auto-asmr rt-sim 音频文件` 不经过声卡、按实时模式把一个文件跑一遍并打印走位（可加 `--gain -30` 模拟小音量麦克风、`--noise -50` 加底噪、`--sparse 1.5/4` 模拟说 1.5 秒停 4 秒、`--manual "5:90,0.105,0;10:auto"` 模拟第 5 秒手动摆到右耳、第 10 秒切回自动）。
