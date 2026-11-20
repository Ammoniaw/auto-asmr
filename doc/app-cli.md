# 命令行（Rust 版）

不带参数启动是图形界面；带子命令时在终端运行。

```bash
auto-asmr convert 输入.m4a -o 输出.wav --style ku100 --duration 60
```

`--style` 有三个内置预设：`ear`（贴耳加强，默认）、`ku100`（像真实 KU100 轻语实录）、`earclean`（掏耳，给棉签 / 挖耳勺 / 刮擦这类触发音用）。风格先生效，后面写的参数再覆盖它。掏耳的目标响度默认是 −20 LUFS，入耳的几项可以单独调（`--inear` 入耳深度、`--canal` 耳道共鸣、`--rumble` 耳内低频、`--muffle` 堵耳闷感，见 `--help`）：

```bash
auto-asmr convert 棉签.wav -o 掏耳.wav --style earclean --hold-time 60 --rumble 0.8
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

同一首歌的几个版本（前奏长短不同、编曲不同）加 `--mix-align` 自动对齐：从一起开始唱的地方叠加，对不上的开头和结尾裁掉，`--start` / `--duration` 按对齐之后的时间算；完成后打印每段从哪里开始用、裁掉多少（做法见[界面使用说明 · 自动对齐](app-usage.md#自动对齐同一首歌的几个版本)）：

```bash
auto-asmr convert 摇滚版.m4a DJ版.m4a --mix --mix-align --duration 60 -o 合成.wav
```

自动平衡（让几个声音每一段时间都差不多一样响，比如一左一右时左右差不多大）加 `--mix-balance`，后面的数值是强度 %（不写就是 100）；完成后打印平衡前后的响度差（做法见[界面使用说明 · 自动平衡](app-usage.md#自动平衡让左右两边差不多大)）：

```bash
auto-asmr convert a.m4a b.m4a --mix --mix-balance 80 -o 合成.wav
```

只想先看怎么对齐、不输出音频：

```bash
auto-asmr align 摇滚版.m4a DJ版.m4a
```

它还会打印偏移、是否逐样本、得分与次好得分之比、重叠部分的相关和几处偏移相差多少，用来判断结果可不可靠。

```bash
auto-asmr --help
```

命令行里降噪是 `--denoise 12`（底噪最多压低 12 dB，范围 0–100）；只写 `--denoise` 是降到最干净。想单独听降噪的效果（不做空间化）：`auto-asmr decode 输入.m4a --denoise 12 -o 降噪后.wav`。

`--save-trajectory 轨迹.csv` 导出实际走位，列是：时间、方位角、距离、不含忽远忽近的距离、说话活动、仰角、入耳深度（米）。

其他命令：`auto-asmr devices` 列出音频设备，`auto-asmr bench` 测处理速度，`auto-asmr rt-sim 音频文件` 不经过声卡、按实时模式把一个文件跑一遍并打印走位（可加 `--style ku100` / `--style earclean`、`--gain -30` 模拟小音量麦克风、`--noise -50` 加底噪、`--sparse 1.5/4` 模拟说 1.5 秒停 4 秒、`--manual "5:90,0.105,0;10:auto"` 模拟第 5 秒手动摆到右耳、第 10 秒切回自动；掏耳风格下还会打印伸进耳道的次数和在耳道里的时间占比）。
