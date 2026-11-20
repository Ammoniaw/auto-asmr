#!/usr/bin/env bash
# 在已连接的手机 / 模拟器上，依次用测试页的几种方式出声，抓 hook 和引擎的日志。
#
#   tools/hook_test.sh                     # 跑全部用例，日志在 hook-test-out/
#   CASES="webaudio hls" tools/hook_test.sh
#   URL=https://www.bilibili.com/video/xxx UA=desktop WAIT=25 tools/hook_test.sh   # 打开任意网页
#   REC=20 CASES=hls tools/hook_test.sh    # 另外录 20 秒处理前后的声音，拉回 hook-test-out/
#
# 每个用例：重启 App → 打开测试页并自动播放（模式 = 左右互换，用来核对截到的就是要播放的声音）→ 等 WAIT 秒
# → 依次切到 直通 / 静音 各 MODE_WAIT 秒 → 切到 ASMR（引擎处理）等 ASMR_WAIT 秒 → 收集日志。
set -euo pipefail
cd "$(dirname "$0")/.."
ADB=${ADB:-adb}
PKG=io.github.nasuiyile.autoasmr
APK=${APK:-app/build/outputs/apk/debug/app-debug.apk}
OUT=${OUT:-hook-test-out}
WAIT=${WAIT:-12}
MODE_WAIT=${MODE_WAIT:-3}
ASMR_WAIT=${ASMR_WAIT:-8}
CASES=${CASES:-webaudio element mp4 hls hlsjs}
UA=${UA:-mobile}
REC=${REC:-0}
TEST=https://appassets.androidplatform.net/assets/test.html
DEV_DIR=/sdcard/Android/data/$PKG/files

mkdir -p "$OUT"
$ADB install -r "$APK" >/dev/null
echo "设备：$($ADB shell getprop ro.product.model) / Android $($ADB shell getprop ro.build.version.release)"
echo "WebView：$($ADB shell dumpsys webviewupdate | grep -m1 'Current WebView package' | sed 's/^ *//')"

start() { $ADB shell am start -n $PKG/.MainActivity "$@" >/dev/null 2>&1; }

run_case() {
  local name=$1 url=$2
  $ADB shell am force-stop $PKG
  $ADB logcat -c
  start --es url "'$url'" --es ua "$UA" --ei mode 2
  sleep "$WAIT"
  for m in 0 3; do
    start --ei mode $m
    sleep "$MODE_WAIT"
  done
  start --ei mode 1
  if [ "$REC" -gt 0 ]; then
    $ADB shell rm -f "$DEV_DIR/rec_*.wav"
    start --ei rec "$REC"
    sleep $((REC + 3))
  fi
  sleep "$ASMR_WAIT"
  $ADB logcat -d -v time -s AsmrHook:V AsmrPage:V AsmrCore:V > "$OUT/$name.log"
  echo "== $name"
  grep -E "hook AAudioStreamBuilder_setDataCallback|流#[0-9]+ (打开|建处理器)|\[test\]|AsmrCore|录音完成" "$OUT/$name.log" | grep -v "video t=" | sed 's/^/   /' | head -10 || true
  grep -E "模式=左右互换" "$OUT/$name.log" | tail -1 | sed -E 's/^.*\| (处理前)/   左右互换：\1/' || true
  grep -E "模式=ASMR" "$OUT/$name.log" | tail -2 | sed -E 's/^.*\| (处理前)/   ASMR：\1/' || true
  if [ "$REC" -gt 0 ]; then
    for f in $($ADB shell ls "$DEV_DIR" | grep "^rec_.*\.wav" || true); do
      $ADB pull "$DEV_DIR/$f" "$OUT/${name}_$f" >/dev/null && echo "   录音：$OUT/${name}_$f"
    done
  fi
}

if [ -n "${URL:-}" ]; then
  run_case custom "$URL"
else
  for c in $CASES; do run_case "$c" "$TEST?auto=$c"; done
fi
