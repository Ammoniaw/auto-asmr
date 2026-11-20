#!/usr/bin/env python3
"""生成插件图标（只用标准库）：蓝底圆角方块，中间一个头，右耳边一个声源和两道声波。

用法：python3 tools/make_icons.py   → 写到 extension/icons/icon{16,32,48,128}.png
"""

import math
import os
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "extension", "icons")


def png(path, w, h, rgba):
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(rgba[y * w * 4 : (y + 1) * w * 4]) for y in range(h))
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(data)


def shade(x, y):
    """单位坐标（0–1）上的颜色（r, g, b, a），a 为 0 表示透明。"""
    # 圆角方块
    m, rad = 0.04, 0.22
    dx = max(abs(x - 0.5) - (0.5 - m - rad), 0)
    dy = max(abs(y - 0.5) - (0.5 - m - rad), 0)
    if math.hypot(dx, dy) > rad:
        return (0, 0, 0, 0)
    t = y
    bg = (int(0x2F + (0x5B - 0x2F) * t), int(0x6F + (0x93 - 0x6F) * t), int(0xDD + (0xF0 - 0xDD) * t))
    white = (255, 255, 255)
    hx, hy, hr = 0.42, 0.52, 0.2
    # 头
    if math.hypot(x - hx, y - hy) < hr:
        return (*white, 255)
    # 鼻子（朝上的小三角）
    if hy - hr - 0.07 < y < hy - hr + 0.02 and abs(x - hx) < (y - (hy - hr - 0.07)) * 0.8:
        return (*white, 255)
    # 右耳边的声源
    sx, sy = hx + hr + 0.07, hy
    if math.hypot(x - sx, y - sy) < 0.055:
        return (255, 214, 102, 255)
    # 两道声波（以声源为圆心、朝右的弧）
    d = math.hypot(x - sx, y - sy)
    ang = math.degrees(math.atan2(y - sy, x - sx))
    if abs(ang) < 50:
        for r in (0.13, 0.21):
            if abs(d - r) < 0.025:
                return (*white, 255)
    return (*bg, 255)


def render(n):
    ss = 4 if n >= 32 else 6
    px = []
    for j in range(n):
        for i in range(n):
            acc = [0, 0, 0, 0]
            for b in range(ss):
                for a in range(ss):
                    c = shade((i + (a + 0.5) / ss) / n, (j + (b + 0.5) / ss) / n)
                    acc[0] += c[0] * c[3]
                    acc[1] += c[1] * c[3]
                    acc[2] += c[2] * c[3]
                    acc[3] += c[3]
            if acc[3] == 0:
                px += [0, 0, 0, 0]
            else:
                px += [round(acc[0] / acc[3]), round(acc[1] / acc[3]), round(acc[2] / acc[3]), round(acc[3] / (ss * ss))]
    return px


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for n in (16, 32, 48, 128):
        png(os.path.join(OUT, f"icon{n}.png"), n, n, render(n))
        print(f"icons/icon{n}.png")
