#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24", "scipy>=1.10", "h5py>=3.8"]
# ///
"""ASMR 双耳空间化：把任意人声变成"有人在你耳边 / 身边说话"（戴耳机听）。

做法
  * 直达声：TH Köln 实测的 Neumann KU100 人工头近场 HRIR（0.25–1.5 m，水平面 1° 分辨率）。
    KU100 正是很多 ASMR 主播用的录音人头。0.25 m 以内（嘴贴耳边）用刚性球头模型
    （Duda & Martens 1998）外推到 0.105 m，近场的低频双耳声级差、头影都按物理算。
  * HRIR 拆成"最小相位滤波 + 双耳时延"，逐块插值；时延逐样本变化，所以声源移动时
    ITD 平滑、带真实的多普勒，不会有拉链声。
  * 运动：默认"慢速穿梭"——贴左耳停 2.5 s，用 2.5 s 经脸前慢慢移到右耳，再移回来，持续不断；
    移动时按双耳总响度补偿，声音是"跑过去"而不是"这边变小那边变大"。始终叠加人体的微小晃动。
  * 忽远忽近（默认开启）：贴耳停留、慢慢往后让、再慢慢凑回来，时长和深浅都随机；声音大的地方
    提前往后让、轻声时更贴近。退开时耳边音量只保留物理衰减的 45%（KU100 实录 40–56%）。
  * 上下（默认开启自动上下）：嘴在耳朵上方 / 下方之间慢慢换位置（--updown）；也可以固定一个高度
    （--height），或者换耳时从头顶绕过去（--over-top）。水平面的查找表不变，离开水平面时再乘上
    取自 KU100 全球面实测的高度修正（耳廓对来声方向的染色）；都关掉时结果与原来逐样本相同。
  * 人头麦贴耳感（默认 ear 风格）：嘴离耳朵约 1.7 cm，远耳额外压低 32 dB，近耳低频更厚、
    临场感更强；离开耳边时平滑恢复。--style ku100 改为按两段 KU100 实录标定（远耳听得到、
    离耳约 6 cm、真实房间）。不改变原来的嗓音，也不添加任何底噪或呼吸声。
  * 贴耳气声（默认开启）：贴耳时把嗓音里轻的送气声提上来（--breath），加上气流扑到耳朵上的
    低频脉冲（--puff，由嗓音自己的送气声导出），并补上近耳 4.3 kHz 的耳甲共鸣（--concha）。
  * 贴耳轻声（--soft-near）、嘴的大小（--mouth，嘴是开口而不是点）、远耳高频遮挡（--far-shadow），默认开启。
  * 耳廓选择（--pinna A…G）：把 KU100 的耳廓换成真人的（HUTUBS 实测），默认仍是 KU100。
  * 房间（可选）：TH Köln 用同一个 KU100 实测的真实房间，或镜像声源法的合成房间。
  * KU100 公共扩散场均衡（CDFC），然后按 BS.1770 响度归一化。

输入如果是立体声，只取一个声道（默认取更响的那个），再做空间化。

用法（uv 会自动装好依赖）：
  uv run asmr/asmr_binaural.py 输入.wav                       # 默认：从左耳慢慢移到右耳再移回
  uv run asmr/asmr_binaural.py 输入.wav --move-time 5 --hold-time 1   # 移得更慢
  uv run asmr/asmr_binaural.py 输入.wav --mode pingpong       # 参考视频的节奏：3 秒一个来回
  uv run asmr/asmr_binaural.py 输入.m4a -o 输出.wav --mode wander   # 在你身边走动着说话
  uv run asmr/asmr_binaural.py 输入.wav --mode orbit --radius 0.3 --period 12
  uv run asmr/asmr_binaural.py 输入.wav --mode static --az 270 --dist 0.13   # 一直贴着左耳
  uv run asmr/asmr_binaural.py 输入.wav --mode path --path 轨迹.json
方位角 az：0=正前，90=右，180=正后，270=左（顺时针，单位度）；距离 dist：到头中心的米数。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal, special

FS = 48000
C = 343.0
HEAD_A = 0.0875            # 球头半径（与 KU100 耳间距一致）
BLOCK = 256                # 滤波器更新块长（5.3 ms）
NTAP = 128                 # 最小相位 HRIR 长度
NFFT_BLK = 512
R_MIN, R_MAX = 0.095, 3.0  # 允许的声源距离（到头中心；耳朵在 0.0875 m 处，0.095 ≈ 嘴碰到耳朵）
EAR_R = 0.105              # 贴耳轻语时嘴到头中心的距离（离耳朵约 1.7 cm）
R_MEAS = (0.25, 0.5, 0.75, 1.0, 1.5)
R_GAIN = (1.0, 0.33, 0.25, 0.16, 0.095)   # TH Köln 给的各距离数据集的还原增益
D0 = 4.0                   # 固定附加时延（样本），保证时延恒为正
NSEC = 12                  # 房间早期反射按 30° 扇区预计算
TABLE_VERSION = 4
ELEV_TAPS = 48             # 高度修正滤波器长度
ELEV_DA = 5.0              # 高度修正表的网格：从耳朵看声源的侧向角每 5° 一格（−90…90）
ELEV_DB = 10.0             # 绕两耳连线的极角每 10° 一格（0 = 前，90 = 上，180 = 后，270 = 下）
REAL_ROOM_DB = -28.0       # 真实房间：贴耳时远耳听到的房间声比近耳直达声低多少（KU100 实测 −24…−35 dB）

HERE = Path(__file__).resolve().parent
HRTF_DIR = HERE / "hrtf"
DATA_DIR = HERE / "data"
PINNA_FILE = DATA_DIR / "pinna_hutubs.npz"   # 耳廓选择的滤波表（由 auto-asmr/tools/build_pinna.py 从 HUTUBS 生成）
ZENODO = "https://zenodo.org/records/4297951/files/{}?download=1"
CIRC_FILES = [f"HRIR_CIRC360_NF{round(r * 100):03d}.sofa" for r in R_MEAS]
SPH_FILE = "HRIR_L2702_NF150.sofa"
CDFC_FILE = "KU100_CDFC.mat"
HPCF_ZIP = "HPCF.zip"

ROOMS = {
    # dims = 宽(左右) x 深(前后) x 高，lis = 听者头部位置，t60 = 中频混响时间，beta = 墙面反射系数
    "treated": dict(dims=(3.2, 2.9, 2.5), lis=(1.45, 1.25, 1.05), t60=0.20, beta=0.62),
    "bedroom": dict(dims=(3.9, 3.4, 2.6), lis=(1.70, 1.45, 1.00), t60=0.32, beta=0.78),
    "living": dict(dims=(5.6, 4.4, 2.8), lis=(2.40, 1.90, 1.10), t60=0.50, beta=0.85),
}
T60_SHAPE = ((125, 250, 500, 1000, 2000, 4000, 8000, 16000),
             (1.25, 1.15, 1.05, 1.00, 0.95, 0.85, 0.70, 0.50))

# ANSI S3.5-1997 表 3：不同说话力度的标准语音频谱（1/3 倍频程，dB）
ANSI_F = (160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000)
ANSI_NORMAL = (32.41, 34.48, 34.75, 33.98, 34.59, 34.27, 32.06, 28.3, 25.01, 23.0, 20.15, 17.32, 13.18, 11.55,
               9.33, 5.31, 2.59, 1.13)
ANSI_RAISED = (33.81, 33.92, 38.98, 38.57, 39.11, 40.15, 38.78, 36.37, 33.86, 31.89, 28.58, 25.32, 22.35, 20.15,
               16.78, 11.47, 7.67, 5.07)

# 以下频谱形状取自 B 站 KU100 贴耳录音的实测（见 README「KU100 实测」），单位 dB
EAR_EQ_F = (500, 2000, 4000, 6000, 8000, 10000, 12500, 16000, 22000)
EAR_EQ_DB = (0.0, 0.0, -1.5, -4.0, -5.0, -5.5, -4.0, -3.0, -3.0)              # 近耳音色校正（对齐 KU100 近耳/居中差）
CONCHA_F = 4300.0          # 耳甲腔深度共振（Shaw 1979：堵住耳道口时 4.3 kHz，各个方向来的声音都能激发）
BREATH_FC = 2500.0         # 气声带下限：F3 以上送气噪声取代谐波（Klatt & Klatt 1990）
BREATH_HOP = 64            # 气声电平检测的块长（1.33 ms，与 Rust 版的控制块相同）
BREATH_REF0 = -90.0        # "响的"电平跟踪的初值（dBFS）：从很低开始，听到第一个响的音就跟上去，与输入音量无关
PUFF_K = 1.5               # 气流脉冲的标定增益（--puff 1 时）
MOUTH_SPREAD = ((-1.0, 0.10), (-0.5, 0.25), (0.5, 0.25), (1.0, 0.10))   # 嘴宽方向的取样点（相对半宽）和权重，中心点权重 0.30
MOUTH_R = 0.4              # 离头中心超过这个距离时嘴可以当成一个点
FAR_SHADOW_F = 6000.0      # 远耳高频遮挡的转折频率
PINNA_TAPS = 64            # 耳廓选择的滤波器长度

# 语句模式的落点：(方位角范围, 距离范围)，右侧定义，左侧镜像
# 风格预设：ear=贴耳加强（默认，远耳额外压低，嘴离耳约 1.7 cm）；
# ku100=按两段 KU100 实录（B 站 BV1X4ZyBREqr、BV1XHBXBaEmt，扣除底噪后）标定：两耳差中位约 12 dB、
# 大部分时间离耳 5–30 cm，远耳清楚地听得到，有真实房间
STYLES = {
    "ear": dict(isolation=32.0, room="none", headphone="diffuse", far_lf_cut=12.0, dist=None),
    "ku100": dict(isolation=0.0, room="real", headphone="native", far_lf_cut=0.0, dist=0.15),
}

ANCHORS = {
    "ear": ((80.0, 98.0), (0.11, 0.125)),     # 贴耳轻语
    "front": ((28.0, 60.0), (0.22, 0.38)),    # 侧前方，凑近脸
    "back": ((112.0, 150.0), (0.20, 0.38)),   # 侧后方
    "center": ((-15.0, 15.0), (0.30, 0.50)),  # 正前方
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- HRTF 数据

def _download(url: str, dst: Path) -> None:
    log(f"  下载 {url.split('/')[-1].split('?')[0]} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "asmr-binaural/1.0"})
    tmp = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dst)


def ensure_hrtf(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for zname, members in (("NFHRIR_CIRC360_SOFA.zip", CIRC_FILES), ("NFHRIR_L2702_SOFA.zip", [SPH_FILE])):
        if all((d / m).exists() for m in members):
            continue
        log("首次运行：从 Zenodo 下载 TH Köln KU100 近场 HRIR（CC BY 4.0）")
        zp = d / zname
        _download(ZENODO.format(zname), zp)
        with zipfile.ZipFile(zp) as z:
            for m in members:
                info = next(i for i in z.infolist() if Path(i.filename).name == m)
                (d / m).write_bytes(z.read(info))
        zp.unlink()
    if not (d / CDFC_FILE).exists():
        _download(ZENODO.format(CDFC_FILE), d / CDFC_FILE)


def headphone_models(d: Path) -> list[Path]:
    hp = d / "HPCF"
    if not any(hp.glob("*.mat")):
        hp.mkdir(parents=True, exist_ok=True)
        log("下载 TH Köln 在 KU100 上实测的耳机补偿滤波器（HPCF）...")
        zp = d / HPCF_ZIP
        _download(ZENODO.format(HPCF_ZIP), zp)
        with zipfile.ZipFile(zp) as z:
            for i in z.infolist():
                nm = Path(i.filename).name
                if nm.endswith(".mat") and not nm.startswith("._") and "__MACOSX" not in i.filename:
                    (hp / nm).write_bytes(z.read(i))
        zp.unlink()
    return sorted(hp.glob("*.mat"))


def playback_filter(tab, d: Path, name: str):
    """回放端补偿（模拟"硬件链路"）：
    diffuse = 通用扩散场补偿（KU100 CDFC，假设耳机是扩散场调音）；
    native  = 不补偿，保留 KU100 出厂内置均衡后的原机音色，和真实 KU100 录音一样（高频略暗）；
    型号    = TH Köln 在同一个 KU100 上实测的该款耳机补偿滤波器（Engel 等 2019：通用耳机补偿可减少染色、改善距离感）。"""
    key = name.strip().lower()
    if key in ("diffuse", "df", ""):
        return tab.cdfc
    if key in ("native", "none", "ku100"):
        return None
    norm = lambda v: re.sub(r"[^a-z0-9]", "", v.lower())
    files = headphone_models(d)
    hits = [f for f in files if norm(name) in norm(f.stem)]
    exact = [f for f in hits if norm(f.stem).endswith(norm(name))]
    hits = exact if len(exact) == 1 else hits
    if len(hits) != 1:
        raise SystemExit(f"耳机型号 {name!r} 没有唯一匹配。可选：diffuse、native、" + "、".join(f.stem for f in files))
    import scipy.io as sio
    log(f"回放补偿：{hits[0].stem}（TH Köln 在 KU100 上实测）")
    return np.asarray(sio.loadmat(hits[0], squeeze_me=True, struct_as_record=False)["hpcf"].minPhase, float)


def read_sofa(p: Path):
    import h5py
    with h5py.File(p, "r") as f:
        ir = np.asarray(f["Data.IR"], dtype=np.float64)
        pos = np.asarray(f["SourcePosition"], dtype=np.float64)
        fs = float(np.asarray(f["Data.SamplingRate"]).ravel()[0])
    if fs != FS:
        raise RuntimeError(f"{p.name}: 采样率 {fs} 不是 {FS}")
    return ir, pos


def minphase_from_mag(mag: np.ndarray, nfft: int, ntap: int) -> np.ndarray:
    """幅度谱 (..., nfft//2+1) -> 最小相位 FIR (..., ntap)，倒谱法。"""
    lm = np.log(np.maximum(mag, 1e-7 * mag.max(axis=-1, keepdims=True)))
    cep = np.fft.irfft(lm, nfft, axis=-1)
    w = np.zeros(nfft)
    w[0] = 1.0
    w[1:nfft // 2] = 2.0
    w[nfft // 2] = 1.0
    h = np.fft.irfft(np.exp(np.fft.rfft(cep * w, axis=-1)), nfft, axis=-1)[..., :ntap]
    nf = ntap // 8
    h[..., -nf:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, nf)))
    return h


def minimum_phase(h: np.ndarray, ntap: int = NTAP, nfft: int = 4096) -> np.ndarray:
    return minphase_from_mag(np.abs(np.fft.rfft(h, nfft, axis=-1)), nfft, ntap)


def onset(h: np.ndarray, up: int = 8, thr: float = 0.15) -> np.ndarray:
    """脉冲响应起始时刻（样本，8 倍上采样后取阈值），用来提取 ITD。"""
    u = np.abs(signal.resample_poly(h, up, 1, axis=-1))
    return np.argmax(u >= thr * u.max(axis=-1, keepdims=True), axis=-1) / up


def smooth_circ(v: np.ndarray, n: int) -> np.ndarray:
    return np.mean([np.roll(v, k, axis=0) for k in range(-(n // 2), n // 2 + 1)], axis=0)


def sphere_mag(r: float, cos_t: np.ndarray, freqs: np.ndarray, a: float = HEAD_A) -> np.ndarray:
    """刚性球头的 |H|（Duda & Martens 1998），以无头时球心处的自由场声压归一化。返回 (len(cos_t), len(freqs))。"""
    rho = r / a
    mu = 2 * np.pi * np.maximum(freqs, 1.0) * a / C
    M = int(np.ceil(mu.max() * rho)) + 200
    acc = np.zeros((len(cos_t), len(freqs)), complex)
    p_prev = np.ones_like(cos_t)
    p_cur = cos_t.copy()
    with np.errstate(all="ignore"):
        for m in range(M + 1):
            if m == 0:
                pm = p_prev
            elif m == 1:
                pm = p_cur
            else:
                pm = ((2 * m - 1) * cos_t * p_cur - (m - 1) * p_prev) / m
                p_prev, p_cur = p_cur, pm
            hn = special.spherical_jn(m, mu * rho) + 1j * special.spherical_yn(m, mu * rho)
            hd = special.spherical_jn(m, mu, derivative=True) + 1j * special.spherical_yn(m, mu, derivative=True)
            ratio = hn / hd
            ratio[~np.isfinite(ratio)] = 0.0   # 高阶项在低频下溢出，真实值趋于 0
            acc += (2 * m + 1) * pm[:, None] * ratio[None, :]
    return (rho / mu)[None, :] * np.abs(acc)


def ear_path(r: float, theta: np.ndarray) -> np.ndarray:
    """声源到球面耳点的路径长度（看得见走直线，看不见绕球面）。"""
    th0 = np.arccos(HEAD_A / r)
    direct = np.sqrt(r * r + HEAD_A ** 2 - 2 * HEAD_A * r * np.cos(theta))
    wrap = np.sqrt(r * r - HEAD_A ** 2) + HEAD_A * (theta - th0)
    return np.where(theta <= th0, direct, wrap)


def parallax_az(r: float, az_deg: np.ndarray, R: float = 0.25) -> np.ndarray:
    """近场声学视差：从每只耳朵看向声源，把这条射线延长到半径 R 的测量圆上，得到该耳应使用的方位角。
    （Romblom & Cook 2008 的几何查表法；Spagnol 2017：视差在 20 cm 以内才明显。）返回 (n, 2)，SOFA 逆时针。"""
    az = np.radians(az_deg)
    s = np.stack([r * np.cos(az), r * np.sin(az)], -1)
    out = []
    for ey in (HEAD_A, -HEAD_A):                      # 通道 0 = 左耳（+y），1 = 右耳
        e = np.array([0.0, ey])
        d = s - e
        a2 = (d * d).sum(-1)
        b = 2 * (d @ e)
        c = e @ e - R * R
        t = (-b + np.sqrt(b * b - 4 * a2 * c)) / (2 * a2)
        q = e + t[:, None] * d
        out.append(np.degrees(np.arctan2(q[:, 1], q[:, 0])) % 360.0)
    return np.stack(out, -1)


def near_extrapolate(h25: np.ndarray, tau25: np.ndarray, r: float):
    """把 0.25 m 实测 HRIR 外推到更近的 r：耳廓部分按声学视差取各耳自己的方向，
    再乘以球头模型的距离变化函数（DVF，含低频视差），时延按几何修正。"""
    nfft = 4096
    f = np.fft.rfftfreq(nfft, 1 / FS)
    fc = np.geomspace(20, FS / 2, 192)
    th = np.radians(np.arange(181.0))
    g = (0.25 / r) * sphere_mag(r, np.cos(th), fc) / sphere_mag(0.25, np.cos(th), fc)
    lf = np.log(np.maximum(f, 20.0))
    G = np.exp(np.stack([np.interp(lf, np.log(fc), row) for row in np.log(g)]))  # (181, F)
    az = np.radians(np.arange(360.0))
    th_l = np.degrees(np.arccos(np.clip(np.sin(az), -1, 1)))   # SOFA：逆时针，90° 在左，通道 0 = 左耳
    idx = np.rint(np.stack([th_l, 180.0 - th_l], axis=1)).astype(int)   # (360, 2)
    pa = parallax_az(r, np.arange(360.0))            # (360, 2)
    a0 = np.floor(pa).astype(int) % 360
    fr = (pa - np.floor(pa))[..., None]
    base = np.empty_like(h25)
    for e in range(2):
        base[:, e] = (1 - fr[:, e]) * h25[a0[:, e], e] + fr[:, e] * h25[(a0[:, e] + 1) % 360, e]
    mag = np.abs(np.fft.rfft(base, nfft, axis=-1)) * G[idx]
    h = minphase_from_mag(mag, nfft, NTAP)
    thr = np.radians(idx)
    dpath = (ear_path(r, thr) - r) - (ear_path(0.25, thr) - 0.25)
    return h, tau25 + dpath / C * FS


@dataclass
class Tables:
    inv_grid: np.ndarray   # (NR,) 1/r，递减
    H: np.ndarray          # (NR, 360, 2, NTAP) 最小相位 HRIR，SOFA 方位角
    TAU: np.ndarray        # (NR, 360, 2) 相对时延（样本）
    sph_dirs: np.ndarray   # (P, 3) 全球面方向单位向量（SOFA 坐标）
    sph_H: np.ndarray      # (P, 2, NTAP) 1.5 m 全球面最小相位 HRIR（已乘还原增益）
    sph_tau: np.ndarray    # (P, 2)
    cdfc: np.ndarray       # (256,) KU100 公共扩散场均衡
    e_ref_1m: float        # 1 m 直达声单耳能量（经 CDFC），房间混响标定用
    elev: np.ndarray       # (2, 37, 36, ELEV_TAPS) 高度修正滤波器：[耳, 侧向角, 极角]


def _smooth_pow(m2: np.ndarray, nfft: int, frac: float = 6.0, bmin: float = 1000.0) -> np.ndarray:
    """功率谱平滑：带宽取 1/frac 倍频程和 bmin Hz 里较宽的那个（48 点的 FIR 只能表示这么细的结构）。"""
    c = np.concatenate([np.zeros(m2.shape[:-1] + (1,)), np.cumsum(m2, -1)], -1)
    n = m2.shape[-1]
    k = np.arange(n)
    hw = np.maximum(k * (2 ** (0.5 / frac) - 1), 0.5 * bmin / (FS / nfft))
    lo = np.clip(np.round(k - hw).astype(int), 0, n - 1)
    hi = np.clip(np.round(k + hw).astype(int), 0, n - 1)
    return (c[..., hi + 1] - c[..., lo]) / (hi + 1 - lo)


def build_elevation(dirs: np.ndarray, h_sph: np.ndarray, r0: float = 1.5) -> np.ndarray:
    """高度修正表。水平面的查找表只有"方位角 × 距离"；声源离开水平面时，两耳时间差、声级差和近场效应
    主要由"偏向哪只耳朵多少"（侧向角）决定，所以先取同一侧向角上水平面前 / 后两点的混合，再乘上这里的
    修正——它只描述耳廓对来声方向的染色怎么随"绕两耳连线转多少"（极角）变化。

    修正取自 KU100 1.5 m 的全球面实测（2702 个方向）：某方向的频谱 ÷ 同侧向角水平面前 / 后混合的频谱，
    方向按"从这只耳朵看过去"算（近处的声源，耳朵看到的方向和从头中心看到的不一样，即声学视差）。
    用它预测 0.5 m 的全球面实测，近耳 6–9 kHz 的误差从 7.0 dB（不修正）降到 1.8 dB。
    水平面上（极角 0 / 180）修正恒为 1，所以不带高度时结果与原来完全相同。
    返回 (2, NA, NB, ELEV_TAPS) 的最小相位 FIR。"""
    from scipy.spatial import cKDTree
    nfft = 512
    al = np.radians(np.arange(-90.0, 90.0 + 1e-6, ELEV_DA))
    be = np.radians(np.arange(0.0, 360.0, ELEV_DB))
    A, B = np.meshgrid(al, be, indexing="ij")
    q = np.stack([np.cos(A) * np.cos(B), np.sin(A), np.cos(A) * np.sin(B)], -1)   # 耳心坐标：y 朝这只耳朵自己的一侧
    wf = 0.5 * (1 + np.cos(be))
    j180 = int(round(180.0 / ELEV_DB))
    out = np.zeros((2, len(al), len(be), ELEV_TAPS), np.float32)
    for e, ey in enumerate((HEAD_A, -HEAD_A)):
        v = r0 * dirs - np.array([0.0, ey, 0.0])
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        if e == 1:
            v = v * np.array([1.0, -1.0, 1.0])
        dist, idx = cKDTree(v).query(q.reshape(-1, 3), k=4)
        ang = 2 * np.arcsin(np.clip(dist / 2, 0, 1))
        w = 1.0 / (ang ** 2 + np.radians(0.7) ** 2)
        w /= w.sum(1, keepdims=True)
        g = (w[..., None] * h_sph[idx, e]).sum(1).reshape(len(al), len(be), -1)   # 各网格方向的 HRIR
        ref = wf[None, :, None] * g[:, 0:1] + (1 - wf)[None, :, None] * g[:, j180:j180 + 1]
        num = _smooth_pow(np.abs(np.fft.rfft(g, nfft, axis=-1)) ** 2, nfft)
        den = _smooth_pow(np.abs(np.fft.rfft(ref, nfft, axis=-1)) ** 2, nfft)
        db = np.clip(10 * np.log10(np.maximum(num, 1e-12) / np.maximum(den, 1e-12)), -15.0, 12.0)
        out[e] = minphase_from_mag(10 ** (db / 20), nfft, ELEV_TAPS)
    return out


def build_tables(d: Path) -> Tables:
    log("预处理 HRIR（只在第一次运行时做，约半分钟）...")
    circ_h, circ_tau = [], []
    for name, g in zip(CIRC_FILES, R_GAIN):
        ir, pos = read_sofa(d / name)
        order = np.argsort(pos[:, 0])
        ir = ir[order] * g
        if len(order) != 360 or not np.allclose(pos[order, 0], np.arange(360)):
            raise RuntimeError(f"{name}: 方位角网格不是 0..359")
        tau = smooth_circ(onset(ir), 5)
        circ_tau.append(tau - tau.mean())
        circ_h.append(minimum_phase(ir))

    inv = np.concatenate([np.linspace(1 / R_MIN, 4.0, 12, endpoint=False), np.linspace(4.0, 1 / 1.5, 36)])
    H = np.zeros((len(inv), 360, 2, NTAP), np.float32)
    TAU = np.zeros((len(inv), 360, 2))
    inv_meas = 1 / np.array(R_MEAS)
    for i, iv in enumerate(inv):
        r = 1 / iv
        if iv <= 4.0 + 1e-9:
            j = int(np.clip(np.searchsorted(-inv_meas, -iv, side="right") - 1, 0, len(R_MEAS) - 2))
            w = (iv - inv_meas[j + 1]) / (inv_meas[j] - inv_meas[j + 1])
            H[i] = w * circ_h[j] + (1 - w) * circ_h[j + 1]
            TAU[i] = w * circ_tau[j] + (1 - w) * circ_tau[j + 1]
        else:
            h, t = near_extrapolate(circ_h[0], circ_tau[0], r)
            H[i], TAU[i] = h, t

    ir, pos = read_sofa(d / SPH_FILE)
    ir = ir * R_GAIN[-1]
    tau = onset(ir)
    azr, elr = np.radians(pos[:, 0]), np.radians(pos[:, 1])
    dirs = np.stack([np.cos(elr) * np.cos(azr), np.cos(elr) * np.sin(azr), np.sin(elr)], axis=1)

    import scipy.io as sio
    cdfc = np.asarray(sio.loadmat(d / CDFC_FILE, squeeze_me=True, struct_as_record=False)["hpcf"].minPhase, float)
    h1 = H[np.argmin(np.abs(inv - 1.0))]
    e1 = np.fft.irfft(np.fft.rfft(h1, 512, axis=-1) * np.fft.rfft(cdfc, 512), 512, axis=-1)
    sph_h = minimum_phase(ir).astype(np.float32)
    tab = Tables(inv, H, TAU, dirs, sph_h, tau - tau.mean(), cdfc,
                 float(np.mean(np.sum(e1 ** 2, axis=-1))), build_elevation(dirs, sph_h.astype(np.float64)))
    np.savez_compressed(d / f"ku100_tables_v{TABLE_VERSION}.npz", **tab.__dict__)
    return tab


def load_tables(d: Path) -> Tables:
    cache = d / f"ku100_tables_v{TABLE_VERSION}.npz"
    if cache.exists():
        z = np.load(cache)
        return Tables(**{k: (float(z[k]) if k == "e_ref_1m" else z[k]) for k in z.files})
    ensure_hrtf(d)
    return build_tables(d)


def lookup(tab: Tables, az_sofa: np.ndarray, r: np.ndarray):
    """按 (方位角, 距离) 双线性插值出 HRIR 与时延。"""
    nr = len(tab.inv_grid)
    rr = np.clip(r, R_MIN, 1.5)
    u = np.interp(1 / rr, tab.inv_grid[::-1], np.arange(nr)[::-1].astype(float))
    i0 = np.minimum(np.floor(u).astype(int), nr - 2)
    fu = u - i0
    a = np.mod(az_sofa, 360.0)
    a0 = np.floor(a).astype(int) % 360
    a1 = (a0 + 1) % 360
    fa = a - np.floor(a)
    w = [((1 - fu) * (1 - fa)), ((1 - fu) * fa), (fu * (1 - fa)), (fu * fa)]
    ii = [(i0, a0), (i0, a1), (i0 + 1, a0), (i0 + 1, a1)]
    h = sum(wk[:, None, None] * tab.H[i, j] for wk, (i, j) in zip(w, ii))
    tau = sum(wk[:, None] * tab.TAU[i, j] for wk, (i, j) in zip(w, ii))
    h = h / (np.maximum(r, 1.5) / 1.5)[:, None, None]   # 1.5 m 以外按 1/r 衰减
    return h.astype(np.float32), tau


def cone_coords(az_sofa: np.ndarray, el: np.ndarray):
    """（SOFA 方位角, 仰角）-> 方向余弦 (dx 前, dy 左, dz 上)、侧向角 alpha（度，朝左为正）、
    极角 beta（度，绕两耳连线：0 前、90 上、180 后、270 下）、水平面"前"那一点的权重 wf。"""
    a, e = np.radians(az_sofa), np.radians(el)
    dx, dy, dz = np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)
    alpha = np.degrees(np.arcsin(np.clip(dy, -1.0, 1.0)))
    beta = np.mod(np.degrees(np.arctan2(dz, dx)), 360.0)
    return (dx, dy, dz), alpha, beta, 0.5 * (1.0 + np.cos(np.radians(beta)))


def grid_taps(g: np.ndarray, lat: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """在（从耳朵看的侧向角, 极角）网格上双线性插值出滤波器 (n, taps)。g 为一只耳朵的表 (NA, NB, taps)。"""
    na, nb = g.shape[0], g.shape[1]
    x = (np.clip(lat, -90.0, 90.0) + 90.0) / ELEV_DA
    i0 = np.minimum(np.floor(x).astype(int), na - 2)
    fa = (x - i0)[:, None]
    y = np.mod(beta, 360.0) / ELEV_DB
    j0 = np.floor(y).astype(int) % nb
    j1 = (j0 + 1) % nb
    fb = (y - np.floor(y))[:, None]
    return (1 - fa) * ((1 - fb) * g[i0, j0] + fb * g[i0, j1]) + fa * ((1 - fb) * g[i0 + 1, j0] + fb * g[i0 + 1, j1])


def elev_taps(tab: Tables, ear: int, lat: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """高度修正滤波器 (n, ELEV_TAPS)：在（从耳朵看的侧向角, 极角）网格上双线性插值。"""
    return grid_taps(tab.elev[ear], lat, beta)


def pinna_names() -> list[str]:
    return [str(n) for n in np.load(PINNA_FILE)["names"]] if PINNA_FILE.exists() else []


def load_pinna(name: str):
    """耳廓选择：返回 (2, NA, NB, PINNA_TAPS) 的滤波表；"ku100" 返回 None（用 KU100 自己的耳廓）。"""
    if name.strip().lower() in ("", "ku100", "none"):
        return None
    z = np.load(PINNA_FILE)
    names = [str(n).upper() for n in z["names"]]
    if name.strip().upper() not in names:
        raise SystemExit(f"--pinna 只能是 ku100 或 {' / '.join(names)}")
    return z["tables"][names.index(name.strip().upper())]


def pinna_apply(h: np.ndarray, az_sofa: np.ndarray, el: np.ndarray | None, r: np.ndarray, P: np.ndarray) -> None:
    """换耳廓（原地修改 h）：给每只耳朵卷上"真人方向传递函数 ÷ KU100 方向传递函数"的修正，方向按从这只耳朵
    看向声源算（所以贴耳时用的是耳朵看到的方向，与近场的声学视差一致）。近场的距离、两耳差、时延仍是 KU100 实测的，
    整体音色也仍是 KU100 的；换掉的只是 1.5 kHz 以上"随方向变化"的耳廓峰谷。表由 HUTUBS 真人实测 HRTF 生成。"""
    a = np.radians(az_sofa)
    e = np.zeros_like(a) if el is None else np.radians(el)
    dx, dy, dz = np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)
    rr = np.maximum(r, R_MIN)
    fade = np.ones(NTAP)
    fade[-16:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, 16)))
    for ear, ey in enumerate((HEAD_A, -HEAD_A)):
        vx, vy, vz = rr * dx, rr * dy - ey, rr * dz
        lat = np.degrees(np.arcsin(np.clip((vy if ear == 0 else -vy) / np.sqrt(vx ** 2 + vy ** 2 + vz ** 2), -1, 1)))
        beta = np.mod(np.degrees(np.arctan2(vz, vx)), 360.0)
        for i in range(0, len(h), 8192):
            d = grid_taps(P[ear], lat[i:i + 8192], beta[i:i + 8192])
            h[i:i + 8192, ear] = np.fft.irfft(np.fft.rfft(h[i:i + 8192, ear], NFFT_BLK, axis=-1) * np.fft.rfft(d, NFFT_BLK, axis=-1),
                                              NFFT_BLK, axis=-1)[:, :NTAP] * fade


def lookup3(tab: Tables, az_sofa: np.ndarray, el: np.ndarray | None, r: np.ndarray):
    """带高度的查表。仰角为 0 的位置直接用水平面的表（结果与 lookup 完全相同）；离开水平面时，
    取同一侧向角上水平面前 / 后两点的混合，再给每只耳朵卷上高度修正（见 build_elevation）。"""
    if el is None or not np.any(np.abs(el) > 1e-4):
        return lookup(tab, az_sofa, r)
    (dx, dy, dz), alpha, beta, wf = cone_coords(az_sofa, el)
    hf, tf = lookup(tab, np.mod(alpha, 360.0), r)
    hb, tb = lookup(tab, np.mod(180.0 - alpha, 360.0), r)
    h = wf[:, None, None] * hf + (1 - wf)[:, None, None] * hb
    tau = wf[:, None] * tf + (1 - wf)[:, None] * tb
    rr = np.maximum(r, R_MIN)
    fade = np.ones(NTAP)
    fade[-16:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, 16)))
    for e, ey in enumerate((HEAD_A, -HEAD_A)):
        vy = rr * dy - ey
        lat = np.degrees(np.arcsin(np.clip((vy if e == 0 else -vy) / np.sqrt((rr * dx) ** 2 + vy ** 2 + (rr * dz) ** 2), -1, 1)))
        for i in range(0, len(h), 8192):          # 分块，免得占太多内存
            d = elev_taps(tab, e, lat[i:i + 8192], beta[i:i + 8192])
            he = np.fft.irfft(np.fft.rfft(h[i:i + 8192, e], NFFT_BLK, axis=-1) * np.fft.rfft(d, NFFT_BLK, axis=-1),
                              NFFT_BLK, axis=-1)[:, :NTAP] * fade
            h[i:i + 8192, e] = he
    flat = np.abs(el) <= 1e-4                       # 仍在水平面上的位置：与不带高度时逐位相同
    if flat.any():
        h0, t0 = lookup(tab, az_sofa[flat], r[flat])
        h[flat], tau[flat] = h0, t0
    return h.astype(np.float32), tau


# ----------------------------------------------------------------------------- 渲染

def frac_delay(x: np.ndarray, idx: np.ndarray, d: np.ndarray, up: int = 4, pad: int = 48) -> np.ndarray:
    """y[i] = x(idx[i] - d[i])：4 倍多相上采样 + Catmull-Rom 插值，越界部分视为 0。"""
    pos = idx - d
    lo = int(np.floor(pos.min())) - pad
    hi = int(np.ceil(pos.max())) + pad
    seg = np.zeros(hi - lo, np.float32)
    s0, s1 = max(lo, 0), min(hi, len(x))
    if s1 > s0:
        seg[s0 - lo:s1 - lo] = x[s0:s1]
    u = signal.resample_poly(seg, up, 1)
    p = (pos - lo) * up
    i = np.floor(p).astype(np.int64)
    f = p - i
    y0, y1, y2, y3 = u[i - 1], u[i], u[i + 1], u[i + 2]
    return (y1 + 0.5 * f * (y2 - y0 + f * (2 * y0 - 5 * y1 + 4 * y2 - y3 + f * (3 * (y1 - y2) + y3 - y0)))).astype(np.float32)


def ear_intimacy(h_b: np.ndarray, az_b: np.ndarray, r_b: np.ndarray, iso_db: float, presence_db: float,
                 ear_eq: bool = True, dark_db: float = 0.0, el_b: np.ndarray | None = None,
                 concha_db: float = 0.0, far_shadow_db: float = 0.0) -> None:
    """人头麦贴耳感（原地修改 h_b）：嘴贴近一侧耳朵时，近耳加临场 / 空气感，远耳额外压低（单耳独占）；
    离开耳边、转到脸前时按距离和侧向程度平滑恢复。"""
    p = closeness(az_b, r_b, el_b)
    if p.max() <= 0:
        return
    ipsi = (np.mod(az_b, 360.0) < 180.0).astype(int)   # 声源在右边 -> 近耳是右耳（通道 1）
    k = np.arange(len(h_b))
    if iso_db > 0:
        h_b[k, 1 - ipsi] *= (10 ** (-iso_db * p / 20))[:, None].astype(np.float32)
    if ear_eq or presence_db != 0 or dark_db != 0 or concha_db != 0:
        fz = np.maximum(np.fft.rfftfreq(NFFT_BLK, 1 / FS), 1.0)
        d = np.interp(np.log(fz), np.log(EAR_EQ_F), EAR_EQ_DB) if ear_eq else np.zeros_like(fz)  # KU100 实测近耳校正
        d = d + presence_db * np.exp(-0.5 * (np.log2(fz / 3500) / 0.7) ** 2) + 0.7 * presence_db / (1 + (9000 / fz) ** 2)
        d = d - dark_db * np.exp(-0.5 * (np.log2(fz / 5000) / 0.6) ** 2)   # "暗"：5 kHz 附近压低（Terashima 2024）
        d = d + concha_db * np.exp(-0.5 * (np.log2(fz / CONCHA_F) / 0.33) ** 2)   # 耳甲共鸣（约 0.8 倍频程宽）
        eq = np.fft.rfft(minphase_from_mag(10 ** (d / 20), NFFT_BLK, 64), NFFT_BLK)
        fade = np.ones(NTAP, np.float32)
        fade[-16:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, 16)))
        for i in range(0, len(h_b), 8192):
            kk, e = k[i:i + 8192], ipsi[i:i + 8192]
            hi = h_b[kk, e]
            he = np.fft.irfft(np.fft.rfft(hi, NFFT_BLK, axis=-1) * eq, NFFT_BLK, axis=-1)[:, :NTAP] * fade
            h_b[kk, e] = hi + p[i:i + 8192, None] * (he - hi)
    if far_shadow_db > 0:
        # 贴耳时远耳的高频遮挡：球头模型低估了近处声源在远耳的高频头影（与 BEM 近距仿真比，离耳 3.8 cm 时
        # 远耳 6 kHz 以上偏亮 2–9 dB；Brungart & Rabinowitz 1999：声源越近，远耳的头影越深）
        fz = np.maximum(np.fft.rfftfreq(NFFT_BLK, 1 / FS), 1.0)
        d = -far_shadow_db / (1 + (FAR_SHADOW_F / fz) ** 2)
        eq = np.fft.rfft(minphase_from_mag(10 ** (d / 20), NFFT_BLK, 64), NFFT_BLK)
        fade = np.ones(NTAP, np.float32)
        fade[-16:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, 16)))
        for i in range(0, len(h_b), 8192):
            kk, e = k[i:i + 8192], 1 - ipsi[i:i + 8192]
            hi = h_b[kk, e]
            he = np.fft.irfft(np.fft.rfft(hi, NFFT_BLK, axis=-1) * eq, NFFT_BLK, axis=-1)[:, :NTAP] * fade
            h_b[kk, e] = hi + p[i:i + 8192, None] * (he - hi)


def closeness(az_b: np.ndarray, r_b: np.ndarray, el_b: np.ndarray | None = None) -> np.ndarray:
    """嘴贴近某一侧耳朵的程度（0–1）：距离越近、越偏向正侧方越大；转到脸前 / 脑后 / 头顶时为 0。"""
    lat = np.abs(np.sin(np.radians(az_b)))
    if el_b is not None:
        lat = lat * np.cos(np.radians(el_b))
    return (np.clip((0.2 - r_b) / (0.2 - EAR_R), 0, 1) * lat ** 2).astype(np.float32)


def apply_ear_lf(y: np.ndarray, az_b: np.ndarray, r_b: np.ndarray, near_db: float, far_db: float,
                 fc: float = 320.0, el_b: np.ndarray | None = None) -> None:
    """贴耳时近耳低频提升、远耳低频再压低，让低频两耳差更大（Brungart：近场距离感的主导线索；
    Honda：128 Hz 两耳差的起伏与战栗感 r=0.93）。用 4 阶 Linkwitz–Riley 分频，两耳做完全相同的分频，
    所以 ITD 不受影响。原地修改 y (2, N)。"""
    if near_db <= 0 and far_db <= 0:
        return
    p = closeness(az_b, r_b, el_b)
    right = np.mod(az_b, 360.0) < 180.0
    lp = signal.butter(2, fc, "low", fs=FS, output="sos")
    hp = signal.butter(2, fc, "high", fs=FS, output="sos")
    lp4, hp4 = np.vstack([lp, lp]), np.vstack([hp, hp])
    for e in range(2):
        near = right if e == 1 else ~right
        g = 10 ** (np.where(near, near_db * p, -far_db * p) / 20)
        lo = signal.sosfilt(lp4, y[e]).astype(np.float32)
        hi = signal.sosfilt(hp4, y[e]).astype(np.float32)
        for n0 in range(0, y.shape[1], 1 << 22):
            n1 = min(y.shape[1], n0 + (1 << 22))
            y[e, n0:n1] = hi[n0:n1] + block_interp(g, n0, n1) * lo[n0:n1]


def soften_voice(x: np.ndarray, amount: float) -> np.ndarray:
    """降低嗓音里的"说话力度"线索：Brungart & Scott 2001 —— 正常音量的说话声，力度每增加 15 dB 听起来远一倍，
    而播放音量几乎不影响；耳语 / 轻声听起来最近。按 ANSI S3.5 标准语音频谱中"正常"与"提高"力度的形状差
    （去掉整体电平）反向施加 amount 倍：中频（0.8–3 kHz）略收，低频更饱满、最高频气声略多。只做均衡，不改变嗓音本身。"""
    if amount <= 0:
        return x
    diff = np.array(ANSI_RAISED) - np.array(ANSI_NORMAL)
    shape = np.clip(-(diff - diff.mean()), -3.0, 3.0)
    shape = np.convolve(np.pad(shape, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    nfft = 4096
    f = np.fft.rfftfreq(nfft, 1 / FS)
    db = amount * np.interp(np.log(np.maximum(f, 20.0)), np.log(ANSI_F), shape)
    fir = minphase_from_mag(10 ** (db / 20), nfft, 1024)
    return signal.oaconvolve(x, fir)[:len(x)].astype(np.float32)


def soften_near(x: np.ndarray, p_b: np.ndarray, amount: float) -> np.ndarray:
    """贴耳轻声：人凑到耳边会不自觉地放轻声音、退开时说大声一点（Pelegrín-García 等 2011），而说话力度本身就是
    语音最主要的距离线索——越轻听起来越近，耳语最近（Brungart & Scott 2001；Philbeck & Mershon 2002）。
    这里让"轻声化"的均衡（soften_voice）跟着贴耳程度走：贴耳时为 amount，离开耳边时恢复原样。只做均衡。
    p_b 是各块边界处的贴耳程度（0–1）。"""
    if amount <= 0:
        return x
    soft = soften_voice(x, amount)
    y = np.empty_like(x)
    for n0 in range(0, len(x), 1 << 22):
        n1 = min(len(x), n0 + (1 << 22))
        y[n0:n1] = x[n0:n1] + block_interp(p_b, n0, n1) * (soft[n0:n1] - x[n0:n1])
    return y


def mouth_spread(tab: "Tables", h_b: np.ndarray, az_sofa: np.ndarray, el_b: np.ndarray | None, r_b: np.ndarray,
                 width: float, pinna: np.ndarray | None = None) -> None:
    """嘴的大小（原地修改 h_b）：嘴不是一个点，而是一个约 2 cm 宽的开口（ITU-T P.51 的人工嘴唇环直径 25 mm；
    Halkosaari & Vaalgamaa 2004：真人的等效口半径约 0.5 cm、人工嘴 1.5 cm）。贴耳时这个开口从耳朵看过去张角很大
    （离耳 1.7 cm、宽 2 cm 约 ±30°），耳廓对各个方向的响应被平均，点声源才有的又窄又深的谷会变浅；
    离远了张角趋于 0，和点声源没有区别。做法：沿嘴宽方向（水平、垂直于"头中心—声源"连线）取 5 个点，
    把各点的 HRIR 加权相加（时延已另外处理，这里是对齐后的最小相位响应）。"""
    rc = np.clip(r_b, R_MIN, R_MAX)
    near = np.nonzero(rc < MOUTH_R)[0]
    for i in range(0, len(near), 8192):
        m = near[i:i + 8192]
        acc = (1.0 - sum(w for _, w in MOUTH_SPREAD)) * h_b[m]
        for t, w in MOUTH_SPREAD:
            d = np.degrees(np.arctan2(t * width / 2, rc[m]))
            em = None if el_b is None else el_b[m]
            hk = lookup3(tab, np.mod(az_sofa[m] + d, 360.0), em, r_b[m])[0]
            if pinna is not None:
                pinna_apply(hk, np.mod(az_sofa[m] + d, 360.0), em, r_b[m], pinna)
            acc += w * hk
        h_b[m] = acc


def _sosfilt32(sos: np.ndarray, x: np.ndarray, chunk: int = 1 << 21) -> np.ndarray:
    """分块做 sosfilt（带状态），结果存成 float32，长音频不占太多内存。"""
    y = np.empty(len(x), np.float32)
    zi = np.zeros((sos.shape[0], 2))
    for n0 in range(0, len(x), chunk):
        y[n0:n0 + chunk], zi = signal.sosfilt(sos, x[n0:n0 + chunk], zi=zi)
    return y


def _hop_ramp(vals: np.ndarray, n0: int, n1: int) -> np.ndarray:
    """每 BREATH_HOP 个样本更新一次的控制量 -> 逐样本：第 k 块里从 vals[k-1] 线性走到 vals[k]（第 0 块保持 vals[0]）。"""
    xp = np.arange(len(vals) + 1, dtype=float) * BREATH_HOP
    return np.interp(np.arange(n0, n1, dtype=float), xp, np.concatenate([vals[:1], vals])).astype(np.float32)


def breath_fx(x: np.ndarray, p_b: np.ndarray, lift_db: float, puff: float):
    """贴耳气声：嘴贴近耳朵时把嗓音自己的气声提上来，并加上气流扑到耳朵上的低频脉冲。
    两样都只用嗓音里已有的送气声，不生成任何噪声、不改变嗓音；p_b 是各块边界处的贴耳程度（0–1），离开耳边时都为 0。

    气声增强（lift_db）：气声带 = 2.5 kHz 以上（F3 以上送气噪声取代谐波，气声感主要靠这部分噪声而不是频谱倾斜，
      Klatt & Klatt 1990；Hillenbrand 1994）。只把"轻"的部分提上来（频带内向上压缩）：电平比响的齿音低 16 dB 以上的
      送气声 / 气息 / 轻辅音最多提 lift_db，接近齿音电平的不提，低到底噪附近（低 36–48 dB）的也不提。
      KU100 贴耳实录里气声段与有声段一样响（差 0…−5 dB），普通配音里气声段要低 12–18 dB。
    气流扑耳（puff）：送气时气流扑到贴着的耳朵 / 话筒上，形成 200 Hz 以下、20–100 ms 的压力脉冲（喷麦声）；
      KU100 贴耳实录的近耳 70–150 Hz 有每分钟约 170–200 次、中位 40 ms 的突发。这里取气声带瞬时功率在
      50–130 Hz 的起伏（除以功率的平方根，幅度与气声成正比），送气越强脉冲越强，爆破音开头最明显。
    返回 (处理后的人声, 气流脉冲或 None)；脉冲要加在直达声的输入上，由近场 HRIR 送到贴着的那只耳朵。"""
    if lift_db <= 0 and puff <= 0:
        return x, None
    H = BREATH_HOP
    N = len(x)
    b = _sosfilt32(signal.butter(2, BREATH_FC, "high", fs=FS, output="sos"), x)
    nblk = -(-N // H)
    bp = np.zeros((2, nblk * H), np.float32)
    bp[0, :N], bp[1, :N] = b, x
    ms, msx = np.mean(np.square(bp.reshape(2, nblk, H), dtype=np.float64), axis=2)
    del bp
    dt = H / FS
    ca, cr = 1 - math.exp(-dt / 0.003), 1 - math.exp(-dt / 0.03)      # 电平检测：起控 3 ms，释放 30 ms
    cu, down = 1 - math.exp(-dt / 0.05), 0.5 * dt                      # "响的"电平：50 ms 跟上去；有声音时每秒落 0.5 dB，
    env, lref = 0.0, BREATH_REF0                                       # 静下来（低 30 dB 以上）就保持。lref = 响的齿音电平
    envx, vref = 0.0, BREATH_REF0                                      # 整个人声；vref = 响的元音电平
    G = np.zeros(nblk + 1)                 # 第 k 块开始时可用的提升比例（0–1）
    P = np.empty(nblk + 1)                 # 第 k 块开始时"响的齿音"的功率
    V = np.empty(nblk + 1)                 # 第 k 块开始时"响的人声"的幅度
    P[0], V[0] = 10 ** (lref / 10), 10 ** (vref / 20)
    for j in range(nblk):
        m = ms[j]
        env += (ca if m > env else cr) * (m - env)
        lb = 10 * math.log10(env + 1e-12)
        lref = lref + cu * (lb - lref) if lb > lref else (lref - down if lb > lref - 30.0 else lref)
        m = msx[j]
        envx += (ca if m > envx else cr) * (m - envx)
        lx = 10 * math.log10(envx + 1e-12)
        vref = vref + cu * (lx - vref) if lx > vref else (vref - down if lx > vref - 30.0 else vref)
        d = lref - lb
        u = min(max((d - 4.0) / 12.0, 0.0), 1.0)
        v = min(max((d - 36.0) / 12.0, 0.0), 1.0)
        G[j + 1] = u * u * (3 - 2 * u) * (1 - v * v * (3 - 2 * v))
        P[j + 1] = 10 ** (lref / 10)
        V[j + 1] = 10 ** (vref / 20)
    tk = (np.arange(nblk + 1) + 1.0) * H
    p64 = np.interp(tk, np.arange(len(p_b), dtype=float) * BLOCK, p_b)      # 各块末尾的贴耳程度
    y = x
    if lift_db > 0:
        T = p64 * (10 ** (lift_db * G / 20) - 1)
        y = np.empty_like(x)
        for n0 in range(0, N, 1 << 22):
            n1 = min(N, n0 + (1 << 22))
            y[n0:n1] = x[n0:n1] + _hop_ramp(T, n0, n1) * b[n0:n1]
    pf = None
    if puff > 0:
        q = np.square(b)
        s = _sosfilt32(signal.butter(1, 30, "low", fs=FS, output="sos"), q)
        lp = signal.butter(2, 130, "low", fs=FS, output="sos")
        w = _sosfilt32(np.vstack([signal.butter(2, 50, "high", fs=FS, output="sos"), lp, lp]), q)
        del q
        U = PUFF_K * puff * p64
        pf = np.empty(N, np.float32)
        for n0 in range(0, N, 1 << 22):
            n1 = min(N, n0 + (1 << 22))
            k = np.arange(n0, n1) // H
            pr = P[k]
            # 气声功率的低频起伏 / 功率的平方根 = 与气声幅度成正比；再除以响的齿音幅度，得到与音色明暗无关的量
            raw = w[n0:n1] / np.sqrt((np.maximum(s[n0:n1], 0.0) + 1e-3 * pr) * pr + 1e-30)
            pf[n0:n1] = _hop_ramp(U, n0, n1) * V[k] * np.tanh(raw)   # 幅度跟人声走，软上限 = 响的人声电平
    return y, pf


def deess(x: np.ndarray, max_db: float, thresh_db: float = -8.0) -> np.ndarray:
    """去齿音：4.5–11 kHz 齿音带的能量接近整体能量时（"s/sh/x"），按 3:1 动态压低该频带，最多 max_db。
    用线性相位分频、干声路径对齐，其他频段完全不动（Terashima 2024：5 kHz 附近能量低时刺痛感更强）。"""
    if max_db <= 0:
        return x
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    taps = 255
    h = signal.firwin(taps, [4500, 11000], pass_zero=False, fs=FS)
    band = signal.oaconvolve(x, h)[taps // 2:taps // 2 + len(x)].astype(np.float32)
    hop = int(0.002 * FS)
    n = len(x) // hop
    if n < 10:
        return x
    eb = 10 * np.log10(np.mean(np.square(band[:n * hop].reshape(n, hop), dtype=np.float64), axis=1) + 1e-12)
    ef = 10 * np.log10(np.mean(np.square(x[:n * hop].reshape(n, hop), dtype=np.float64), axis=1) + 1e-12)
    gr = np.clip((eb - ef - thresh_db) * (1 - 1 / 3), 0, max_db)
    gr[ef < ef.max() - 60] = 0
    gr = uniform_filter1d(maximum_filter1d(gr, 9), 9)                    # 起控约 2 ms，释放约 20 ms
    g = 10 ** (-np.interp(np.arange(len(x)), np.arange(n) * hop + hop // 2, gr) / 20)
    return (x - (1 - g.astype(np.float32)) * band).astype(np.float32)


def binaural_power(h: np.ndarray) -> np.ndarray:
    """HRIR 对 (K, 2, NTAP) 的双耳总功率，按语音频谱 + 听感粗略加权。"""
    f = np.fft.rfftfreq(NFFT_BLK, 1 / FS)
    _, bp = signal.sosfreqz(signal.butter(2, [150, 7000], "band", fs=FS, output="sos"), worN=f, fs=FS)
    w = np.abs(bp) ** 2 / (1 + f / 800)
    return (np.abs(np.fft.rfft(h, NFFT_BLK, axis=-1)) ** 2 * w).sum(axis=(-1, -2))


def render_direct(x: np.ndarray, az_b: np.ndarray, r_b: np.ndarray, tab: Tables,
                  level_follow: float = 1.0, iso_db: float = 0.0, presence_db: float = 0.0, ear_eq: bool = True,
                  dark_db: float = 0.0, r_level: np.ndarray | None = None, gain_b: np.ndarray | None = None,
                  chunk_blocks: int = 2048, el_b: np.ndarray | None = None, level_pos=None,
                  concha_db: float = 0.0, mouth: float = 0.0, far_shadow_db: float = 0.0,
                  pinna: np.ndarray | None = None) -> np.ndarray:
    """时变 HRIR 卷积（重叠保留 + 块内交叉淡化）+ 逐样本分数时延。az_b/r_b 为各块边界处的位置。
    r_level：响度补偿按这条轨迹算（不含忽远忽近，退开时的音量变化另由 gain_b 决定）。
    el_b：各块边界处的仰角（度）；level_pos：带高度时响度补偿所用轨迹的 (方位角, 仰角, 距离)。"""
    N = len(x)
    nb = -(-N // BLOCK)
    assert len(az_b) == nb + 1
    if el_b is not None and not np.any(np.abs(el_b) > 1e-4):
        el_b = None
    h_b, tau_b = lookup3(tab, np.mod(-az_b, 360.0), el_b, r_b)   # 用户方位角顺时针 -> SOFA 逆时针
    rc = np.clip(r_b, R_MIN, R_MAX)
    if pinna is not None:
        pinna_apply(h_b, np.mod(-az_b, 360.0), el_b, r_b, pinna)
    lg = None
    if level_follow != 1.0:
        # 响度补偿：按语音频段加权的双耳总功率，相对"贴耳 0.14 m"拉平；方向线索（ITD/ILD/耳廓）不变
        if el_b is not None:
            az0, el0, r0 = level_pos if level_pos is not None else (az_b, el_b, r_b)
            p = path_power(tab, az0, r0, el0)
        elif r_level is None:
            p = np.concatenate([binaural_power(h_b[i:i + 8192]) for i in range(0, len(h_b), 8192)])
        else:
            p = path_power(tab, az_b, r_level)
        pref = binaural_power(lookup(tab, np.array([270.0]), np.array([0.14]))[0])[0]
        lg = ((p / pref) ** (-(1.0 - level_follow) / 2))[:, None, None].astype(np.float32)
    if mouth > 0:
        mouth_spread(tab, h_b, np.mod(-az_b, 360.0), el_b, r_b, mouth, pinna)
    if lg is not None:
        h_b *= lg
    if gain_b is not None:
        h_b *= gain_b[:, None, None].astype(np.float32)
    if ear_eq or iso_db > 0 or presence_db != 0 or dark_db != 0 or concha_db != 0 or far_shadow_db > 0:
        ear_intimacy(h_b, az_b, rc, iso_db, presence_db, ear_eq, dark_db, el_b, concha_db, far_shadow_db)
    d_b = (rc / C * FS)[:, None] + tau_b + D0
    tb = np.arange(nb + 1, dtype=float) * BLOCK
    L = NTAP
    out = np.zeros((2, nb * BLOCK), np.float32)
    w = ((np.arange(BLOCK) + 0.5) / BLOCK).astype(np.float32)
    for k0 in range(0, nb, chunk_blocks):
        k1 = min(nb, k0 + chunk_blocks)
        n0, n1 = k0 * BLOCK, k1 * BLOCK
        idx = np.arange(n0 - (L - 1), n1, dtype=float)
        for e in range(2):
            xd = frac_delay(x, idx, np.interp(idx, tb, d_b[:, e]))
            seg = sliding_window_view(xd, BLOCK + L - 1)[::BLOCK]
            X = np.fft.rfft(seg, NFFT_BLK, axis=-1)
            ya = np.fft.irfft(X * np.fft.rfft(h_b[k0:k1, e], NFFT_BLK, axis=-1), NFFT_BLK, axis=-1)[:, L - 1:L - 1 + BLOCK]
            yb = np.fft.irfft(X * np.fft.rfft(h_b[k0 + 1:k1 + 1, e], NFFT_BLK, axis=-1), NFFT_BLK, axis=-1)[:, L - 1:L - 1 + BLOCK]
            out[e, n0:n1] = ((1 - w) * ya + w * yb).ravel()
    return out[:, :N]


def path_power(tab: Tables, az_b: np.ndarray, r_b: np.ndarray, el_b: np.ndarray | None = None) -> np.ndarray:
    """沿轨迹各块的双耳总功率（语音加权），分块查表以免占用太多内存。带高度时取同一侧向角上
    水平面前 / 后两点功率的混合（高度修正只动 5 kHz 以上，对语音频段的总功率影响很小）。"""
    out = np.empty(len(az_b))
    for i in range(0, len(az_b), 8192):
        az, r = np.mod(-az_b[i:i + 8192], 360.0), r_b[i:i + 8192]
        if el_b is None:
            out[i:i + 8192] = binaural_power(lookup(tab, az, r)[0])
            continue
        el = el_b[i:i + 8192]
        _, alpha, _, wf = cone_coords(az, el)
        pw = (wf * binaural_power(lookup(tab, np.mod(alpha, 360.0), r)[0])
              + (1 - wf) * binaural_power(lookup(tab, np.mod(180.0 - alpha, 360.0), r)[0]))
        flat = np.abs(el) <= 1e-4
        if flat.any():
            pw[flat] = binaural_power(lookup(tab, az[flat], r[flat])[0])
        out[i:i + 8192] = pw
    return out


def sway_gain(tab: Tables, az_b: np.ndarray, r_b: np.ndarray, r_base: np.ndarray, keep: float,
              el_b: np.ndarray | None = None, base_pos=None) -> np.ndarray:
    """忽远忽近时说话人自己的音量变化：退开时会不自觉地说大声一点、凑近时放轻（Pelegrín-García 等 2011；
    KU100 实录里近耳音量只保留了物理衰减的约 44%）。返回每块的增益，耳边音量 = 物理衰减 ** keep。
    这是说话人发出的声音本身变了，所以房间声也要乘同样的增益。
    带高度时 base_pos 给出不含忽远忽近的位置 (方位角, 仰角, 距离)。"""
    if keep >= 1.0 or np.allclose(r_b, r_base):
        return np.ones(len(r_b))
    if el_b is None:
        return (path_power(tab, az_b, r_b) / path_power(tab, az_b, r_base)) ** (-(1.0 - keep) / 2)
    az0, el0, r0 = base_pos
    return (path_power(tab, az_b, r_b, el_b) / path_power(tab, az0, r0, el0)) ** (-(1.0 - keep) / 2)


def conv_add(out: np.ndarray, x: np.ndarray, h: np.ndarray, chunk: int = 1 << 20) -> None:
    """分块 FFT 卷积并累加进 out（超出 out 长度的部分丢弃）：x (N,) 或 (C,N)，h (C,L)。"""
    x = np.atleast_2d(x)
    h = np.atleast_2d(h)
    ch = out.shape[0]
    N, L = x.shape[1], h.shape[1]
    hb = np.broadcast_to(h, (ch, L))
    for n0 in range(0, N, chunk):
        seg = np.broadcast_to(x[:, n0:n0 + chunk], (ch, min(chunk, N - n0)))
        yy = signal.fftconvolve(seg, hb, axes=1)
        m = min(yy.shape[1], out.shape[1] - n0)
        out[:, n0:n0 + m] += yy[:, :m]


def block_interp(vals_b: np.ndarray, n0: int, n1: int) -> np.ndarray:
    tb = np.arange(len(vals_b), dtype=float) * BLOCK
    return np.interp(np.arange(n0, n1, dtype=float), tb, vals_b).astype(np.float32)


def apply_warmth(x: np.ndarray, r_b: np.ndarray, warmth_db: float) -> np.ndarray:
    """近讲温暖感：越近低频越厚（随距离变化的一阶低架）。"""
    if warmth_db <= 0:
        return x
    lp = signal.sosfilt(signal.butter(1, 220, "low", fs=FS, output="sos"), x).astype(np.float32)
    prox = np.clip((0.6 - r_b) / (0.6 - 0.13), 0, 1)
    gb = 10 ** (warmth_db * prox / 20) - 1
    y = np.empty_like(x)
    for n0 in range(0, len(x), 1 << 22):
        n1 = min(len(x), n0 + (1 << 22))
        y[n0:n1] = x[n0:n1] + block_interp(gb, n0, n1) * lp[n0:n1]
    return y


@dataclass
class Room:
    er: np.ndarray     # (NSEC, 2, Ler) 早期反射（已含 CDFC），按扇区
    tail: np.ndarray   # (2, Lt) 混响尾巴


def _img(i: int, L: float, s: float) -> float:
    return i * L + (s if i % 2 == 0 else L - s)


def build_room_measured(tab: Tables, wet_db: float, pb="cdfc") -> Room:
    """真实房间：TH Köln 用同一个 KU100 在真实房间（Audiolab）实测的双耳房间脉冲响应，12 个方向，
    去掉直达声、混响时间按频段缩短到有家具的卧室水平（约 0.35 s）。电平按 KU100 贴耳录音实测定：
    贴耳时远耳听到的房间声比近耳直达声低约 28 dB（--wet 可调），远耳主要是房间声而不是寂静。"""
    z = np.load(DATA_DIR / "room_audiolab_ku100.npz")
    S = z["sectors"].astype(np.float64) * float(z["scale"])                 # (12, 2, L)
    pbf = tab.cdfc if isinstance(pb, str) else pb
    if pbf is not None:
        S = signal.fftconvolve(S, pbf[None, None, :], axes=-1)[..., :S.shape[-1]]
    h, _ = lookup(tab, np.array([260.0]), np.array([EAR_R]))              # 右耳贴耳（方位 100°）
    hn = h[0, 1].astype(np.float64)
    if pbf is not None:
        hn = np.convolve(hn, pbf)
    g = math.sqrt(float(np.sum(hn ** 2)) / float(np.sum(S[3, 0] ** 2)) * 10 ** ((REAL_ROOM_DB + wet_db) / 10))
    return Room((S * g).astype(np.float32), np.zeros((2, 1), np.float32))


def build_room(tab: Tables, name: str, wet_db: float, seed: int = 7, pb="cdfc") -> Room:
    p = ROOMS[name]
    W, D, Hh = p["dims"]
    lis = np.array(p["lis"])
    t60 = p["t60"]
    pbf = tab.cdfc if isinstance(pb, str) else pb
    ner = int(0.06 * FS) + (len(pbf) if pbf is not None else 0)
    nfft = 1 << int(np.ceil(np.log2(ner + NTAP + 256)))
    f = np.fft.rfftfreq(nfft, 1 / FS)
    Hs = np.fft.rfft(tab.sph_H, nfft, axis=-1)
    Hc = np.fft.rfft(pbf, nfft) if pbf is not None else 1.0
    images = [(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)
              if 0 < abs(i) + abs(j) + abs(k) <= 2]
    wet = 10 ** (wet_db / 20)
    er = np.zeros((NSEC, 2, ner), np.float32)
    for s in range(NSEC):
        az = np.radians(s * 360 / NSEC)
        src = lis + 0.6 * np.array([np.sin(az), np.cos(az), 0.0])
        spec = np.zeros((2, len(f)), complex)
        for i, j, k in images:
            v = np.array([_img(i, W, src[0]), _img(j, D, src[1]), _img(k, Hh, src[2])]) - lis
            dist = float(np.linalg.norm(v))
            order = abs(i) + abs(j) + abs(k)
            sv = np.array([v[1], -v[0], v[2]]) / dist          # 房间坐标 -> SOFA（x 前, y 左, z 上）
            pi = int(np.argmax(tab.sph_dirs @ sv))
            g = p["beta"] ** order * 1.5 / dist
            wall = (1 / np.sqrt(1 + (f / 7000) ** 2)) ** order  # 每次反射高频多吸收一点
            dl = dist / C * FS + tab.sph_tau[pi] + D0
            spec += g * wall * Hs[pi] * np.exp(-2j * np.pi * f[None, :] * dl[:, None] / FS)
        er[s] = wet * np.fft.irfft(spec * Hc, nfft)[:, :ner]

    # 混响尾巴：两路独立噪声，频率相关衰减 + 双耳扩散场相干度
    rng = np.random.default_rng(seed)
    nt = int(min(2.0, 1.4 * t60 * T60_SHAPE[1][0]) * FS)
    nseg = 512
    n1, n2 = rng.standard_normal((2, nt + nseg))
    ff, tt, Z1 = signal.stft(n1, FS, nperseg=nseg)
    _, _, Z2 = signal.stft(n2, FS, nperseg=nseg)
    t60f = t60 * np.interp(np.log(np.maximum(ff, 20)), np.log(T60_SHAPE[0]), T60_SHAPE[1])
    env = np.exp(-6.91 * tt[None, :] / t60f[:, None])
    gam = np.clip(np.sinc(2 * ff * 0.21 / C), 0, 1)
    phi = (np.arccos(gam) / 2)[:, None]
    zl = env * (np.cos(phi) * Z1 + np.sin(phi) * Z2)
    zr = env * (np.cos(phi) * Z1 - np.sin(phi) * Z2)
    tail = np.stack([signal.istft(z, FS, nperseg=nseg)[1][:nt] for z in (zl, zr)])
    tsec = np.arange(nt) / FS
    ramp = np.clip((tsec - 0.012) / (0.045 - 0.012), 0, 1)
    tail *= 0.5 - 0.5 * np.cos(np.pi * ramp)
    rcrit = 0.057 * math.sqrt(W * D * Hh / t60) * math.sqrt(2.0)   # 临界距离（说话人指向性 Q≈2）
    e_target = tab.e_ref_1m / rcrit ** 2 * math.exp(-13.8 * 0.03 / t60)
    tail *= wet * math.sqrt(e_target / np.mean(np.sum(tail ** 2, axis=1)))
    return Room(er, tail.astype(np.float32))


def render_room(out: np.ndarray, x: np.ndarray, az_b: np.ndarray, room: Room, gain_b: np.ndarray | None = None,
                el_b: np.ndarray | None = None) -> None:
    """早期反射按声源方向在 12 个扇区间插值（权重加在输入端），尾巴共用；结果累加进 out。
    gain_b：说话人音量的变化（忽远忽近），房间声跟着说话人的音量走，而不跟距离走。
    el_b：仰角。房间只在水平面上测过 12 个方向；声源升到 45° 以上时逐渐换成各方向的平均，到 80° 完全平均
    （从头顶绕过去、方位角翻到另一侧时不会突变）。"""
    if gain_b is not None:
        x = (x * np.concatenate([block_interp(gain_b, n0, min(len(x), n0 + (1 << 22)))
                                 for n0 in range(0, len(x), 1 << 22)])).astype(np.float32)
    s = np.mod(az_b, 360.0) / (360.0 / NSEC)
    k0 = np.floor(s).astype(int) % NSEC
    fr = s - np.floor(s)
    Wb = np.zeros((NSEC, len(az_b)))
    ar = np.arange(len(az_b))
    np.add.at(Wb, (k0, ar), 1 - fr)
    np.add.at(Wb, ((k0 + 1) % NSEC, ar), fr)
    if el_b is not None:
        up = smoothstep((np.abs(el_b) - 45.0) / 35.0)
        Wb = Wb * (1 - up) + up / NSEC
    N = len(x)
    xs = np.empty_like(x)
    for sec in range(NSEC):
        if Wb[sec].max() < 1e-4:
            continue
        for n0 in range(0, N, 1 << 22):
            n1 = min(N, n0 + (1 << 22))
            xs[n0:n1] = x[n0:n1] * block_interp(Wb[sec], n0, n1)
        conv_add(out, xs, room.er[sec])
    conv_add(out, x, room.tail)


# ----------------------------------------------------------------------------- 运动轨迹

def smoothstep(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0, 1)
    return s * s * (3 - 2 * s)


def arc_delta(a0: float, a1: float, via: str = "short") -> float:
    """从 a0 到 a1 的方位角增量（顺时针为正）。via: short / front / back。"""
    d = (a1 - a0 + 180.0) % 360.0 - 180.0
    if via == "short" or abs(d) < 120.0:   # 只有接近对穿（如左耳到右耳）时才需要选择从前面还是后面绕
        return d
    dcw = (a1 - a0) % 360.0
    cw_front = (a0 % 360.0) + dcw >= 360.0   # 顺时针这条弧是否经过正前方
    return dcw if (cw_front == (via == "front")) else dcw - 360.0


class PolarTrack:
    """以"停留 + 平滑移动"描述的极坐标轨迹，最后在块边界上采样。"""

    def __init__(self, t: np.ndarray, az0: float, r0: float):
        self.t = t
        self.az = np.full(len(t), az0, float)
        self.r = np.full(len(t), r0, float)
        self.lift = np.zeros(len(t))          # 绕两耳连线向上转多少度（从头顶绕时为 90）
        self.cur_az, self.cur_r = az0, r0

    def move(self, t0: float, t1: float, az1: float, r1: float, via: str = "short", pass_r: float | None = None,
             top: bool = False):
        """top：从头顶绕——方位角照"从前面绕"走，同时把整条路径绕两耳连线向上转 90°，
        于是声音从这只耳朵升到头顶、再落到另一只耳朵（起止各用四分之一的行程转上去 / 转回来）。"""
        if top:
            via = "front"
        d = arc_delta(self.cur_az, az1, via)
        a0, rr0 = self.cur_az, self.cur_r
        bump = 0.0
        if pass_r is not None:
            bump = max(0.0, pass_r - 0.5 * (rr0 + r1))
        m = self.t >= t0
        s = smoothstep((self.t[m] - t0) / max(t1 - t0, 1e-6))
        self.az[m] = a0 + d * s
        self.r[m] = (1 - s) * rr0 + s * r1 + bump * np.sin(np.pi * s)
        self.lift[m] = 90.0 * smoothstep(s / 0.25) * smoothstep((1 - s) / 0.25) if top else 0.0
        self.cur_az, self.cur_r = a0 + d, r1


def detect_gaps(x: np.ndarray, min_gap: float):
    """能量 VAD，返回不短于 min_gap 的停顿 [(t0, t1), ...]。"""
    hop = int(0.02 * FS)
    n = len(x) // hop
    if n < 10:
        return []
    x2 = x[:n * hop].reshape(n, hop)
    ms = np.concatenate([np.mean(np.square(x2[i:i + 50000], dtype=np.float64), axis=1) for i in range(0, n, 50000)])
    e = 10 * np.log10(ms + 1e-12)
    e = np.convolve(e, np.ones(3) / 3, mode="same")
    thr = max(np.percentile(e, 15) + 9.0, np.percentile(e, 98) - 38.0)
    quiet = e <= thr
    gaps, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * 0.02 >= min_gap:
                gaps.append((i * 0.02, j * 0.02))
            i = j
        else:
            i += 1
    return gaps


def pick_anchor(rng, side: str, kinds=None, probs=None):
    kinds = kinds or ["ear", "front", "back", "center"]
    probs = probs or [0.58, 0.22, 0.12, 0.08]
    kind = str(rng.choice(kinds, p=np.array(probs) / np.sum(probs)))
    (a0, a1), (r0, r1) = ANCHORS[kind]
    az = rng.uniform(a0, a1)
    r = math.exp(rng.uniform(math.log(r0), math.log(r1)))
    if side == "L" and kind != "center":
        az = -az
    return az % 360.0, r, kind


def traj_phrase(x, t, rng, a):
    T = float(t[-1])
    gaps = detect_gaps(x, a.min_gap)
    changes, last = [], 0.0
    for g0, g1 in gaps:
        c = 0.5 * (g0 + g1)
        if c - last >= a.min_hold and T - c >= 0.8:
            changes.append((c, float(np.clip(0.9 * (g1 - g0), 0.35, 1.1)), False))
            last = c
    # 太久没有停顿（长句、连续音）就在句中缓慢换个位置
    bounds = [0.0] + [c for c, _, _ in changes] + [T]
    forced = []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        k = int((b1 - b0) // a.max_hold)
        for q in range(1, k + 1):
            c = b0 + q * (b1 - b0) / (k + 1)
            forced.append((c, 1.6, True))
    changes = sorted(changes + forced)
    side = "L" if rng.random() < 0.5 else "R"
    az, r, kind = pick_anchor(rng, side, ["ear"], [1.0])
    tr = PolarTrack(t, az, r)
    for c, dur, is_forced in changes:
        if is_forced:   # 句中：同一侧换个近的位置，动作慢一点
            others = [k for k in ("ear", "front") if k != kind] or ["front"]
            az1, r1, kind = pick_anchor(rng, side, others, [1.0] * len(others))
            tr.move(c - dur / 2, c + dur / 2, az1, r1, "short")
        else:
            if rng.random() < 0.85:
                side = "L" if side == "R" else "R"
            az1, r1, kind = pick_anchor(rng, side)
            via = "front" if rng.random() < 0.75 else "back"
            sweep = abs(arc_delta(tr.cur_az, az1, via))
            tr.move(c - dur / 2, c + dur / 2, az1, r1, via, pass_r=0.26 + 0.12 * sweep / 180.0)
    return tr.az, tr.r, len(changes)


def traj_sweep(t, rng, a):
    """慢速穿梭：贴左耳停一会儿 → 经脸前慢慢移到右耳 → 停一会儿 → 再移回去，持续不断，移动全程伴着声音。
    贴耳位置略靠后（--ear-angle，Kitagawa 2005：脑后近处的声音"幻触"最强）。
    每次换边有 --approach 的概率改成"凑近"：先退到另一侧斜前方约 0.5 m（身体周围空间之外，Serino 2015），
    再用 2–3 s 慢慢贴到耳边（逼近的声音会增强触感与唤醒）。"""
    T = float(t[-1])
    r_ear = a.dist if a.dist_set else EAR_R
    ea = a.ear_angle
    tr = PolarTrack(t, 360.0 - ea, r_ear)
    rng_top = np.random.default_rng(a.seed + 404)   # 单独的随机数，不打乱原来的走位
    side, k, n_app = "L", 0, 0
    t0 = a.hold_time
    while t0 < T:
        side = "R" if side == "L" else "L"
        az1 = ea + rng.uniform(-6.0, 6.0)
        az1 = az1 if side == "R" else 360.0 - az1
        r1 = r_ear * rng.uniform(0.95, 1.05)
        if k > 0 and rng.random() < a.approach:
            azm = rng.uniform(30.0, 55.0)
            azm = azm if side == "R" else 360.0 - azm
            mv = a.move_time * rng.uniform(0.92, 1.08)
            tr.move(t0, t0 + mv, azm, rng.uniform(0.45, 0.6), "short")
            t0 += mv + rng.uniform(0.3, 0.6)
            loom = rng.uniform(2.2, 3.0)
            tr.move(t0, t0 + loom, az1, r1, "short")
            t0 += loom + a.hold_time * rng.uniform(1.0, 1.3)
            n_app += 1
        else:
            via = a.via if a.via != "alternate" else ("front" if k % 2 == 0 else "back")
            mv = a.move_time * rng.uniform(0.92, 1.08)
            top = a.over_top > 0 and rng_top.random() < a.over_top
            a.n_top = getattr(a, "n_top", 0) + int(top)
            tr.move(t0, t0 + mv, az1, r1, via, pass_r=a.pass_dist, top=top)
            t0 += mv + a.hold_time * rng.uniform(0.9, 1.1)
        k += 1
    a.lift_track = tr.lift
    return tr.az, tr.r, (k, n_app)


def traj_pingpong(t, rng, a):
    """与参考视频 BV1KPJDzHEVV 相同的节奏：3 秒一个来回，每只耳朵停约 1 秒，穿过中间约 0.5 秒。"""
    P = a.period or 3.0
    T = float(t[-1])
    r_ear = a.dist if a.dist_set else EAR_R
    tr = PolarTrack(t, 270.0, r_ear)
    k, t0 = 0, 0.33 * P
    while t0 < T:
        az1 = 90.0 if k % 2 == 0 else 270.0
        tr.move(t0, t0 + 0.17 * P, az1, r_ear, a.via if a.via != "alternate" else ("front" if k % 2 == 0 else "back"),
                pass_r=a.pass_dist)
        t0 += 0.5 * P
        k += 1
    return tr.az, tr.r, k


def traj_orbit(t, rng, a):
    P = a.period or 12.0
    r = a.radius
    az = (a.az if a.az_set else 0.0) + 360.0 * t / P * (-1 if a.ccw else 1)
    return az, np.full(len(t), r), int(t[-1] / P)


def traj_wander(t, rng, a):
    T = float(t[-1])
    tr = PolarTrack(t, rng.uniform(0, 360), math.exp(rng.uniform(math.log(0.3), math.log(0.9))))
    t0, n = rng.uniform(2.0, 5.0), 0
    while t0 < T:
        dur = rng.uniform(1.2, 2.6)
        tr.move(t0, t0 + dur, rng.uniform(0, 360), math.exp(rng.uniform(math.log(0.25), math.log(1.2))), "short")
        t0 += dur + rng.uniform(2.5, 6.0)
        n += 1
    return tr.az, tr.r, n


def traj_path(t, path: Path):
    keys = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = sorted(keys, key=lambda k: float(k["t"]))
    tr = PolarTrack(t, float(keys[0]["az"]) % 360, float(keys[0]["dist"]))
    for k0, k1 in zip(keys[:-1], keys[1:]):
        tr.move(float(k0["t"]), float(k1["t"]), float(k1["az"]) % 360, float(k1["dist"]),
                k1.get("via", "short"), k1.get("pass_dist"))
    return tr.az, tr.r, len(keys)


def speech_activity(x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """说话活动（0–1；词与词之间不到 0.3 s 的空隙算作还在说）和此刻音量（dB，相对说话时的中位数，
    约 0.3 s 平滑，停顿里保持），在块边界 t 上取值。"""
    hop = int(0.05 * FS)
    n = len(x) // hop
    if n < 20:
        return np.ones(len(t)), np.zeros(len(t))
    v = signal.sosfilt(signal.butter(2, [150, 5000], "band", fs=FS, output="sos"), x)
    e = 10 * np.log10(np.mean(np.square(v[:n * hop].reshape(n, hop), dtype=np.float64), axis=1) + 1e-12)
    act = e > max(np.percentile(e, 97) - 30.0, np.percentile(e, 20) + 6.0)
    if act.sum() < 10:
        return np.ones(len(t)), np.zeros(len(t))
    a = act.astype(float)
    idx = np.nonzero(act)[0]
    for i0, g in zip(idx[:-1], np.diff(idx)):
        if 1 < g <= 6:
            a[i0 + 1:i0 + g] = 1.0
    a = np.clip(signal.sosfiltfilt(signal.butter(2, 2.0, fs=1.0 / 0.05, output="sos"), a), 0.0, 1.0)
    d = e - np.median(e[act])
    lf = np.empty(n)
    cur = 0.0
    for i in range(n):
        if act[i]:
            cur = d[i]
        lf[i] = cur
    lf = signal.sosfiltfilt(signal.butter(2, 1.5, fs=1.0 / 0.05, output="sos"), lf)
    tc = (np.arange(n) + 0.5) * hop / FS
    return np.interp(t, tc, a), np.interp(t, tc, lf)


def speech_clock(t: np.ndarray, act: np.ndarray, idle: float) -> np.ndarray:
    """说话时钟：说话时按真实时间走，停顿时只走 idle 倍。用它驱动晃动和凑近 / 退开，动作就发生在
    说话的时候，停顿里几乎不动，下一句从上一句停下的位置接着说，不会"跳"一下
    （Hadar 等 1983：说话时 89.9% 的时间头在动，不说话时只有 12.8%）。"""
    g = idle + (1.0 - idle) * act
    return np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(t))])


def add_micro(az, r, t, rng, sigma, clock=None, amp=None, act=None):
    """人体自然晃动：笛卡尔坐标里叠加几条慢速正弦（0.06–0.45 Hz），默认 RMS 1.2 cm。
    clock：说话时钟（停顿时几乎不动）；amp：动作幅度随音量变化（声音大时动得多，Hadar 等 1983），
    此时按说话时段（act）把 RMS 定为 sigma。两者都不给时与旧版完全相同。"""
    X = r * np.sin(np.radians(az))
    Y = r * np.cos(np.radians(az))
    if sigma > 0:
        tt = t if clock is None else clock
        for arr in (X, Y):
            fq, ph, am = rng.uniform(0.06, 0.45, 4), rng.uniform(0, 2 * np.pi, 4), rng.uniform(0.5, 1.0, 4)
            s = (am[:, None] * np.sin(2 * np.pi * fq[:, None] * tt[None, :] + ph[:, None])).sum(0)
            if amp is None:
                arr += s / (s.std() + 1e-12) * sigma
            else:
                s = s * amp
                ref = s[act > 0.5] if act is not None and np.count_nonzero(act > 0.5) > 10 else s
                arr += s / (ref.std() + 1e-12) * sigma
    return np.degrees(np.arctan2(X, Y)) % 360.0, np.clip(np.hypot(X, Y), R_MIN, R_MAX)


def speech_loudness(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """说话人此刻的音量（dB，相对说话时的中位数），在块边界 t 上取值，供"声音大就退远一点"使用。
    语音频段 50 ms 能量，只在说话时更新，停顿里保持不动。再做"提前让开、慢慢回来"的包络：
    响的字到来前约 0.8 s 就开始往后让（每 6 dB），之后只在说话时慢慢凑回来（约 2.5 s / 6 dB），
    不会贴着耳朵炸出来，不会一下子弹回耳边，也不会在停顿里偷偷挪位置。"""
    hop = int(0.05 * FS)
    n = len(x) // hop
    if n < 20:
        return np.zeros(len(t))
    v = signal.sosfilt(signal.butter(2, [150, 5000], "band", fs=FS, output="sos"), x)
    e = 10 * np.log10(np.mean(np.square(v[:n * hop].reshape(n, hop), dtype=np.float64), axis=1) + 1e-12)
    act = e > max(np.percentile(e, 97) - 30.0, np.percentile(e, 20) + 6.0)
    if act.sum() < 10:
        return np.zeros(len(t))
    k = np.ones(9)
    num = np.convolve(np.where(act, 10 ** (e / 10), 0.0), k, "same")
    den = np.convolve(act.astype(float), k, "same")
    d = 10 * np.log10(np.maximum(num, 1e-30) / np.maximum(den, 1e-9))
    d -= np.median(d[act])
    out = np.empty(n)
    cur = 0.0
    for i in range(n):
        if act[i]:
            cur = d[i]
        out[i] = cur
    att, rel = 6.0 * 0.05 / 0.8, 6.0 * 0.05 / 2.5        # 每帧最多变化多少 dB：提前让开快、回来慢
    for i in range(1, n):
        out[i] = max(out[i], out[i - 1] - (rel if act[i] else 0.0))
    for i in range(n - 2, -1, -1):
        out[i] = max(out[i], out[i + 1] - att)
    out -= np.median(out[act])
    out = signal.sosfiltfilt(signal.butter(2, 0.6, fs=1.0 / 0.05, output="sos"), out)
    return np.interp(t, (np.arange(n) + 0.5) * hop / FS, out)


def sway_gestures(t: np.ndarray, rng, period: float) -> tuple[np.ndarray, int]:
    """有节奏又不固定的"凑近—退开"：贴着耳朵停一会儿 → 慢慢往后让 → 稍停 → 再慢慢凑回来，循环往复。
    每段时长和退开的深浅都随机（偶尔只让一点，偶尔退得更远），凑回来比退开慢（逼近更引人注意，
    Neuhoff 2016；进入身体周围空间时触觉增强，Canzoneri 2012）；贴近时至少停 2 s（Koumura 等 2021：
    战栗感比声音滞后约 2 s）。一个来回约 period 秒；t 用说话时钟时，停顿会让动作停住，实际更长
    （KU100 实录 2–23 s）。返回离耳距离相对基准的倍数（log2，倍频程）和退开次数。"""
    T = float(t[-1])
    sc = max(period, 1.0) / 8.0
    u0 = rng.uniform(-0.3, 0.1)
    keys = [(0.0, u0)]
    tc, n = rng.uniform(0.3, 2.5) * sc, 0
    keys.append((tc, u0))
    while tc < T + 5.0:
        q = rng.random()
        depth = rng.uniform(0.4, 0.9) if q < 0.12 else rng.uniform(1.0, 2.0) if q < 0.8 else rng.uniform(2.0, 3.2)
        slow = 0.8 + 0.2 * depth
        tc += rng.uniform(1.2, 2.6) * sc * slow
        keys.append((tc, depth))
        tc += rng.uniform(0.3, 1.8) * sc
        keys.append((tc, depth * rng.uniform(0.85, 1.05)))
        tc += rng.uniform(1.6, 3.4) * sc * slow
        u_in = rng.uniform(-0.4, 0.1)
        keys.append((tc, u_in))
        tc += max(2.0, min(7.0, math.exp(rng.normal(math.log(2.6), 0.4))) * sc)
        keys.append((tc, float(np.clip(u_in + rng.uniform(-0.15, 0.15), -0.45, 0.2))))
        n += 1
    kt = np.array([k[0] for k in keys])
    kv = np.array([k[1] for k in keys])
    i = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 2)
    u = smoothstep((t - kt[i]) / np.maximum(kt[i + 1] - kt[i], 1e-6))
    return kv[i] + (kv[i + 1] - kv[i]) * u, int(np.sum(kt[2::4] < T))


def updown_gestures(t: np.ndarray, rng, period: float, rnd: float) -> tuple[np.ndarray, int]:
    """自动上下：嘴在耳朵上方 / 下方之间慢慢换位置——停一会儿 → 慢慢移到另一边 → 停一会儿 → 再移回来。
    rnd=0 时节奏和高低都固定（上、下各到幅度的 75%，一个来回 period 秒）；rnd=1 时停留和移动的时长、
    每次到多高都随机，偶尔连续两次停在同一侧。返回 −1…1 的曲线和换位次数。t 用说话时钟时停顿里不动。"""
    T = float(t[-1])
    sc = max(period, 1.0) / 6.4
    rho = float(np.clip(rnd, 0.0, 1.0))

    def mix(mid: float, lo: float, hi: float) -> float:
        return mid + (rng.uniform(lo, hi) - mid) * rho

    v = rng.uniform(-0.5, 0.5) * rho
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    keys = [(0.0, v)]
    tc, n = 0.0, 0
    while tc < T + 5.0:
        tc += mix(1.3, 0.5, 2.1) * sc
        keys.append((tc, v))
        if rng.random() >= 0.2 * rho:
            sgn = -sgn
        v = sgn * mix(0.75, 0.35, 1.0)
        tc += mix(1.9, 1.3, 2.5) * sc
        keys.append((tc, v))
        n += 1
    kt = np.array([k[0] for k in keys])
    kv = np.array([k[1] for k in keys])
    i = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 2)
    u = smoothstep((t - kt[i]) / np.maximum(kt[i + 1] - kt[i], 1e-6))
    return kv[i] + (kv[i + 1] - kv[i]) * u, int(np.sum(kt[2::2] < T))


def lift_tilt(az: np.ndarray, r: np.ndarray, lift: np.ndarray, tilt: np.ndarray):
    """把水平面上的位置（方位角 az、距离 r）抬离水平面，返回 (方位角, 仰角, 距离)。
      1. lift：整个位置绕两耳连线向上转这么多度（从头顶绕时用；正前方转 90° 就到了头顶）。
      2. tilt：以"耳朵"为支点向上（正）/ 向下（负）倾这么多度，嘴到耳朵的距离不变——贴耳时就是嘴绕着
         耳朵上下挪。支点随声源的侧向程度在两耳之间滑动：声源在正侧方时支点就是那只耳朵，在正前 / 正后方
         时是头中心（这时就是普通的仰角）。从头顶绕到最高处时 tilt 不起作用。
    lift 和 tilt 都为 0 的位置原样返回，仰角为 0。"""
    a = np.radians(az)
    x, y = r * np.sin(a), r * np.cos(a)                    # x 朝右，y 朝前
    rho = np.radians(lift)
    y2, z2 = y * np.cos(rho), y * np.sin(rho)
    cx = HEAD_A * x / np.maximum(r, 1e-9)
    vx = x - cx
    vh = np.hypot(vx, y2)
    d = np.hypot(vh, z2)
    phi = np.clip(np.arctan2(z2, vh) + np.radians(tilt) * np.cos(rho), -0.5 * np.pi, 0.5 * np.pi)
    ux, uy = vx / np.maximum(vh, 1e-9), y2 / np.maximum(vh, 1e-9)
    px, py, pz = cx + ux * d * np.cos(phi), uy * d * np.cos(phi), d * np.sin(phi)
    rr = np.sqrt(px * px + py * py + pz * pz)
    el = np.degrees(np.arcsin(np.clip(pz / np.maximum(rr, 1e-9), -1.0, 1.0)))
    az2 = np.degrees(np.arctan2(px, py)) % 360.0
    flat = (lift == 0) & (tilt == 0)
    return np.where(flat, az, az2), np.where(flat, 0.0, el), np.where(flat, r, np.clip(rr, R_MIN, R_MAX))


def apply_sway(x: np.ndarray, t: np.ndarray, r: np.ndarray, rng, a,
               clock: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """忽远忽近：改变嘴到头表面的距离（方位不变）= 随机的凑近 / 退开 + 跟着说话音量（每响 6 dB 退远一倍，
    轻声时更贴近）。只在耳边附近生效：基准距离 0.2 m 以内全量，0.45 m 以外不晃（"从远处凑近"保持原样）。
    clock：说话时钟，随机的凑近 / 退开只在说话时进行。返回新的距离、log2 倍数曲线、退开次数。"""
    w = np.zeros(len(t))
    n = 0
    if a.sway > 0:
        g, n = sway_gestures(t if clock is None else clock, rng, a.sway_period)
        w += a.sway * g
    if a.sway_follow > 0:
        w += a.sway_follow * np.clip(speech_loudness(x, t), -3.0, 12.0) / 6.0
    w = signal.sosfiltfilt(signal.butter(2, 0.6, fs=FS / BLOCK, output="sos"), w)   # 人的动作是平滑的
    near = np.clip((0.45 - r) / (0.45 - 0.2), 0.0, 1.0)
    d0 = np.maximum(r - HEAD_A, 0.005)
    d = np.minimum(d0 * 2 ** (w * near), np.maximum(d0, 0.35))
    return np.clip(HEAD_A + d, R_MIN, R_MAX), w, n


# ----------------------------------------------------------------------------- 音频读写与响度

def find_tool(name: str) -> str | None:
    exe = name + (".exe" if os.name == "nt" else "")
    p = shutil.which(name)
    if p:
        return p
    for cand in (Path.home() / "ai/qwenasr/ffmpeg" / exe, Path("E:/ai/qwenasr/ffmpeg") / exe):
        if cand.exists():
            return str(cand)
    return None


def probe_channels(path: Path, ffmpeg: str) -> int:
    ffprobe = find_tool("ffprobe")
    if ffprobe:
        out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                              "stream=channels", "-of", "json", str(path)], capture_output=True, text=True, check=True)
        return int(json.loads(out.stdout)["streams"][0]["channels"])
    err = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Audio:.*?, (\d+) Hz, ([^,]+)", err)
    if not m:
        raise RuntimeError("读不出声道数")
    lay = m.group(2).strip()
    return 1 if lay == "mono" else 2 if lay == "stereo" else int(re.match(r"(\d+)", lay).group(1))


def load_audio(path: Path, start: float | None, dur: float | None) -> np.ndarray:
    """返回 (声道, 样本) float32，48 kHz。"""
    ffmpeg = find_tool("ffmpeg")
    if ffmpeg:
        ch = probe_channels(path, ffmpeg)
        cmd = [ffmpeg, "-v", "error"]
        if start:
            cmd += ["-ss", str(start)]
        cmd += ["-i", str(path)]
        if dur:
            cmd += ["-t", str(dur)]
        cmd += ["-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(FS), "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.float32).reshape(-1, ch).T   # 只读视图，选声道后即释放
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    data = data.astype(np.float32) / (np.iinfo(data.dtype).max if data.dtype.kind == "i" else 1.0)
    data = np.atleast_2d(data.T) if data.ndim > 1 else data[None, :]
    s0 = int((start or 0) * sr)
    data = data[:, s0:s0 + int(dur * sr) if dur else None]
    if sr != FS:
        g = math.gcd(FS, sr)
        data = signal.resample_poly(data, FS // g, sr // g, axis=1).astype(np.float32)
    return data


def save_audio(path: Path, y: np.ndarray, sr: int) -> None:
    ffmpeg = find_tool("ffmpeg")
    ext = path.suffix.lower()
    if ffmpeg:
        codec = {".wav": ["-c:a", "pcm_s24le"], ".flac": ["-c:a", "flac"], ".m4a": ["-c:a", "aac", "-b:a", "256k"],
                 ".aac": ["-c:a", "aac", "-b:a", "256k"], ".mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
                 ".ogg": ["-c:a", "libopus", "-b:a", "192k"], ".opus": ["-c:a", "libopus", "-b:a", "192k"]}.get(ext, [])
        cmd = [ffmpeg, "-v", "error", "-y", "-f", "f32le", "-ar", str(FS), "-ac", "2", "-i", "-"]
        if sr != FS:
            cmd += ["-af", f"aresample={sr}"]
        proc = subprocess.Popen(cmd + codec + [str(path)], stdin=subprocess.PIPE)
        for n0 in range(0, y.shape[1], 1 << 20):
            proc.stdin.write(np.ascontiguousarray(y[:, n0:n0 + (1 << 20)].T).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg 写文件失败")
        return
    if ext != ".wav":
        raise RuntimeError("没有找到 ffmpeg，只能输出 .wav")
    from scipy.io import wavfile
    wavfile.write(path, FS, np.ascontiguousarray(y.T))


def integrated_lufs(y: np.ndarray) -> float:
    """ITU-R BS.1770-4 积分响度（48 kHz K 加权，绝对/相对门限）。"""
    sos = np.array([[1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585],
                    [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]])
    hop = FS // 10
    zi = np.zeros((2, y.shape[0], 2))
    subs = []
    step = hop * 600
    for n0 in range(0, y.shape[1], step):
        seg = y[:, n0:n0 + step].astype(np.float64)
        z, zi = signal.sosfilt(sos, seg, axis=1, zi=zi)
        k = z.shape[1] // hop
        subs.append((z[:, :k * hop] ** 2).reshape(z.shape[0], k, hop).mean(axis=2).sum(axis=0))
    s = np.concatenate(subs)
    if len(s) < 4:
        return -70.0
    blk = np.convolve(s, np.ones(4) / 4, mode="valid")
    lk = -0.691 + 10 * np.log10(blk + 1e-20)
    g1 = blk[lk > -70]
    if not len(g1):
        return -70.0
    gate = -0.691 + 10 * np.log10(g1.mean()) - 10
    g2 = blk[(lk > -70) & (lk > gate)]
    return float(-0.691 + 10 * np.log10(g2.mean()))


def limit(y: np.ndarray, ceil: float, win_ms: float = 6.0) -> tuple[np.ndarray, float]:
    """前瞻峰值限制：最小值滤波 + 两次滑动平均，保证增益在峰值处不高于所需值，起落约 2×win_ms。"""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    pk = np.abs(y).max(axis=0)
    g = np.minimum(1.0, ceil / np.maximum(pk, 1e-12)).astype(np.float32)
    if g.min() >= 1.0:
        return y, 0.0
    h = int(win_ms / 1000 * FS)
    g = minimum_filter1d(g, 4 * h + 1)
    g = uniform_filter1d(uniform_filter1d(g, 2 * h + 1), 2 * h + 1)
    y *= g[None, :]
    return y, float(-20 * np.log10(g.min()))


def true_peak(y: np.ndarray) -> float:
    pk = 0.0
    step = 1 << 20
    for n0 in range(0, y.shape[1], step):
        seg = y[:, max(0, n0 - 64):n0 + step + 64]
        pk = max(pk, float(np.abs(signal.resample_poly(seg, 4, 1, axis=1)).max()))
    return pk


# ----------------------------------------------------------------------------- 主流程

def select_channel(x: np.ndarray, how: str) -> np.ndarray:
    if x.shape[0] == 1:
        return x[0]
    if how == "mix":
        return x[:2].mean(axis=0)
    if how in ("left", "right"):
        return x[0 if how == "left" else 1]
    rms = np.sqrt(np.mean(x[:2].astype(np.float64) ** 2, axis=1))
    k = int(np.argmax(rms))
    other = 1 - k
    corr = float(np.dot(x[0], x[1]) / (np.linalg.norm(x[0]) * np.linalg.norm(x[1]) + 1e-12))
    log(f"立体声输入：左 {20 * np.log10(rms[0] + 1e-12):.1f} dB，右 {20 * np.log10(rms[1] + 1e-12):.1f} dB，"
        f"相关 {corr:.2f} -> 取{'左' if k == 0 else '右'}声道")
    if rms[other] > 0.5 * rms[k] and corr < 0.5:
        log("  注意：两个声道内容差别较大（像是已经做过左右声像），只取一个会丢掉另一侧的内容；"
            "如需保留可加 --channel mix")
    return x[k]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="输出文件（.wav/.flac/.m4a/.mp3），默认 输入名_asmr.wav")
    ap.add_argument("--mode", default="sweep",
                    choices=["sweep", "pingpong", "phrase", "orbit", "wander", "static", "path"],
                    help="sweep=从左耳慢慢移到右耳再移回（默认）；pingpong=参考视频的节奏（3 秒一个来回）；"
                         "phrase=只在说话停顿时换耳朵；orbit=绕头转圈；wander=在身边走动；static=固定位置；"
                         "path=按 JSON 关键帧")
    ap.add_argument("--move-time", type=float, default=2.5, help="sweep：从一只耳朵移到另一只耳朵用多少秒")
    ap.add_argument("--hold-time", type=float, default=2.5, help="sweep：每只耳朵边停留多少秒")
    ap.add_argument("--via", default="front", choices=["front", "back", "alternate"],
                    help="sweep/pingpong：从脸前绕、从脑后绕，还是前后交替")
    ap.add_argument("--pass-dist", type=float, default=0.2, help="sweep/pingpong：穿过中间时离头中心的距离（米）")
    ap.add_argument("--ear-angle", type=float, default=100.0, help="sweep：贴耳的方位角（90=正侧方，100=略靠后）")
    ap.add_argument("--approach", type=float, default=0.25, help="sweep：每次换边改成\"从远处凑近\"的概率（0=关闭）")
    ap.add_argument("--style", default="ear", choices=list(STYLES),
                    help="ear=贴耳加强（默认：嘴离耳约 1.7 cm，远耳额外压低 32 dB）；"
                         "ku100=像真实 KU100 实录（远耳不额外压低、离耳约 6 cm、真实房间、KU100 原机音色）")
    ap.add_argument("--channel", default="auto", choices=["auto", "left", "right", "mix"], help="立体声输入取哪个声道")
    ap.add_argument("--az", type=float, help="static/orbit 的方位角（度，0 前 90 右 180 后 270 左）")
    ap.add_argument("--dist", type=float, help="sweep/pingpong 贴耳 / static 的距离（米，到头中心；默认 ear 0.105、ku100 0.15）")
    ap.add_argument("--radius", type=float, default=0.35, help="orbit 半径（米）")
    ap.add_argument("--period", type=float, help="orbit 一圈 / pingpong 一个来回的秒数")
    ap.add_argument("--ccw", action="store_true", help="orbit 逆时针转")
    ap.add_argument("--path", type=Path, help='path 模式的关键帧 JSON：[{"t":0,"az":270,"dist":0.15}, ...]')
    ap.add_argument("--min-gap", type=float, default=0.3, help="phrase：多长的停顿可以换位置（秒）")
    ap.add_argument("--min-hold", type=float, default=1.5, help="phrase：一个位置至少停留多久（秒）")
    ap.add_argument("--max-hold", type=float, default=9.0, help="phrase：一个位置最多停留多久，超过就在句中慢慢挪（秒）")
    ap.add_argument("--room", default=None, choices=["none", "real", "treated", "bedroom", "living"],
                    help="房间声学：none=无（ear 风格默认）；real=KU100 实测真实房间（卧室混响时间，ku100 风格默认）；"
                         "treated/bedroom/living=合成房间")
    ap.add_argument("--wet", type=float, default=-3.0, help="房间反射/混响电平偏移（dB）")
    ap.add_argument("--isolation", type=float, default=None,
                    help="人头麦单耳独占感：贴耳时远侧耳额外压低多少 dB（穿过中间时自动恢复，0=纯物理；默认 ear 32、ku100 0）")
    ap.add_argument("--presence", type=float, default=3.0, help="贴耳时近耳的临场感（3.5 kHz）和空气感（9 kHz 以上），dB")
    ap.add_argument("--ku100-eq", action="store_true", help="近耳音色改用 KU100 实测标定的校正曲线（高频更收）")
    ap.add_argument("--far-lf-cut", type=float, default=None,
                    help="贴耳时远耳 320 Hz 以下再压低多少 dB（加大低频两耳差；默认 ear 12、ku100 0）")
    ap.add_argument("--dark", type=float, default=0.0, help="贴耳时近耳 5 kHz 附近压低多少 dB（更\"暗\"的音色）")
    ap.add_argument("--warmth", type=float, default=6.0, help="凑近时两耳共同的低频温暖感（dB，0 关闭）")
    ap.add_argument("--ear-warmth", type=float, default=5.0,
                    help="贴耳时只给近耳加的低频（dB，0 关闭；KU100 实测近耳 250 Hz 比居中高 6–11 dB）")
    ap.add_argument("--micro", type=float, default=1.0, help="自然晃动幅度倍数（0 关闭，1 约 1.2 cm）")
    ap.add_argument("--sway", type=float, default=1.0,
                    help="忽远忽近：说话人在耳边随机地凑近、退开（0=关闭；1=标准：多数时间贴耳 1–2 cm，"
                         "每隔几秒往后让到 3.5–7 cm，偶尔 7–16 cm）")
    ap.add_argument("--sway-follow", type=float, default=1.0,
                    help="跟着说话音量：声音大时往后让、轻声时更贴近，听着更舒服（0=关闭；1=每响 6 dB 退远一倍）")
    ap.add_argument("--sway-level", type=float, default=0.45,
                    help="退开时耳边音量保留物理衰减的比例（0=音量不变，1=完全按距离衰减；KU100 实录约 0.44）")
    ap.add_argument("--sway-period", type=float, default=4.0,
                    help="说话时一次凑近—退开的大致时长（秒，随机，贴近至少停 2 s，所以实际约 4–7 s；"
                         "停顿时动作会停住，按真实时间算更长）")
    ap.add_argument("--height", type=float, default=0.0,
                    help="高度：嘴相对耳朵向上（正）/ 向下（负）偏多少度，离耳朵的距离不变（0=与耳朵齐平）")
    ap.add_argument("--updown", type=float, default=25.0,
                    help="自动上下：嘴在耳朵上方 / 下方之间慢慢换位置的幅度（度，0=关闭）")
    ap.add_argument("--updown-period", type=float, default=9.0, help="自动上下：说话时一个来回的大致秒数")
    ap.add_argument("--updown-random", type=float, default=0.6,
                    help="自动上下的随机程度（0=节奏和高低固定，1=时长和高低都随机）")
    ap.add_argument("--over-top", type=float, default=0.0,
                    help="sweep：每次换耳改成\"从头顶绕过去\"的概率（0=只从前 / 后绕，1=每次都从头顶绕）")
    ap.add_argument("--pause-motion", type=float, default=0.13,
                    help="停顿时晃动和凑近 / 退开的速度（相对说话时）：0.13=停顿里几乎不动，下一句从原位置接着说"
                         "（Hadar 1983）；1=一直在动（旧版）")
    ap.add_argument("--level-follow", type=float, default=None,
                    help="响度随位置变化的程度：1=物理真实（贴耳很响、到脸前会弱很多），0=处处一样响。"
                         "默认 sweep 0.2、pingpong 0.1、orbit 0.3、phrase 0.5、其他 1")
    ap.add_argument("--lufs", type=float, default=-16.0, help="目标积分响度（LUFS）")
    ap.add_argument("--peak", type=float, default=-1.0, help="真峰值上限（dBTP）")
    ap.add_argument("--max-gr", type=float, default=8.0, help="峰值限制器最多压多少 dB（0 = 不压，只整体降低）")
    ap.add_argument("--no-dfeq", action="store_true", help="同 --headphone native")
    ap.add_argument("--headphone", default=None,
                    help="回放补偿：diffuse=通用扩散场（ear 风格默认）；native=KU100 原机音色（像真实 KU100 录音，ku100 风格默认）；"
                         "或耳机型号（如 HD600、DT770、K701、M50），用 TH Köln 在 KU100 上实测的补偿；list=列出型号")
    ap.add_argument("--breath", type=float, default=6.0,
                    help="贴耳气声增强：嘴贴近耳朵时，嗓音里轻的送气声 / 气息（2.5 kHz 以上）最多提高多少 dB；"
                         "响的齿音不提（0=关闭）")
    ap.add_argument("--puff", type=float, default=1.0,
                    help="气流扑耳：送气时气流扑到贴着的那只耳朵上的低频脉冲（50–130 Hz，由嗓音自己的送气声导出，"
                         "不加噪声；0=关闭，1=标准）")
    ap.add_argument("--concha", type=float, default=4.0,
                    help="耳甲共鸣：贴耳时近耳 4.3 kHz 的耳甲腔共振，dB（KU100 贴耳实录的共有特征；0=关闭）")
    ap.add_argument("--mouth", type=float, default=2.0,
                    help="嘴的大小（cm）：嘴是一个约 2 cm 宽的开口而不是一个点，贴耳时耳廓的窄谷被平均变浅（0=点声源）")
    ap.add_argument("--far-shadow", type=float, default=5.0,
                    help="贴耳时远耳 6 kHz 以上再压低多少 dB（近处声源在远耳的高频头影比球头模型深；0=关闭）")
    ap.add_argument("--soft-near", type=float, default=0.5,
                    help="贴耳轻声：凑到耳边时嗓音均衡成\"放轻声音\"（同 --soft-voice，但只在贴耳时），退开时恢复（0=关闭）")
    ap.add_argument("--pinna", default="ku100",
                    help="耳廓选择：ku100（默认，人工头自己的耳廓）或真人耳廓 A–G（HUTUBS 实测，挑最像自己耳朵的那只）；list 列出")
    ap.add_argument("--soft-voice", type=float, default=0.0,
                    help="轻声化：按 ANSI S3.5 说话力度频谱，把嗓音往\"压低声音说\"的方向均衡（0=关闭，1=低一档力度）")
    ap.add_argument("--deess", type=float, default=0.0, help="去齿音：齿音最多压低多少 dB（0=关闭，建议 4–8）")
    ap.add_argument("--start", type=float, help="只处理从这里开始（秒）")
    ap.add_argument("--duration", type=float, help="只处理这么长（秒）")
    ap.add_argument("--sr", type=int, default=FS, help="输出采样率")
    ap.add_argument("--seed", type=int, default=1, help="随机种子（换一个就换一套走位）")
    ap.add_argument("--save-trajectory", type=Path, help="把轨迹写成 CSV（时间, 方位角, 距离, 仰角）")
    ap.add_argument("--hrtf-dir", type=Path, default=HRTF_DIR)
    a = ap.parse_args(argv)
    for k, v in STYLES[a.style].items():      # 没有手动指定的参数按风格取默认值
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.az_set, a.dist_set = a.az is not None, a.dist is not None
    if a.level_follow is None:
        a.level_follow = {"sweep": 0.2, "pingpong": 0.1, "orbit": 0.3, "phrase": 0.5}.get(a.mode, 1.0)
    if a.mode == "path" and not a.path:
        ap.error("--mode path 需要 --path")

    t_start = time.time()
    if a.pinna.strip().lower() == "list":
        print("ku100（默认，KU100 人工头自己的耳廓）")
        if PINNA_FILE.exists():
            z = np.load(PINNA_FILE)
            print("\n".join(f"{n}（真人，HUTUBS pp{sid}）" for n, sid in zip(z["names"], z["subjects"])))
        return 0
    if a.headphone.strip().lower() == "list":
        print("diffuse（默认，通用扩散场补偿）\nnative（不补偿，KU100 原机音色）")
        print("\n".join(f.stem for f in headphone_models(a.hrtf_dir)))
        return 0
    tab = load_tables(a.hrtf_dir)
    x_all = load_audio(a.input, a.start, a.duration)
    if x_all.shape[1] < FS // 10:
        ap.error("输入太短")
    x = select_channel(x_all, a.channel)
    del x_all
    x = signal.sosfilt(signal.butter(2, 40, "high", fs=FS, output="sos"), x).astype(np.float32)
    x /= max(float(np.abs(x).max()), 1e-9)
    if a.deess > 0:
        x = deess(x, a.deess)
    if a.soft_voice > 0:
        x = soften_voice(x, a.soft_voice)
    x /= max(float(np.abs(x).max()), 1e-9)
    n_sig = len(x)
    pb = playback_filter(tab, a.hrtf_dir, "native" if a.no_dfeq else a.headphone)
    if a.room == "none":
        room = None
    elif a.room == "real":
        room = build_room_measured(tab, a.wet, pb=pb)
    else:
        room = build_room(tab, a.room, a.wet, pb=pb)
    pad = (max(room.er.shape[2], room.tail.shape[1]) if room else 0) + 4 * BLOCK
    x = np.concatenate([x, np.zeros(pad, np.float32)])
    nb = -(-len(x) // BLOCK)
    t = np.arange(nb + 1) * BLOCK / FS

    clock = amp = act_b = None
    if a.pause_motion < 1.0:      # 动作跟着说话：说话时动、停顿里几乎不动，幅度随音量
        act_b, lf_b = speech_activity(x, t)
        clock = speech_clock(t, act_b, max(a.pause_motion, 0.0))
        amp = np.clip(10 ** (lf_b / 20), 0.5, 1.6)
    # sweep / orbit / wander 按说话时钟走：换耳、绕行、走动都发生在说话的时候（停顿里不会"跳"过去）；
    # pingpong 复刻参考视频的固定节奏、phrase 专门在停顿处换位、path 按给定时间，仍按真实时间
    tm = clock if clock is not None else t
    rng = np.random.default_rng(a.seed)
    if a.mode == "sweep":
        az, r, (n, n_app) = traj_sweep(tm, rng, a)
        desc = f"{n} 次左右穿梭（其中 {n_app} 次从远处凑近；移动 {a.move_time:g} s，停留 {a.hold_time:g} s）"
    elif a.mode == "phrase":
        az, r, n = traj_phrase(x[:n_sig], t, rng, a)
        desc = f"{n} 次换位"
    elif a.mode == "pingpong":
        az, r, n = traj_pingpong(t, rng, a)
        desc = f"{n} 次左右切换"
    elif a.mode == "orbit":
        az, r, n = traj_orbit(tm, rng, a)
        desc = f"约 {n} 圈"
    elif a.mode == "wander":
        az, r, n = traj_wander(tm, rng, a)
        desc = f"{n} 次走动"
    elif a.mode == "static":
        az, r, desc = np.full(len(t), a.az if a.az_set else 270.0), np.full(len(t), a.dist or 0.15), "固定"
    else:
        az, r, n = traj_path(t, a.path)
        desc = f"{n} 个关键帧"
    az, r = add_micro(az, r, t, rng, 0.012 * a.micro, clock, amp, act_b)
    log(f"轨迹：{a.mode}，{desc}；距离 {r.min():.2f}–{r.max():.2f} m"
        + (f"；动作跟着说话（说话占 {np.mean(act_b > 0.5) * 100:.0f}%，停顿时速度 {a.pause_motion:g}）" if clock is not None else ""))
    r_base, g_sway = r, None
    swaying = a.sway > 0 or a.sway_follow > 0
    if swaying:
        r, _, n_sw = apply_sway(x, t, r_base, np.random.default_rng(a.seed + 202), a, clock)
    # 上下：固定高度 + 自动上下（含一点上下的晃动）+ 从头顶绕。都没有时 el 为 None，后面与不带高度时完全相同
    el = base_pos = None
    tilt = np.full(len(t), float(a.height))
    lift = getattr(a, "lift_track", None)
    lift = np.zeros(len(t)) if lift is None else lift
    if a.updown > 0:
        rng_ud = np.random.default_rng(a.seed + 303)
        g_ud, n_ud = updown_gestures(tm, rng_ud, a.updown_period, a.updown_random)
        fq, ph, am = rng_ud.uniform(0.06, 0.45, 4), rng_ud.uniform(0, 2 * np.pi, 4), rng_ud.uniform(0.5, 1.0, 4)
        wob = (am[:, None] * np.sin(2 * np.pi * fq[:, None] * tm[None, :] + ph[:, None])).sum(0) / np.sqrt((am ** 2).sum() / 2)
        tilt += a.updown * (g_ud + 0.15 * wob * (amp if amp is not None else 1.0))
    tilt = np.clip(tilt, -80.0, 85.0)
    if np.any(tilt != 0) or np.any(lift != 0):
        az_h = az
        az, el, r = lift_tilt(az_h, r, lift, tilt)
        base_pos = lift_tilt(az_h, r_base, lift, tilt)
        side = (np.abs(np.sin(np.radians(az_h))) > 0.9) & (lift == 0)
        if not side.any():
            side = np.ones(len(t), bool)
        log(f"上下：贴耳时嘴相对耳朵 {np.percentile(tilt[side], 5):+.0f}°…{np.percentile(tilt[side], 95):+.0f}°"
            f"（上为正；仰角 {el.min():+.0f}°…{el.max():+.0f}°）"
            + (f"，自动上下 {n_ud} 次" if a.updown > 0 else "")
            + (f"，从头顶绕 {a.n_top} 次" if getattr(a, "n_top", 0) else ""))
    if swaying:
        g_sway = sway_gain(tab, az, r, base_pos[2] if el is not None else r_base, a.sway_level, el, base_pos)
        ear = (np.abs(np.sin(np.radians(az))) > 0.9) & (r_base < 0.3)
        dcm = (r[ear] if ear.any() else r) * 100 - HEAD_A * 100
        log(f"忽远忽近：贴耳时离耳 {np.percentile(dcm, 10):.1f}–{np.percentile(dcm, 90):.1f} cm"
            f"（中位 {np.median(dcm):.1f} cm，最远 {dcm.max():.0f} cm），随机退开 {n_sw} 次，"
            f"说话人随距离调整音量 {20 * np.log10(g_sway.min()):+.1f}…{20 * np.log10(g_sway.max()):+.1f} dB")
    if a.save_trajectory:
        step = max(1, int(0.05 * FS / BLOCK))
        np.savetxt(a.save_trajectory, np.stack([t, az, r, el if el is not None else np.zeros(len(t))], 1)[::step],
                   delimiter=",", fmt="%.3f", header="t_sec,az_deg,dist_m,el_deg", comments="")

    y = np.zeros((2, len(x)), np.float32)
    # 贴耳轻声、贴耳气声（气声增强 + 气流扑耳）：只作用于直达声（房间声仍用原来的人声）
    p_ear = closeness(az, r, el)
    xs = soften_near(x, p_ear, a.soft_near)
    xs, puffs = breath_fx(xs, p_ear, a.breath, a.puff)
    xd = apply_warmth(xs, r, a.warmth)
    if puffs is not None:
        xd = xd + puffs
        log(f"气流扑耳：脉冲峰值 {20 * np.log10(float(np.abs(puffs).max()) + 1e-12):.1f} dB（相对人声峰值）")
    del xs, puffs
    direct = render_direct(xd, az, r, tab, a.level_follow, a.isolation, a.presence, a.ku100_eq, a.dark,
                           r_level=r_base if g_sway is not None else None, gain_b=g_sway, el_b=el, level_pos=base_pos,
                           concha_db=a.concha, mouth=a.mouth / 100.0, far_shadow_db=a.far_shadow, pinna=load_pinna(a.pinna))
    del xd
    apply_ear_lf(direct, az, r, a.ear_warmth, a.far_lf_cut, el_b=el)
    if pb is None:
        y += direct
    else:
        conv_add(y, direct, pb[None, :])
    del direct
    if room is not None:
        render_room(y, x, az, room, g_sway, el)
    sos = signal.butter(2, 25, "high", fs=FS, output="sos")
    zi = np.zeros((sos.shape[0], 2, 2))
    for n0 in range(0, y.shape[1], 1 << 21):
        y[:, n0:n0 + (1 << 21)], zi = signal.sosfilt(sos, y[:, n0:n0 + (1 << 21)], axis=1, zi=zi)
    # 去掉尾部多余的静音
    env = np.abs(y).max(axis=0)
    last = int(np.nonzero(env > 1e-5 * env.max())[0][-1]) if env.max() > 0 else n_sig
    y = y[:, :max(n_sig, last + 1)]

    # 响度归一化；贴耳的峰值很高，用前瞻限制器最多压 --max-gr dB，再不够就整体降低
    y *= 10 ** ((a.lufs - integrated_lufs(y)) / 20)
    lim = 10 ** (a.peak / 20)
    tp = true_peak(y)
    if tp > lim:
        need = 20 * np.log10(tp / lim)
        if need > a.max_gr:
            y *= 10 ** (-(need - a.max_gr) / 20)
            log(f"  峰值太高（需压 {need:.1f} dB，上限 --max-gr {a.max_gr:g} dB），响度会低于目标，实际值见下；想更响可调大 --max-gr")
        y, gr = limit(y, lim * 0.97)
        log(f"  峰值限制：最大压缩 {gr:.1f} dB")
        tp = true_peak(y)
        if tp > lim:
            y *= lim / tp
    out = a.output or a.input.with_name(a.input.stem + "_asmr.wav")
    np.clip(y, -1, 1, out=y)
    save_audio(out, y, a.sr)
    log(f"完成：{out}  ({y.shape[1] / FS:.1f} s，{integrated_lufs(y):.1f} LUFS，"
        f"真峰值 {20 * np.log10(true_peak(y) + 1e-12):.1f} dBTP，用时 {time.time() - t_start:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
