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
  * 人头麦贴耳感（默认 ear 风格）：嘴离耳朵约 1.7 cm，远耳额外压低 32 dB，近耳低频更厚、
    临场感更强；离开耳边时平滑恢复。--style ku100 改为按两段 KU100 实录标定（远耳听得到、
    离耳约 6 cm、真实房间）。不改变原来的嗓音，也不添加任何底噪或呼吸声。
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
TABLE_VERSION = 3
REAL_ROOM_DB = -28.0       # 真实房间：贴耳时远耳听到的房间声比近耳直达声低多少（KU100 实测 −24…−35 dB）

HERE = Path(__file__).resolve().parent
HRTF_DIR = HERE / "hrtf"
DATA_DIR = HERE / "data"
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
    tab = Tables(inv, H, TAU, dirs, minimum_phase(ir).astype(np.float32), tau - tau.mean(), cdfc,
                 float(np.mean(np.sum(e1 ** 2, axis=-1))))
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
                 ear_eq: bool = True, dark_db: float = 0.0) -> None:
    """人头麦贴耳感（原地修改 h_b）：嘴贴近一侧耳朵时，近耳加临场 / 空气感，远耳额外压低（单耳独占）；
    离开耳边、转到脸前时按距离和侧向程度平滑恢复。"""
    p = closeness(az_b, r_b)
    if p.max() <= 0:
        return
    ipsi = (np.mod(az_b, 360.0) < 180.0).astype(int)   # 声源在右边 -> 近耳是右耳（通道 1）
    k = np.arange(len(h_b))
    if iso_db > 0:
        h_b[k, 1 - ipsi] *= (10 ** (-iso_db * p / 20))[:, None].astype(np.float32)
    if ear_eq or presence_db != 0 or dark_db != 0:
        fz = np.maximum(np.fft.rfftfreq(NFFT_BLK, 1 / FS), 1.0)
        d = np.interp(np.log(fz), np.log(EAR_EQ_F), EAR_EQ_DB) if ear_eq else np.zeros_like(fz)  # KU100 实测近耳校正
        d = d + presence_db * np.exp(-0.5 * (np.log2(fz / 3500) / 0.7) ** 2) + 0.7 * presence_db / (1 + (9000 / fz) ** 2)
        d = d - dark_db * np.exp(-0.5 * (np.log2(fz / 5000) / 0.6) ** 2)   # "暗"：5 kHz 附近压低（Terashima 2024）
        eq = np.fft.rfft(minphase_from_mag(10 ** (d / 20), NFFT_BLK, 64), NFFT_BLK)
        fade = np.ones(NTAP, np.float32)
        fade[-16:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, 16)))
        for i in range(0, len(h_b), 8192):
            kk, e = k[i:i + 8192], ipsi[i:i + 8192]
            hi = h_b[kk, e]
            he = np.fft.irfft(np.fft.rfft(hi, NFFT_BLK, axis=-1) * eq, NFFT_BLK, axis=-1)[:, :NTAP] * fade
            h_b[kk, e] = hi + p[i:i + 8192, None] * (he - hi)


def closeness(az_b: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """嘴贴近某一侧耳朵的程度（0–1）：距离越近、越偏向正侧方越大；转到脸前 / 脑后时为 0。"""
    lat = np.abs(np.sin(np.radians(az_b)))
    return (np.clip((0.2 - r_b) / (0.2 - EAR_R), 0, 1) * lat ** 2).astype(np.float32)


def apply_ear_lf(y: np.ndarray, az_b: np.ndarray, r_b: np.ndarray, near_db: float, far_db: float,
                 fc: float = 320.0) -> None:
    """贴耳时近耳低频提升、远耳低频再压低，让低频两耳差更大（Brungart：近场距离感的主导线索；
    Honda：128 Hz 两耳差的起伏与战栗感 r=0.93）。用 4 阶 Linkwitz–Riley 分频，两耳做完全相同的分频，
    所以 ITD 不受影响。原地修改 y (2, N)。"""
    if near_db <= 0 and far_db <= 0:
        return
    p = closeness(az_b, r_b)
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


def detect_bursts(x: np.ndarray):
    """找送气爆破音（p/t/k/q/ch/c）：成阻的静默（约 20 ms）之后 1.5–8 kHz 能量突然跳起、且以高频为主。
    返回 [(样本位置, 强度)]，强度为随后 40 ms 送气段能量相对典型值的比例。"""
    from scipy.ndimage import maximum_filter1d
    hop = int(0.005 * FS)
    n = len(x) // hop
    if n < 20:
        return []
    hf = signal.sosfilt(signal.butter(4, [1500, 8000], "band", fs=FS, output="sos"), x)
    lf = signal.sosfilt(signal.butter(4, [100, 1000], "band", fs=FS, output="sos"), x)
    eh = 10 * np.log10(np.mean(np.square(hf[:n * hop].reshape(n, hop), dtype=np.float64), axis=1) + 1e-12)
    el = 10 * np.log10(np.mean(np.square(lf[:n * hop].reshape(n, hop), dtype=np.float64), axis=1) + 1e-12)
    ref = float(np.percentile(eh[eh > eh.max() - 60], 95))
    mf = maximum_filter1d(np.maximum(eh, el), size=4, origin=1)            # 覆盖 k-3..k
    prev = np.concatenate([[-np.inf], mf[:-1]])                               # 前 20 ms（k-4..k-1）的最大值
    cand = np.nonzero((eh - prev > 10) & (eh > ref - 30) & (eh - el > -6))[0]
    out, last = [], -10 ** 9
    for k in cand:
        if k - last < 30 or k + 8 >= n:                                     # 150 ms 内只算一次
            continue
        out.append((k * hop, float(np.mean(10 ** ((eh[k:k + 8] - ref) / 10)))))
        last = k
    return out


def make_puffs(x: np.ndarray, direct: np.ndarray, az_b: np.ndarray, r_b: np.ndarray, amount: float, rng):
    """爆破音气团：送气音爆破时，气团比声音晚约 10 ms 扑到贴着的那只耳朵上，形成低频压力脉冲
    （Derrick 等 2009 测量过"pa"的气团和话筒"噗"声的时差；去爆破音工具在 20–80 Hz 检测这类声音）。
    只在嘴贴近耳边时出现。返回 (2, N) 和次数。"""
    out = np.zeros((2, len(x)), np.float32)
    if amount <= 0:
        return out, 0
    p = closeness(az_b, r_b)
    tb = np.arange(len(p)) * BLOCK
    right = np.mod(az_b, 360.0) < 180.0
    L = int(0.12 * FS)
    tt = np.arange(L) / FS
    rise = 0.006
    shape = np.where(tt < rise, 0.5 - 0.5 * np.cos(np.pi * tt / rise), np.exp(-(tt - rise) / 0.035))
    turb_sos = signal.butter(2, 150, "low", fs=FS, output="sos")
    delay = int(0.010 * FS)
    n = 0
    for pos, strength in detect_bursts(x):
        s0 = pos + delay
        if s0 + L >= len(x):
            continue
        k = min(int(np.searchsorted(tb, s0)), len(p) - 1)
        if p[k] < 0.05:
            continue
        turb = signal.sosfilt(turb_sos, rng.standard_normal(L))
        turb /= turb.std() + 1e-12
        out[1 if right[k] else 0, s0:s0 + L] += (shape * (0.7 + 0.3 * turb) * math.sqrt(min(strength, 4.0)) * p[k]).astype(np.float32)
        n += 1
    if n:
        sos = np.vstack([signal.butter(2, 15, "high", fs=FS, output="sos"), signal.butter(2, 200, "low", fs=FS, output="sos")])
        out = signal.sosfilt(sos, out, axis=1).astype(np.float32)
        vpk = float(np.percentile(np.abs(direct), 99.5))
        out *= 0.3 * amount * vpk / (float(np.percentile(np.abs(out[np.abs(out) > 0]), 99)) + 1e-12)
        cap = 0.2 * vpk * max(amount, 1e-6) ** 0.5                       # 软上限：叠在元音上也不顶到限制器
        out = (cap * np.tanh(out / cap)).astype(np.float32)
    return out, n


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
                  chunk_blocks: int = 2048) -> np.ndarray:
    """时变 HRIR 卷积（重叠保留 + 块内交叉淡化）+ 逐样本分数时延。az_b/r_b 为各块边界处的位置。
    r_level：响度补偿按这条轨迹算（不含忽远忽近，退开时的音量变化另由 gain_b 决定）。"""
    N = len(x)
    nb = -(-N // BLOCK)
    assert len(az_b) == nb + 1
    h_b, tau_b = lookup(tab, np.mod(-az_b, 360.0), r_b)   # 用户方位角顺时针 -> SOFA 逆时针
    rc = np.clip(r_b, R_MIN, R_MAX)
    if level_follow != 1.0:
        # 响度补偿：按语音频段加权的双耳总功率，相对"贴耳 0.14 m"拉平；方向线索（ITD/ILD/耳廓）不变
        if r_level is None:
            p = np.concatenate([binaural_power(h_b[i:i + 8192]) for i in range(0, len(h_b), 8192)])
        else:
            p = path_power(tab, az_b, r_level)
        pref = binaural_power(lookup(tab, np.array([270.0]), np.array([0.14]))[0])[0]
        h_b *= ((p / pref) ** (-(1.0 - level_follow) / 2))[:, None, None].astype(np.float32)
    if gain_b is not None:
        h_b *= gain_b[:, None, None].astype(np.float32)
    if ear_eq or iso_db > 0 or presence_db != 0 or dark_db != 0:
        ear_intimacy(h_b, az_b, rc, iso_db, presence_db, ear_eq, dark_db)
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


def path_power(tab: Tables, az_b: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """沿轨迹各块的双耳总功率（语音加权），分块查表以免占用太多内存。"""
    out = np.empty(len(az_b))
    for i in range(0, len(az_b), 8192):
        h, _ = lookup(tab, np.mod(-az_b[i:i + 8192], 360.0), r_b[i:i + 8192])
        out[i:i + 8192] = binaural_power(h)
    return out


def sway_gain(tab: Tables, az_b: np.ndarray, r_b: np.ndarray, r_base: np.ndarray, keep: float) -> np.ndarray:
    """忽远忽近时说话人自己的音量变化：退开时会不自觉地说大声一点、凑近时放轻（Pelegrín-García 等 2011；
    KU100 实录里近耳音量只保留了物理衰减的约 44%）。返回每块的增益，耳边音量 = 物理衰减 ** keep。
    这是说话人发出的声音本身变了，所以房间声也要乘同样的增益。"""
    if keep >= 1.0 or np.allclose(r_b, r_base):
        return np.ones(len(r_b))
    return (path_power(tab, az_b, r_b) / path_power(tab, az_b, r_base)) ** (-(1.0 - keep) / 2)


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


def render_room(out: np.ndarray, x: np.ndarray, az_b: np.ndarray, room: Room, gain_b: np.ndarray | None = None) -> None:
    """早期反射按声源方向在 12 个扇区间插值（权重加在输入端），尾巴共用；结果累加进 out。
    gain_b：说话人音量的变化（忽远忽近），房间声跟着说话人的音量走，而不跟距离走。"""
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
        self.cur_az, self.cur_r = az0, r0

    def move(self, t0: float, t1: float, az1: float, r1: float, via: str = "short", pass_r: float | None = None):
        d = arc_delta(self.cur_az, az1, via)
        a0, rr0 = self.cur_az, self.cur_r
        bump = 0.0
        if pass_r is not None:
            bump = max(0.0, pass_r - 0.5 * (rr0 + r1))
        m = self.t >= t0
        s = smoothstep((self.t[m] - t0) / max(t1 - t0, 1e-6))
        self.az[m] = a0 + d * s
        self.r[m] = (1 - s) * rr0 + s * r1 + bump * np.sin(np.pi * s)
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
            tr.move(t0, t0 + mv, az1, r1, via, pass_r=a.pass_dist)
            t0 += mv + a.hold_time * rng.uniform(0.9, 1.1)
        k += 1
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


def add_micro(az, r, t, rng, sigma):
    """人体自然晃动：笛卡尔坐标里叠加几条慢速正弦（0.06–0.45 Hz），默认 RMS 1.2 cm。"""
    X = r * np.sin(np.radians(az))
    Y = r * np.cos(np.radians(az))
    if sigma > 0:
        for arr in (X, Y):
            fq, ph, am = rng.uniform(0.06, 0.45, 4), rng.uniform(0, 2 * np.pi, 4), rng.uniform(0.5, 1.0, 4)
            s = (am[:, None] * np.sin(2 * np.pi * fq[:, None] * t[None, :] + ph[:, None])).sum(0)
            arr += s / (s.std() + 1e-12) * sigma
    return np.degrees(np.arctan2(X, Y)) % 360.0, np.clip(np.hypot(X, Y), R_MIN, R_MAX)


def speech_loudness(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """说话人此刻的音量（dB，相对说话时的中位数），在块边界 t 上取值，供"声音大就退远一点"使用。
    语音频段 50 ms 能量，只在说话时更新，停顿里保持、停久了慢慢回到中位。再做"提前让开、慢慢回来"
    的包络：响的字到来前约 0.8 s 就开始往后让（每 6 dB），说完后约 2.5 s 才凑回来（每 6 dB），
    不会贴着耳朵炸出来，也不会一下子弹回耳边。"""
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
    cur, relax = 0.0, math.exp(-0.05 / 1.5)
    for i in range(n):
        cur = d[i] if act[i] else cur * relax
        out[i] = cur
    att, rel = 6.0 * 0.05 / 0.8, 6.0 * 0.05 / 2.5        # 每帧最多变化多少 dB：提前让开快、回来慢
    for i in range(1, n):
        out[i] = max(out[i], out[i - 1] - rel)
    for i in range(n - 2, -1, -1):
        out[i] = max(out[i], out[i + 1] - att)
    out -= np.median(out[act])
    out = signal.sosfiltfilt(signal.butter(2, 0.6, fs=1.0 / 0.05, output="sos"), out)
    return np.interp(t, (np.arange(n) + 0.5) * hop / FS, out)


def sway_gestures(t: np.ndarray, rng, period: float) -> tuple[np.ndarray, int]:
    """有节奏又不固定的"凑近—退开"：贴着耳朵停一会儿 → 慢慢往后让 → 稍停 → 再慢慢凑回来，循环往复。
    每段时长和退开的深浅都随机（偶尔只让一点，偶尔退得更远），凑回来比退开慢（逼近更引人注意，
    Neuhoff 1998；进入身体周围空间时触觉增强，Canzoneri 2012）。一个来回的中位数约 period 秒
    （KU100 实录 4–23 s，中位约 9 s）。返回离耳距离相对基准的倍数（log2，倍频程）和退开次数。"""
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
        tc += min(7.0, math.exp(rng.normal(math.log(2.5), 0.45))) * sc
        keys.append((tc, float(np.clip(u_in + rng.uniform(-0.15, 0.15), -0.45, 0.2))))
        n += 1
    kt = np.array([k[0] for k in keys])
    kv = np.array([k[1] for k in keys])
    i = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 2)
    u = smoothstep((t - kt[i]) / np.maximum(kt[i + 1] - kt[i], 1e-6))
    return kv[i] + (kv[i + 1] - kv[i]) * u, int(np.sum(kt[2::4] < T))


def apply_sway(x: np.ndarray, t: np.ndarray, r: np.ndarray, rng, a) -> tuple[np.ndarray, np.ndarray, int]:
    """忽远忽近：改变嘴到头表面的距离（方位不变）= 随机的凑近 / 退开 + 跟着说话音量（每响 6 dB 退远一倍，
    轻声时更贴近）。只在耳边附近生效：基准距离 0.2 m 以内全量，0.45 m 以外不晃（"从远处凑近"保持原样）。
    返回新的距离、log2 倍数曲线、退开次数。"""
    w = np.zeros(len(t))
    n = 0
    if a.sway > 0:
        g, n = sway_gestures(t, rng, a.sway_period)
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
    ap.add_argument("--sway-period", type=float, default=7.0, help="一次凑近—退开的平均时长（秒，实际随机，约 4–14 s）")
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
    ap.add_argument("--puff", type=float, default=0.0, help="爆破音气团扑到耳朵上的低频脉冲（0=关闭，1=标准）")
    ap.add_argument("--soft-voice", type=float, default=0.0,
                    help="轻声化：按 ANSI S3.5 说话力度频谱，把嗓音往\"压低声音说\"的方向均衡（0=关闭，1=低一档力度）")
    ap.add_argument("--deess", type=float, default=0.0, help="去齿音：齿音最多压低多少 dB（0=关闭，建议 4–8）")
    ap.add_argument("--start", type=float, help="只处理从这里开始（秒）")
    ap.add_argument("--duration", type=float, help="只处理这么长（秒）")
    ap.add_argument("--sr", type=int, default=FS, help="输出采样率")
    ap.add_argument("--seed", type=int, default=1, help="随机种子（换一个就换一套走位）")
    ap.add_argument("--save-trajectory", type=Path, help="把轨迹写成 CSV（时间, 方位角, 距离）")
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
    rng_fx = np.random.default_rng(a.seed + 101)
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

    rng = np.random.default_rng(a.seed)
    if a.mode == "sweep":
        az, r, (n, n_app) = traj_sweep(t, rng, a)
        desc = f"{n} 次左右穿梭（其中 {n_app} 次从远处凑近；移动 {a.move_time:g} s，停留 {a.hold_time:g} s）"
    elif a.mode == "phrase":
        az, r, n = traj_phrase(x[:n_sig], t, rng, a)
        desc = f"{n} 次换位"
    elif a.mode == "pingpong":
        az, r, n = traj_pingpong(t, rng, a)
        desc = f"{n} 次左右切换"
    elif a.mode == "orbit":
        az, r, n = traj_orbit(t, rng, a)
        desc = f"约 {n} 圈"
    elif a.mode == "wander":
        az, r, n = traj_wander(t, rng, a)
        desc = f"{n} 次走动"
    elif a.mode == "static":
        az, r, desc = np.full(len(t), a.az if a.az_set else 270.0), np.full(len(t), a.dist or 0.15), "固定"
    else:
        az, r, n = traj_path(t, a.path)
        desc = f"{n} 个关键帧"
    az, r = add_micro(az, r, t, rng, 0.012 * a.micro)
    log(f"轨迹：{a.mode}，{desc}；距离 {r.min():.2f}–{r.max():.2f} m")
    r_base, g_sway = r, None
    if a.sway > 0 or a.sway_follow > 0:
        r, _, n_sw = apply_sway(x, t, r_base, np.random.default_rng(a.seed + 202), a)
        g_sway = sway_gain(tab, az, r, r_base, a.sway_level)
        ear = (np.abs(np.sin(np.radians(az))) > 0.9) & (r_base < 0.3)
        dcm = (r[ear] if ear.any() else r) * 100 - HEAD_A * 100
        log(f"忽远忽近：贴耳时离耳 {np.percentile(dcm, 10):.1f}–{np.percentile(dcm, 90):.1f} cm"
            f"（中位 {np.median(dcm):.1f} cm，最远 {dcm.max():.0f} cm），随机退开 {n_sw} 次，"
            f"说话人随距离调整音量 {20 * np.log10(g_sway.min()):+.1f}…{20 * np.log10(g_sway.max()):+.1f} dB")
    if a.save_trajectory:
        step = max(1, int(0.05 * FS / BLOCK))
        np.savetxt(a.save_trajectory, np.stack([t, az, r], 1)[::step], delimiter=",", fmt="%.3f",
                   header="t_sec,az_deg,dist_m", comments="")

    y = np.zeros((2, len(x)), np.float32)
    xd = apply_warmth(x, r, a.warmth)
    direct = render_direct(xd, az, r, tab, a.level_follow, a.isolation, a.presence, a.ku100_eq, a.dark,
                           r_level=r_base if g_sway is not None else None, gain_b=g_sway)
    del xd
    apply_ear_lf(direct, az, r, a.ear_warmth, a.far_lf_cut)
    if a.puff > 0:
        puffs, npf = make_puffs(x, direct, az, r, a.puff, rng_fx)
        direct += puffs
        del puffs
        log(f"爆破音气团：{npf} 次")
    if pb is None:
        y += direct
    else:
        conv_add(y, direct, pb[None, :])
    del direct
    if room is not None:
        render_room(y, x, az, room, g_sway)
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
