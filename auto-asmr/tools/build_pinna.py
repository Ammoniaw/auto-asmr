#!/usr/bin/env python3
"""从 HUTUBS 真人 HRTF 库生成"耳廓选择"用的滤波表（只在重新生成表时需要运行；运行软件的人不需要这个数据集）。

数据：F. Brinkmann 等, "The HUTUBS head-related transfer function (HRTF) database", TU Berlin 2019,
      https://depositonce.tu-berlin.de/handle/11303/9429 （CC BY 4.0），用其中的 HRIRs.zip（实测，440 个方向，1.47 m）。

做法：近场的距离、两耳差、时延都保留 KU100 的实测数据，只把"耳廓那一部分"换成真人的——
  修正(从耳朵看的方向, 频率) = 真人的方向传递函数 ÷ 本程序 KU100 渲染结果的方向传递函数
  方向传递函数 = 某方向的响应 ÷ 所有方向的平均（扩散场）。两边都除掉各自的平均，所以整体音色仍是 KU100 的，
  换掉的只是"随方向变化"的那部分（耳廓的峰谷位置）。1.5 kHz 以下不动，1.5–3 kHz 渐入；
  耳朵背对声源（在头影里）的方向不换。

选哪几只：把所有真人（不含人工头和重复测量）的修正曲线做 k-medoids 聚类，取各类的中心，
  思路同 Katz & Parseihian 2012（少数几套就能让每个人找到合适的），但这里按频谱聚类，不是按听感评分。

用法（在仓库根目录）：
  uv run --with numpy --with scipy --with h5py python -I auto-asmr/tools/build_pinna.py hrtf/hutubs/HRIRs.zip
输出：data/pinna_hutubs.npz（Python 版）和 auto-asmr/assets/pinna.bin（Rust 版）。
"""
import io
import re
import struct
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import asmr_binaural as ab  # noqa: E402

NF = 2048
R_KU, R_HUM, EAR_HUM = 1.5, 1.47, 0.075     # KU100 取 1.5 m；HUTUBS 的测量距离；真人耳朵离头中心的典型值
F_LO, F_HI = 1500.0, 3000.0                 # 这以下不换 / 这以上全换
SKIP = {1, 96, 88}                          # 1、96 是 FABIAN 人工头；88 是 22 号的重复测量
K_PICK = 7
BANDS = (3150, 4000, 5000, 6300, 8000, 10000, 12500)


def grid():
    al = np.radians(np.arange(-90.0, 90.0 + 1e-6, ab.ELEV_DA))
    be = np.radians(np.arange(0.0, 360.0, ab.ELEV_DB))
    A, B = np.meshgrid(al, be, indexing="ij")
    q = np.stack([np.cos(A) * np.cos(B), np.sin(A), np.cos(A) * np.sin(B)], -1)   # 耳心坐标：x 前，y 朝这只耳朵自己的一侧，z 上
    return A, q.reshape(-1, 3)


def ku_power(tab, q):
    """本程序在"从耳朵看的方向"网格上渲染出的 KU100 响应（平滑功率谱），(2, G, F)。"""
    out = []
    for e, sgn in enumerate((1.0, -1.0)):
        p = np.array([0.0, sgn * ab.HEAD_A, 0.0]) + R_KU * q * np.array([1.0, sgn, 1.0])
        r = np.linalg.norm(p, axis=1)
        az = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360.0
        el = np.degrees(np.arcsin(np.clip(p[:, 2] / r, -1, 1)))
        h, _ = ab.lookup3(tab, az, el, r)
        out.append(ab._smooth_pow(np.abs(np.fft.rfft(h[:, e].astype(np.float64), NF, axis=-1)) ** 2, NF))
    return np.stack(out)


def human_power(ir, pos, q):
    """一个真人在同一套方向上的响应（平滑功率谱），(2, G, F)。ir (M, 2, N) 44.1 kHz；pos (M, 3) 度 / 度 / 米。"""
    P = np.abs(np.fft.rfft(signal.resample_poly(ir, 160, 147, axis=-1), NF, axis=-1)) ** 2
    a, e = np.radians(pos[:, 0]), np.radians(pos[:, 1])
    tree = cKDTree(np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1))
    out = []
    for ear, sgn in enumerate((1.0, -1.0)):
        p = np.array([0.0, sgn * EAR_HUM, 0.0]) + R_HUM * q * np.array([1.0, sgn, 1.0])
        dist, idx = tree.query(p / np.linalg.norm(p, axis=1, keepdims=True), k=4)
        ang = 2 * np.arcsin(np.clip(dist / 2, 0, 1))
        w = 1.0 / (ang ** 2 + np.radians(1.0) ** 2)
        w /= w.sum(1, keepdims=True)
        out.append(ab._smooth_pow((w[..., None] * P[idx, ear]).sum(1), NF))
    return np.stack(out)


def correction_db(Ps, Pk, A):
    """真人方向传递函数 ÷ KU100 方向传递函数，dB，(2, G, F)。"""
    wq = np.cos(A).reshape(-1)
    wq = wq / wq.sum()
    dfs = (wq[None, :, None] * Ps).sum(1, keepdims=True)
    dfk = (wq[None, :, None] * Pk).sum(1, keepdims=True)
    c = 10 * np.log10((Ps / dfs) / (Pk / dfk) + 1e-20)
    f = np.fft.rfftfreq(NF, 1 / ab.FS)
    k18 = int(18000 / (ab.FS / NF))
    c[..., k18:] = c[..., k18:k18 + 1]                      # 实测数据 18 kHz 以上不可靠，沿用 18 kHz 的值
    s = np.clip(np.log2(np.maximum(f, 1.0) / F_LO) / np.log2(F_HI / F_LO), 0, 1)
    side = np.clip((np.degrees(A).reshape(-1) + 60.0) / 60.0, 0, 1)   # 耳朵背对声源 60° 以上：不换
    return np.clip(c, -18.0, 12.0) * (s * s * (3 - 2 * s))[None, None, :] * (side * side * (3 - 2 * side))[None, :, None]


def to_fir(c):
    mag = 10 ** (c[..., ::NF // ab.NFFT_BLK] / 20)
    return ab.minphase_from_mag(mag, ab.NFFT_BLK, ab.PINNA_TAPS).astype(np.float32)


def features(c, A):
    """聚类用：朝向自己一侧的方向上、3–12.5 kHz 各 1/3 倍频程带的修正量（按立体角加权）。"""
    f = np.fft.rfftfreq(NF, 1 / ab.FS)
    own = np.degrees(A).reshape(-1) >= 0
    w = np.sqrt(np.cos(A).reshape(-1)[own])
    b = np.stack([c[:, own][..., (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))].mean(-1) for fc in BANDS], -1)
    return (b * w[None, :, None]).reshape(-1) / np.sqrt((w ** 2).mean())


def k_medoids(D, k, iters=50):
    med = [int(np.argmin(D.sum(1)))]
    while len(med) < k:                                       # 最远点初始化
        med.append(int(np.argmax(D[:, med].min(1))))
    for _ in range(iters):
        lab = np.argmin(D[:, med], 1)
        new = [int(np.nonzero(lab == j)[0][np.argmin(D[np.ix_(lab == j, lab == j)].sum(1))]) for j in range(k)]
        if new == med:
            break
        med = new
    return med, np.argmin(D[:, med], 1)


def main():
    zpath = Path(sys.argv[1])
    tab = ab.load_tables(ab.HRTF_DIR)
    A, q = grid()
    Pk = ku_power(tab, q)
    subj, corr = [], []
    with zipfile.ZipFile(zpath) as z:
        names = sorted((int(m.group(1)), n) for n in z.namelist() if (m := re.search(r"pp(\d+)_HRIRs_measured\.sofa$", n)))
        for sid, n in names:
            if sid in SKIP:
                continue
            with h5py.File(io.BytesIO(z.read(n)), "r") as f:
                ir = np.asarray(f["Data.IR"], dtype=np.float64)
                pos = np.asarray(f["SourcePosition"], dtype=np.float64)
                fs = float(np.asarray(f["Data.SamplingRate"]).ravel()[0])
            assert fs == 44100.0 and ir.shape[1] == 2, (sid, fs, ir.shape)
            corr.append(correction_db(human_power(ir, pos, q), Pk, A).astype(np.float32))
            subj.append(sid)
    corr = np.stack(corr)                                      # (S, 2, G, F)
    print(f"{len(subj)} 个真人；方向网格 {A.shape}，频率点 {corr.shape[-1]}")
    X = np.stack([features(c, A) for c in corr])
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).mean(-1))
    to_ku = np.sqrt((X ** 2).mean(-1))
    print(f"修正量（3–12.5 kHz，朝向自己一侧）的均方根：相对 KU100 中位 {np.median(to_ku):.1f} dB"
          f"（{to_ku.min():.1f}–{to_ku.max():.1f}）；真人两两之间中位 {np.median(D[np.triu_indices(len(subj), 1)]):.1f} dB")
    med, lab = k_medoids(D, K_PICK)
    order = np.argsort([-(lab == j).sum() for j in range(K_PICK)])
    picks = [med[j] for j in order]
    near = D[np.arange(len(subj)), np.array(med)[lab]]
    print(f"选 {K_PICK} 只后，每个人到最像的那只的距离：中位 {np.median(near):.1f} dB，最差 {near.max():.1f} dB")
    labels = [chr(ord("A") + i) for i in range(K_PICK)]
    for lb, j, i in zip(labels, order, picks):
        print(f"  {lb}: HUTUBS pp{subj[i]}，代表 {(lab == j).sum()} 人，与 KU100 相差 {to_ku[i]:.1f} dB")
    tables = np.stack([to_fir(corr[i]).reshape(2, A.shape[0], A.shape[1], ab.PINNA_TAPS) for i in picks])
    # 滤波器能不能表示目标曲线
    f = np.fft.rfftfreq(NF, 1 / ab.FS)
    band = (f >= 3000) & (f <= 14000)
    real = 20 * np.log10(np.abs(np.fft.rfft(tables.reshape(K_PICK, 2, -1, ab.PINNA_TAPS), NF, axis=-1)) + 1e-9)
    err = (real - corr[picks])[..., band]
    print(f"{ab.PINNA_TAPS} 点滤波器与目标曲线的差（3–14 kHz）：均方根 {np.sqrt((err ** 2).mean()):.2f} dB，最大 {np.abs(err).max():.1f} dB")
    out = ROOT / "data" / "pinna_hutubs.npz"
    np.savez_compressed(out, names=np.array(labels), subjects=np.array([subj[i] for i in picks]), tables=tables)
    binp = ROOT / "auto-asmr" / "assets" / "pinna.bin"
    with open(binp, "wb") as fo:
        fo.write(b"ASMRPIN1" + struct.pack("<4I", K_PICK, A.shape[0], A.shape[1], ab.PINNA_TAPS))
        for lb, i, t in zip(labels, picks, tables):
            nb = lb.encode()
            fo.write(struct.pack("<I", len(nb)) + nb + struct.pack("<I", subj[i]) + t.astype("<f4").tobytes())
    print(f"已写 {out}（{out.stat().st_size / 1e6:.1f} MB）和 {binp}（{binp.stat().st_size / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
