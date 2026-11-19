#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24", "scipy>=1.10", "h5py>=3.8"]
# ///
"""把 Python 版（../asmr_binaural.py）预处理好的 KU100 数据导出成 Rust 程序内嵌的二进制资源 assets/ku100.bin。

内容（小端）：
  近场 HRIR 查找表（48 个距离 × 360 个方位 × 2 耳 × 128 点，逐条归一化后存 int16）、时延表、
  响度补偿用的双耳功率表、KU100 扩散场均衡、实测真实房间（12 个方向）、23 款耳机补偿滤波器、
  高度修正表（2 耳 × 37 个侧向角 × 36 个极角 × 48 点，int16）。

用法：uv run tools/export_assets.py            （在 auto-asmr/auto-asmr 目录下运行）
数据来源与许可见 README：TH Köln，Zenodo 4297951 / 5031335，CC BY 4.0。
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                     # 仓库根目录（asmr_binaural.py 所在处）
sys.path.insert(0, str(ROOT))
import asmr_binaural as ab                    # noqa: E402

HP_LEN = 1024


def q16(a: np.ndarray):
    """按最后一维逐条归一化成 int16，返回 (scale float32, data int16)。"""
    pk = np.abs(a).max(axis=-1)
    scale = np.where(pk > 0, pk / 32767.0, 1.0).astype(np.float32)
    q = np.round(a / scale[..., None]).clip(-32767, 32767).astype("<i2")
    return scale, q


def main() -> None:
    tab = ab.load_tables(ab.HRTF_DIR)
    out = bytearray()
    out += b"ASMRKU03"
    n_r, n_az, _, n_tap = tab.H.shape
    out += struct.pack("<IIII", ab.FS, n_r, n_az, n_tap)
    out += tab.inv_grid.astype("<f4").tobytes()
    hs, hq = q16(tab.H.astype(np.float64))
    out += hs.astype("<f4").tobytes()
    out += hq.tobytes()
    out += tab.TAU.astype("<f4").tobytes()
    power = np.stack([ab.binaural_power(tab.H[i]) for i in range(n_r)])       # (n_r, n_az)
    out += power.astype("<f4").tobytes()
    cdfc = np.asarray(tab.cdfc, np.float64)
    out += struct.pack("<I", len(cdfc)) + cdfc.astype("<f4").tobytes()

    z = np.load(ab.DATA_DIR / "room_audiolab_ku100.npz")
    room = z["sectors"].astype(np.float64) * float(z["scale"])                # (12, 2, L)
    rs, rq = q16(room)
    out += struct.pack("<II", room.shape[0], room.shape[2])
    out += rs.astype("<f4").tobytes()
    out += rq.tobytes()

    import scipy.io as sio
    files = ab.headphone_models(ab.HRTF_DIR)
    out += struct.pack("<II", len(files), HP_LEN)
    fade = np.ones(HP_LEN)
    fade[-128:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, 128)))
    for f in files:
        h = np.asarray(sio.loadmat(f, squeeze_me=True, struct_as_record=False)["hpcf"].minPhase, float)[:HP_LEN] * fade
        name = f.stem.encode("utf-8")
        out += struct.pack("<I", len(name)) + name + h.astype("<f4").tobytes()

    # 高度修正：[耳][从耳朵看的侧向角 −90…90][绕两耳连线的极角 0…350] 的最小相位 FIR
    elev = tab.elev.astype(np.float64)
    es, eq = q16(elev)
    out += struct.pack("<IIIff", elev.shape[1], elev.shape[2], elev.shape[3], ab.ELEV_DA, ab.ELEV_DB)
    out += es.astype("<f4").tobytes()
    out += eq.tobytes()

    dst = HERE.parent / "assets" / "ku100.bin"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(bytes(out))
    err = np.abs(hq.astype(np.float64) * hs[..., None] - tab.H).max() / np.abs(tab.H).max()
    print(f"写入 {dst}  {len(out) / 1e6:.2f} MB；HRIR {tab.H.shape}，量化误差（相对最大值）{20 * np.log10(err + 1e-30):.0f} dB；"
          f"房间 {room.shape}；耳机 {len(files)} 款；高度修正 {elev.shape}（{eq.nbytes / 1e3:.0f} KB）")


if __name__ == "__main__":
    main()
