//! 耳廓选择：把 KU100 的耳廓换成真人的。
//!
//! 近场的距离、两耳差、时延仍是 KU100 实测的，整体音色也仍是 KU100 的；换掉的只是 1.5 kHz 以上
//! "随方向变化"的耳廓峰谷。做法是给每只耳朵的 HRIR 再卷上一个 64 点的修正滤波器：
//! 真人的方向传递函数 ÷ 本程序 KU100 渲染结果的方向传递函数，方向按从这只耳朵看向声源算。
//!
//! 滤波表由 `tools/build_pinna.py` 从 HUTUBS 真人实测 HRTF 库（TU Berlin，CC BY 4.0）生成，随程序打包；
//! 运行时不需要原始数据集。

use super::{HEAD_A, NTAP, R_MIN};
use std::sync::OnceLock;


/// 修正滤波器的长度。
pub const PINNA_TAPS: usize = 64;

/// 一只真人耳廓（左右耳各一张表）。
pub struct Pinna {
    /// 名字（A、B、…）。
    pub name: String,
    /// HUTUBS 里的受试者编号。
    pub subject: u32,
    data: Vec<f32>,
}

pub struct PinnaSet {
    na: usize,
    nb: usize,
    pub items: Vec<Pinna>,
}

/// 打包在程序里的真人耳廓。
pub fn pinnas() -> &'static PinnaSet {
    static P: OnceLock<PinnaSet> = OnceLock::new();
    P.get_or_init(|| PinnaSet::parse(super::data::pinna()).expect("assets/pinna.bin 损坏"))
}

impl PinnaSet {
    fn parse(d: &[u8]) -> Option<PinnaSet> {
        if d.len() < 24 || &d[..8] != b"ASMRPIN1" {
            return None;
        }
        let u32_at = |o: usize| -> Option<usize> { Some(u32::from_le_bytes(d.get(o..o + 4)?.try_into().ok()?) as usize) };
        let (n, na, nb, taps) = (u32_at(8)?, u32_at(12)?, u32_at(16)?, u32_at(20)?);
        if taps != PINNA_TAPS || na < 2 || nb < 1 {
            return None;
        }
        let len = 2 * na * nb * taps;
        let mut o = 24;
        let mut items = Vec::with_capacity(n);
        for _ in 0..n {
            let nl = u32_at(o)?;
            o += 4;
            let name = String::from_utf8(d.get(o..o + nl)?.to_vec()).ok()?;
            o += nl;
            let subject = u32_at(o)? as u32;
            o += 4;
            let raw = d.get(o..o + 4 * len)?;
            o += 4 * len;
            let data = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            items.push(Pinna { name, subject, data });
        }
        Some(PinnaSet { na, nb, items })
    }

    /// 按名字找（不分大小写）。"KU100"、空串或找不到时返回 None，即用 KU100 自己的耳廓。
    pub fn index(&self, name: &str) -> Option<usize> {
        let n = name.trim();
        self.items.iter().position(|p| p.name.eq_ignore_ascii_case(n))
    }

    /// 在（从耳朵看的侧向角, 极角）网格上双线性插值出修正滤波器。
    fn taps(&self, i: usize, ear: usize, lat: f32, beta: f32, out: &mut [f32; PINNA_TAPS]) {
        let (na, nb) = (self.na, self.nb);
        let x = (lat.clamp(-90.0, 90.0) + 90.0) / (180.0 / (na - 1) as f32);
        let i0 = (x.floor() as usize).min(na - 2);
        let fa = x - i0 as f32;
        let y = beta.rem_euclid(360.0) / (360.0 / nb as f32);
        let j0 = (y.floor() as usize) % nb;
        let j1 = (j0 + 1) % nb;
        let fb = y - y.floor();
        let g = &self.items[i].data;
        let at = |a: usize, b: usize| {
            let o = ((ear * na + a) * nb + b) * PINNA_TAPS;
            &g[o..o + PINNA_TAPS]
        };
        let (g00, g01, g10, g11) = (at(i0, j0), at(i0, j1), at(i0 + 1, j0), at(i0 + 1, j1));
        for k in 0..PINNA_TAPS {
            out[k] = (1.0 - fa) * ((1.0 - fb) * g00[k] + fb * g01[k]) + fa * ((1.0 - fb) * g10[k] + fb * g11[k]);
        }
    }

    /// 把第 `i` 只耳廓作用到一对 HRIR 上。`az_sofa` / `el`（度）、`r`（米）是声源相对头中心的位置；
    /// 每只耳朵按"从这只耳朵看向声源"的方向取修正，所以贴耳时与近场的声学视差一致。
    pub fn apply(&self, i: usize, h: &mut [[f32; NTAP]; 2], az_sofa: f32, el: f32, r: f32) {
        let (sa, ca) = az_sofa.to_radians().sin_cos();
        let (se, ce) = el.to_radians().sin_cos();
        let rr = r.max(R_MIN);
        let (vx, vz) = (rr * ce * ca, rr * se);
        let mut taps = [0f32; PINNA_TAPS];
        for (ear, ey) in [HEAD_A, -HEAD_A].into_iter().enumerate() {
            let vy = rr * ce * sa - ey;
            let n = (vx * vx + vy * vy + vz * vz).sqrt().max(1e-9);
            let lat = ((if ear == 0 { vy } else { -vy }) / n).clamp(-1.0, 1.0).asin().to_degrees();
            let beta = vz.atan2(vx).to_degrees().rem_euclid(360.0);
            self.taps(i, ear, lat, beta, &mut taps);
            let src = h[ear];
            for (m, o) in h[ear].iter_mut().enumerate() {
                let kmax = m.min(PINNA_TAPS - 1);
                let mut s = 0.0;
                for k in 0..=kmax {
                    s += taps[k] * src[m - k];
                }
                *o = s;
            }
            for k in 0..16 {
                let t = k as f32 / 15.0;
                h[ear][NTAP - 16 + k] *= 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::filters::fir_mag;
    use crate::dsp::FS;

    #[test]
    fn asset_loads() {
        let p = pinnas();
        assert!(p.items.len() >= 3, "{} 只耳廓", p.items.len());
        assert_eq!(p.index("a"), Some(0));
        assert_eq!(p.index("KU100"), None);
        assert_eq!(p.index(""), None);
    }

    /// 每只耳廓、每个方向：1 kHz 以下不动；耳朵背对声源时整个不动；朝向声源时高频确实有变化。
    #[test]
    fn only_pinna_band_changes() {
        let p = pinnas();
        let db = |m: &[f32], f: f32| 20.0 * m[(f / FS * 4096.0).round() as usize].log10();
        let mut taps = [0f32; PINNA_TAPS];
        for i in 0..p.items.len() {
            let mut hf_change = 0f32;
            for ear in 0..2 {
                for lat in [-90.0f32, -70.0, 0.0, 45.0, 80.0] {
                    for beta in [0.0f32, 95.0, 180.0, 270.0] {
                        p.taps(i, ear, lat, beta, &mut taps);
                        let m = fir_mag(&taps, 4096);
                        assert!(db(&m, 300.0).abs() < 0.3 && db(&m, 1000.0).abs() < 0.5, "耳廓 {i} 低频 {} {}", db(&m, 300.0), db(&m, 1000.0));
                        for f in [4000.0f32, 8000.0, 12000.0] {
                            let v = db(&m, f);
                            assert!(v > -22.0 && v < 16.0, "耳廓 {i} {f} Hz {v} dB");
                            if lat <= -70.0 {
                                assert!(v.abs() < 0.5, "背对声源应不动：{v}");
                            } else if lat >= 45.0 {
                                hf_change = hf_change.max(v.abs());
                            }
                        }
                    }
                }
            }
            assert!(hf_change > 2.0, "耳廓 {i} 朝向声源时高频变化只有 {hf_change} dB");
        }
    }

    /// 作用到 HRIR 上：直流增益不变，远耳（背对声源）几乎不变。
    #[test]
    fn apply_keeps_low_frequencies() {
        let tab = crate::dsp::tables::tables();
        let mut hr = crate::dsp::tables::Hrir::default();
        tab.lookup(260.0, 0.105, &mut hr); // 声源贴右耳（SOFA 260°）
        let mut h = hr.h;
        pinnas().apply(0, &mut h, 260.0, 0.0, 0.105);
        for ear in 0..2 {
            let (a, b): (f32, f32) = (hr.h[ear].iter().sum(), h[ear].iter().sum());
            assert!((20.0 * (b / a).abs().log10()).abs() < 0.3, "耳 {ear} 直流增益变化 {}", 20.0 * (b / a).abs().log10());
        }
        let d: f32 = hr.h[0].iter().zip(h[0].iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / hr.h[0].iter().map(|x| x * x).sum::<f32>();
        assert!(d < 1e-3, "远耳变化 {d}");
        let d: f32 = hr.h[1].iter().zip(h[1].iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / hr.h[1].iter().map(|x| x * x).sum::<f32>();
        assert!(d > 1e-3, "近耳应该有变化 {d}");
    }
}
