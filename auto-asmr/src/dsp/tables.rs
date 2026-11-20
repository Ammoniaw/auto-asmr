//! 内嵌的 KU100 数据：近场 HRIR 查找表、时延、双耳功率、扩散场均衡、实测真实房间、耳机补偿滤波器、高度修正。
//! 由 `tools/export_assets.py` 从 Python 版的预处理结果导出（TH Köln，CC BY 4.0）。

use super::{HEAD_A, NTAP, R_MIN};
use std::sync::OnceLock;


pub struct Tables {
    pub n_r: usize,
    pub n_az: usize,
    /// 1/r 网格，递减（10.53 … 0.667，即 0.095 m … 1.5 m）。
    pub inv_grid: Vec<f32>,
    /// `[n_r][n_az][2][NTAP]` 最小相位 HRIR（SOFA 方位角：逆时针，0 = 正前，90 = 左）。
    pub h: Vec<f32>,
    /// `[n_r][n_az][2]` 相对时延（样本）。
    pub tau: Vec<f32>,
    /// `[n_r][n_az]` 语音加权的双耳总功率，用于响度补偿。
    pub power: Vec<f32>,
    /// KU100 公共扩散场均衡（最小相位 FIR）。
    pub cdfc: Vec<f32>,
    pub n_sec: usize,
    pub room_len: usize,
    /// `[n_sec][2][room_len]` 实测真实房间（去掉直达声），按声源方向分 12 个扇区。
    pub room: Vec<f32>,
    /// 耳机补偿滤波器（型号, 最小相位 FIR）。
    pub headphones: Vec<(String, Vec<f32>)>,
    /// 高度修正：`[2 耳][el_na 个侧向角][el_nb 个极角][el_tap]` 的最小相位 FIR。
    /// 侧向角 = 从这只耳朵看声源偏向自己这一侧多少度（−90…90）；极角 = 绕两耳连线转多少度（0 前、90 上、180 后、270 下）。
    pub elev: Vec<f32>,
    pub el_na: usize,
    pub el_nb: usize,
    pub el_tap: usize,
    pub el_da: f32,
    pub el_db: f32,
}

/// 高度修正滤波器最多这么长。
pub const ELEV_MAX_TAPS: usize = 64;

/// （SOFA 方位角, 仰角）换成绕两耳连线的坐标：方向余弦 (前, 左, 上)、侧向角（度，朝左为正）、
/// 极角（度）、水平面"前"那一点的权重。
#[inline]
pub fn cone_coords(az_sofa: f32, el: f32) -> ([f32; 3], f32, f32, f32) {
    let (a, e) = (az_sofa.to_radians(), el.to_radians());
    let d = [e.cos() * a.cos(), e.cos() * a.sin(), e.sin()];
    let alpha = d[1].clamp(-1.0, 1.0).asin().to_degrees();
    let beta = d[2].atan2(d[0]).to_degrees().rem_euclid(360.0);
    (d, alpha, beta, 0.5 * (1.0 + beta.to_radians().cos()))
}

/// 插值得到的一对 HRIR 和两耳时延。
#[derive(Clone)]
pub struct Hrir {
    pub h: [[f32; NTAP]; 2],
    pub tau: [f32; 2],
}

impl Default for Hrir {
    fn default() -> Self {
        Hrir { h: [[0.0; NTAP]; 2], tau: [0.0; 2] }
    }
}

struct Cur<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.d[self.p..self.p + n];
        self.p += n;
        s
    }
    fn u32(&mut self) -> usize {
        u32::from_le_bytes(self.bytes(4).try_into().unwrap()) as usize
    }
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        self.bytes(n * 4).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    /// 逐条（每条 `len` 点）带缩放的 int16。
    fn q16(&mut self, scales: &[f32], len: usize) -> Vec<f32> {
        let raw = self.bytes(scales.len() * len * 2);
        let mut out = Vec::with_capacity(scales.len() * len);
        for (i, &s) in scales.iter().enumerate() {
            let row = &raw[i * len * 2..(i + 1) * len * 2];
            out.extend(row.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 * s));
        }
        out
    }
}

/// 全局只读数据表（首次调用时解码，约几十毫秒）。
pub fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(Tables::load)
}

impl Tables {
    fn load() -> Tables {
        let mut c = Cur { d: super::data::ku100(), p: 0 };
        assert_eq!(c.bytes(8), b"ASMRKU03", "assets/ku100.bin 格式不对（用 tools/export_assets.py 重新导出）");
        let fs = c.u32();
        assert_eq!(fs, 48_000);
        let (n_r, n_az, n_tap) = (c.u32(), c.u32(), c.u32());
        assert_eq!(n_tap, NTAP);
        let inv_grid = c.f32s(n_r);
        let hs = c.f32s(n_r * n_az * 2);
        let h = c.q16(&hs, n_tap);
        let tau = c.f32s(n_r * n_az * 2);
        let power = c.f32s(n_r * n_az);
        let n_cdfc = c.u32();
        let cdfc = c.f32s(n_cdfc);
        let (n_sec, room_len) = (c.u32(), c.u32());
        let rs = c.f32s(n_sec * 2);
        let room = c.q16(&rs, room_len);
        let (n_hp, hp_len) = (c.u32(), c.u32());
        let mut headphones = Vec::with_capacity(n_hp);
        for _ in 0..n_hp {
            let nl = c.u32();
            let name = String::from_utf8_lossy(c.bytes(nl)).into_owned();
            headphones.push((name, c.f32s(hp_len)));
        }
        let (el_na, el_nb, el_tap) = (c.u32(), c.u32(), c.u32());
        assert!(el_tap <= ELEV_MAX_TAPS && el_tap < NTAP);
        let dd = c.f32s(2);
        let es = c.f32s(2 * el_na * el_nb);
        let elev = c.q16(&es, el_tap);
        Tables { n_r, n_az, inv_grid, h, tau, power, cdfc, n_sec, room_len, room, headphones, elev, el_na, el_nb, el_tap, el_da: dd[0], el_db: dd[1] }
    }

    /// 距离 -> (网格下标 i0, 小数部分)，在 1/r 上分段线性（与 Python 版 `lookup` 相同）。
    #[inline]
    fn grid_pos(&self, r: f32) -> (usize, f32) {
        let iv = 1.0 / r.clamp(R_MIN, 1.5);
        // inv_grid 递减：找第一个 < iv 的位置
        let k = self.inv_grid.partition_point(|&g| g >= iv);
        let i0 = k.saturating_sub(1).min(self.n_r - 2);
        let (g0, g1) = (self.inv_grid[i0], self.inv_grid[i0 + 1]);
        (i0, ((g0 - iv) / (g0 - g1)).clamp(0.0, 1.0))
    }

    #[inline]
    fn az_pos(&self, az_sofa: f32) -> (usize, usize, f32) {
        let a = az_sofa.rem_euclid(360.0);
        let a0 = (a.floor() as usize) % self.n_az;
        (a0, (a0 + 1) % self.n_az, a - a.floor())
    }

    /// 按（SOFA 方位角, 距离）双线性插值出 HRIR 与时延；1.5 m 以外按 1/r 衰减。
    pub fn lookup(&self, az_sofa: f32, r: f32, out: &mut Hrir) {
        let (i0, fu) = self.grid_pos(r);
        let (a0, a1, fa) = self.az_pos(az_sofa);
        let far = 1.5 / r.max(1.5);
        let w = [(1.0 - fu) * (1.0 - fa), (1.0 - fu) * fa, fu * (1.0 - fa), fu * fa];
        let idx = [(i0, a0), (i0, a1), (i0 + 1, a0), (i0 + 1, a1)];
        for e in 0..2 {
            let dst = &mut out.h[e];
            dst.fill(0.0);
            let mut t = 0.0;
            for (wk, &(i, j)) in w.iter().zip(idx.iter()) {
                let base = ((i * self.n_az + j) * 2 + e) * NTAP;
                let src = &self.h[base..base + NTAP];
                let g = wk * far;
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += g * s;
                }
                t += wk * self.tau[(i * self.n_az + j) * 2 + e];
            }
            out.tau[e] = t;
        }
    }

    /// 同样插值出双耳总功率（语音频段加权）。
    pub fn lookup_power(&self, az_sofa: f32, r: f32) -> f32 {
        let (i0, fu) = self.grid_pos(r);
        let (a0, a1, fa) = self.az_pos(az_sofa);
        let far = 1.5 / r.max(1.5);
        let p = |i: usize, j: usize| self.power[i * self.n_az + j];
        let v = (1.0 - fu) * ((1.0 - fa) * p(i0, a0) + fa * p(i0, a1)) + fu * ((1.0 - fa) * p(i0 + 1, a0) + fa * p(i0 + 1, a1));
        v * far * far
    }

    /// 一只耳朵的高度修正滤波器：在（从耳朵看的侧向角, 极角）网格上双线性插值，写进 `out[..el_tap]`。
    fn elev_taps(&self, ear: usize, lat: f32, beta: f32, out: &mut [f32; ELEV_MAX_TAPS]) {
        let x = (lat.clamp(-90.0, 90.0) + 90.0) / self.el_da;
        let i0 = (x.floor() as usize).min(self.el_na - 2);
        let fa = x - i0 as f32;
        let y = beta.rem_euclid(360.0) / self.el_db;
        let j0 = (y.floor() as usize) % self.el_nb;
        let j1 = (j0 + 1) % self.el_nb;
        let fb = y - y.floor();
        let n = self.el_tap;
        let at = |i: usize, j: usize| {
            let b = ((ear * self.el_na + i) * self.el_nb + j) * n;
            &self.elev[b..b + n]
        };
        let (c00, c01, c10, c11) = (at(i0, j0), at(i0, j1), at(i0 + 1, j0), at(i0 + 1, j1));
        let (w00, w01, w10, w11) = ((1.0 - fa) * (1.0 - fb), (1.0 - fa) * fb, fa * (1.0 - fb), fa * fb);
        for k in 0..n {
            out[k] = w00 * c00[k] + w01 * c01[k] + w10 * c10[k] + w11 * c11[k];
        }
    }

    /// 带高度的查表（`el`：仰角，度）。仰角为 0 时就是 [`lookup`]，结果逐位相同。
    ///
    /// 离开水平面时：两耳时间差、声级差和近场效应主要由"偏向哪只耳朵多少"（侧向角）决定，所以先取
    /// 同一侧向角上水平面前 / 后两点的混合；再给每只耳朵卷上高度修正——耳廓对来声方向的染色随极角的变化，
    /// 取自 KU100 1.5 m 的全球面实测，方向按"从这只耳朵看过去"算（近处的声源有声学视差）。
    /// `tmp` 是调用方提供的临时空间。
    pub fn lookup3(&self, az_sofa: f32, el: f32, r: f32, out: &mut Hrir, tmp: &mut Hrir) {
        if el.abs() <= 1e-4 {
            self.lookup(az_sofa, r, out);
            return;
        }
        let (d, alpha, beta, wf) = cone_coords(az_sofa, el);
        self.lookup(alpha.rem_euclid(360.0), r, out);
        self.lookup((180.0 - alpha).rem_euclid(360.0), r, tmp);
        let rr = r.max(R_MIN);
        let mut taps = [0f32; ELEV_MAX_TAPS];
        let mut he = [0f32; NTAP];
        let n = self.el_tap;
        for e in 0..2 {
            for (a, b) in out.h[e].iter_mut().zip(tmp.h[e].iter()) {
                *a = wf * *a + (1.0 - wf) * b;
            }
            out.tau[e] = wf * out.tau[e] + (1.0 - wf) * tmp.tau[e];
            // 从这只耳朵看声源的侧向角（朝自己这一侧为正）
            let ey = if e == 0 { HEAD_A } else { -HEAD_A };
            let vy = rr * d[1] - ey;
            let len = ((rr * d[0]).powi(2) + vy * vy + (rr * d[2]).powi(2)).sqrt().max(1e-9);
            let lat = (if e == 0 { vy } else { -vy } / len).clamp(-1.0, 1.0).asin().to_degrees();
            self.elev_taps(e, lat, beta, &mut taps);
            let h = &mut out.h[e];
            for (i, o) in he.iter_mut().enumerate() {
                let kmax = i.min(n - 1);
                let mut s = 0.0;
                for k in 0..=kmax {
                    s += taps[k] * h[i - k];
                }
                *o = s;
            }
            for i in 0..16 {
                he[NTAP - 16 + i] *= 0.5 * (1.0 + (std::f32::consts::PI * i as f32 / 15.0).cos());
            }
            *h = he;
        }
    }

    /// 带高度的双耳总功率：同一侧向角上水平面前 / 后两点功率的混合
    /// （高度修正只动 5 kHz 以上，对语音频段的总功率影响很小）。
    pub fn lookup_power3(&self, az_sofa: f32, el: f32, r: f32) -> f32 {
        if el.abs() <= 1e-4 {
            return self.lookup_power(az_sofa, r);
        }
        let (_, alpha, _, wf) = cone_coords(az_sofa, el);
        wf * self.lookup_power(alpha.rem_euclid(360.0), r) + (1.0 - wf) * self.lookup_power((180.0 - alpha).rem_euclid(360.0), r)
    }

    /// 一个扇区、一只耳朵的房间脉冲响应。
    pub fn room_ir(&self, sec: usize, ear: usize) -> &[f32] {
        let base = (sec * 2 + ear) * self.room_len;
        &self.room[base..base + self.room_len]
    }
}
