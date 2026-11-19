//! 内嵌的 KU100 数据：近场 HRIR 查找表、时延、双耳功率、扩散场均衡、实测真实房间、耳机补偿滤波器。
//! 由 `tools/export_assets.py` 从 Python 版的预处理结果导出（TH Köln，CC BY 4.0）。

use super::{NTAP, R_MIN};
use std::sync::OnceLock;

static DATA: &[u8] = include_bytes!("../../assets/ku100.bin");

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
        let mut c = Cur { d: DATA, p: 0 };
        assert_eq!(c.bytes(8), b"ASMRKU02", "assets/ku100.bin 格式不对");
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
        Tables { n_r, n_az, inv_grid, h, tau, power, cdfc, n_sec, room_len, room, headphones }
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

    /// 一个扇区、一只耳朵的房间脉冲响应。
    pub fn room_ir(&self, sec: usize, ear: usize) -> &[f32] {
        let base = (sec * 2 + ear) * self.room_len;
        &self.room[base..base + self.room_len]
    }
}
