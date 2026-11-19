//! 直达声：逐样本分数时延（ITD、传播时延、多普勒）+ 时变最小相位 HRIR 卷积。
//! 每个控制块给一组新的目标滤波器和时延，块内从旧的线性过渡到新的，声源移动时没有拉链声。

use super::filters::{dot, SincTable};
use super::{CTRL, NTAP};

/// 分数延时插值器的点数（一半是"向前看"的样本数）。
pub const SINC_TAPS: usize = 24;
/// 固定基础时延（样本）：保证最近的那只耳朵的读位置也落在已有样本之内。
pub const BASE_DELAY: f32 = 16.0;
const XN: usize = 1024; // 单声道历史长度（最大时延约 450 个样本）

pub struct DirectRenderer {
    xbuf: Vec<f32>,
    xpos: usize,
    ebuf: [Vec<f32>; 2],
    epos: usize,
    h_old: [[f32; NTAP]; 2],
    h_new: [[f32; NTAP]; 2],
    d_old: [f32; 2],
    d_new: [f32; 2],
    sinc: SincTable,
    primed: bool,
}

impl DirectRenderer {
    pub fn new() -> Self {
        DirectRenderer {
            xbuf: vec![0.0; 2 * XN],
            xpos: 0,
            ebuf: [vec![0.0; 2 * NTAP], vec![0.0; 2 * NTAP]],
            epos: 0,
            h_old: [[0.0; NTAP]; 2],
            h_new: [[0.0; NTAP]; 2],
            d_old: [BASE_DELAY; 2],
            d_new: [BASE_DELAY; 2],
            sinc: SincTable::new(SINC_TAPS, 512, 6.5, 1.0),
            primed: false,
        }
    }

    /// 控制块开始时调用：给出块末尾（下一个边界）的滤波器和两耳时延（样本）。
    pub fn set_target(&mut self, h: &[[f32; NTAP]; 2], delay: [f32; 2]) {
        if self.primed {
            self.h_old = self.h_new;
            self.d_old = self.d_new;
        } else {
            self.h_old = *h;
            self.d_old = delay;
            self.primed = true;
        }
        self.h_new = *h;
        self.d_new = delay;
    }

    /// 处理控制块内从第 `phase` 个样本开始的一段。
    pub fn process(&mut self, x: &[f32], phase: usize, out_l: &mut [f32], out_r: &mut [f32]) {
        let half = SINC_TAPS / 2 - 1;
        for (i, &xi) in x.iter().enumerate() {
            self.xpos = if self.xpos == 0 { XN - 1 } else { self.xpos - 1 };
            self.xbuf[self.xpos] = xi;
            self.xbuf[self.xpos + XN] = xi;
            self.epos = if self.epos == 0 { NTAP - 1 } else { self.epos - 1 };
            let k = (phase + i) as f32;
            let wd = k / CTRL as f32;
            let wf = (k + 0.5) / CTRL as f32;
            let mut y = [0f32; 2];
            for e in 0..2 {
                let d = (self.d_old[e] + (self.d_new[e] - self.d_old[e]) * wd).clamp(half as f32 + 1.0, (XN - SINC_TAPS - 2) as f32);
                let di = d.floor();
                let start = self.xpos + di as usize - half;
                let v = self.sinc.interp(&self.xbuf[start..start + SINC_TAPS], d - di);
                let eb = &mut self.ebuf[e];
                eb[self.epos] = v;
                eb[self.epos + NTAP] = v;
                let hist = &eb[self.epos..self.epos + NTAP];
                let ya = dot(&self.h_old[e], hist);
                let yb = dot(&self.h_new[e], hist);
                y[e] = ya + (yb - ya) * wf;
            }
            out_l[i] = y[0];
            out_r[i] = y[1];
        }
    }
}
