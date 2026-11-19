//! 真实房间：TH Köln 用同一个 KU100 实测的双耳房间脉冲响应（去掉直达声，12 个方向的扇区）。
//! 输入按声源方向在相邻两个扇区之间分配（权重加在输入端，所以移动中的声源每一段都留在自己的方向上）。
//!
//! 卷积用两级均匀分块的频域卷积（UPOLS）：第一级块长 64（覆盖响应的前 21 ms），第二级块长 1024
//! （覆盖其余部分）。房间响应开头本来就有约 127 个样本的静默（直达声之后 2 ms 才有反射），
//! 正好抵掉分块带来的时延，所以整体不增加任何延迟。

use super::tables::Tables;
use super::CTRL;
use realfft::{num_complex::Complex32, ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const NSEC: usize = 12;
const BA: usize = 64;
const BB: usize = 1024;
const HEAD_A_LEN: usize = BB - BA; // 第一级覆盖的响应长度（960 点）

/// 均匀分块重叠保留卷积：多个输入（扇区）、两个输出（耳朵）。
struct Upols {
    b: usize,
    nbins: usize,
    nparts: usize,
    nsec: usize,
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    h: Vec<Complex32>,   // [sec][ear][part][bin]
    fdl: Vec<Complex32>, // [sec][slot][bin]
    nz: Vec<bool>,       // [sec][slot]
    pos: usize,
    win: Vec<f32>, // [sec][2b]
    prev_nz: Vec<bool>,
    tbuf: Vec<f32>,
    obuf: Vec<f32>,
    acc: [Vec<Complex32>; 2],
    scratch_f: Vec<Complex32>,
    scratch_i: Vec<Complex32>,
}

impl Upols {
    /// `ir(sec, ear)` 返回这一级负责的那段响应。
    fn new<'a>(nsec: usize, b: usize, len: usize, ir: impl Fn(usize, usize) -> &'a [f32]) -> Self {
        let nfft = 2 * b;
        let nbins = b + 1;
        let nparts = len.div_ceil(b).max(1);
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(nfft);
        let c2r = planner.plan_fft_inverse(nfft);
        let mut scratch_f = r2c.make_scratch_vec();
        let scratch_i = c2r.make_scratch_vec();
        let mut h = vec![Complex32::new(0.0, 0.0); nsec * 2 * nparts * nbins];
        let mut tbuf = vec![0f32; nfft];
        for sec in 0..nsec {
            for ear in 0..2 {
                let src = ir(sec, ear);
                for p in 0..nparts {
                    tbuf.fill(0.0);
                    let s0 = p * b;
                    if s0 < src.len() {
                        let s1 = (s0 + b).min(src.len());
                        tbuf[..s1 - s0].copy_from_slice(&src[s0..s1]);
                    }
                    let base = ((sec * 2 + ear) * nparts + p) * nbins;
                    let _ = r2c.process_with_scratch(&mut tbuf, &mut h[base..base + nbins], &mut scratch_f);
                }
            }
        }
        Upols {
            b,
            nbins,
            nparts,
            nsec,
            r2c,
            c2r,
            h,
            fdl: vec![Complex32::new(0.0, 0.0); nsec * nparts * nbins],
            nz: vec![false; nsec * nparts],
            pos: 0,
            win: vec![0.0; nsec * nfft],
            prev_nz: vec![false; nsec],
            tbuf,
            obuf: vec![0.0; nfft],
            acc: [vec![Complex32::new(0.0, 0.0); nbins], vec![Complex32::new(0.0, 0.0); nbins]],
            scratch_f,
            scratch_i,
        }
    }

    /// `input`：[sec][b]；`active[sec]`：这一块该扇区有没有非零输入。输出各 b 个样本。
    fn process(&mut self, input: &[f32], active: &[bool], out_l: &mut [f32], out_r: &mut [f32]) {
        let (b, nbins, nparts) = (self.b, self.nbins, self.nparts);
        let nfft = 2 * b;
        self.pos = (self.pos + 1) % nparts;
        let mut any_slot = false;
        for sec in 0..self.nsec {
            let w = &mut self.win[sec * nfft..(sec + 1) * nfft];
            w.copy_within(b.., 0);
            if active[sec] {
                w[b..].copy_from_slice(&input[sec * b..(sec + 1) * b]);
            } else {
                w[b..].fill(0.0);
            }
            let any = active[sec] || self.prev_nz[sec];
            self.prev_nz[sec] = active[sec];
            let si = sec * nparts + self.pos;
            if any {
                self.tbuf.copy_from_slice(w);
                let slot = &mut self.fdl[si * nbins..(si + 1) * nbins];
                let _ = self.r2c.process_with_scratch(&mut self.tbuf, slot, &mut self.scratch_f);
            }
            self.nz[si] = any;
            any_slot |= self.nz[sec * nparts..(sec + 1) * nparts].iter().any(|&v| v);
        }
        if !any_slot {
            out_l[..b].fill(0.0);
            out_r[..b].fill(0.0);
            return;
        }
        for a in self.acc.iter_mut() {
            a.fill(Complex32::new(0.0, 0.0));
        }
        for sec in 0..self.nsec {
            for p in 0..nparts {
                let slot = (self.pos + nparts - p) % nparts;
                let si = sec * nparts + slot;
                if !self.nz[si] {
                    continue;
                }
                let x = &self.fdl[si * nbins..(si + 1) * nbins];
                for ear in 0..2 {
                    let hb = ((sec * 2 + ear) * nparts + p) * nbins;
                    let h = &self.h[hb..hb + nbins];
                    for ((a, xv), hv) in self.acc[ear].iter_mut().zip(x).zip(h) {
                        *a += xv * hv;
                    }
                }
            }
        }
        let scale = 1.0 / nfft as f32;
        for (ear, out) in [out_l, out_r].into_iter().enumerate() {
            // 实信号的频谱在直流和奈奎斯特处虚部应为 0，清掉舍入误差
            self.acc[ear][0].im = 0.0;
            self.acc[ear][nbins - 1].im = 0.0;
            let _ = self.c2r.process_with_scratch(&mut self.acc[ear], &mut self.obuf, &mut self.scratch_i);
            for (o, v) in out[..b].iter_mut().zip(&self.obuf[b..]) {
                *o = v * scale;
            }
        }
    }
}

/// 定长先进先出队列。
struct Fifo {
    buf: Vec<f32>,
    head: usize,
    len: usize,
}

impl Fifo {
    fn new(cap: usize, prefill: usize) -> Self {
        Fifo { buf: vec![0.0; cap], head: 0, len: prefill }
    }
    #[inline]
    fn push(&mut self, v: f32) {
        let cap = self.buf.len();
        let i = (self.head + self.len) % cap;
        self.buf[i] = v;
        self.len += 1;
    }
    #[inline]
    fn pop(&mut self) -> f32 {
        let v = self.buf[self.head];
        self.head = (self.head + 1) % self.buf.len();
        self.len -= 1;
        v
    }
}

pub struct Room {
    a: Upols,
    b: Upols,
    in_a: Vec<f32>,
    in_b: Vec<f32>,
    fill_a: usize,
    fill_b: usize,
    act_a: [bool; NSEC],
    act_b: [bool; NSEC],
    qa: [Fifo; 2],
    qb: [Fifo; 2],
    w_old: [f32; NSEC],
    w_new: [f32; NSEC],
    g_old: f32,
    g_new: f32,
    tmp: [Vec<f32>; 2],
    primed: bool,
}

/// 声源方位角（顺时针，度）-> 12 个扇区的权重（相邻两个扇区线性分配）。
pub fn sector_weights(az: f32) -> [f32; NSEC] {
    let s = az.rem_euclid(360.0) / (360.0 / NSEC as f32);
    let k0 = (s.floor() as usize) % NSEC;
    let fr = s - s.floor();
    let mut w = [0.0; NSEC];
    w[k0] += 1.0 - fr;
    w[(k0 + 1) % NSEC] += fr;
    w
}

impl Room {
    pub fn new(tab: &'static Tables) -> Self {
        assert_eq!(tab.n_sec, NSEC);
        let len = tab.room_len;
        let a = Upols::new(NSEC, BA, HEAD_A_LEN, |s, e| &tab.room_ir(s, e)[BA..BA + HEAD_A_LEN]);
        let b = Upols::new(NSEC, BB, len - BA - HEAD_A_LEN, |s, e| &tab.room_ir(s, e)[BA + HEAD_A_LEN..]);
        Room {
            a,
            b,
            in_a: vec![0.0; NSEC * BA],
            in_b: vec![0.0; NSEC * BB],
            fill_a: 0,
            fill_b: 0,
            act_a: [false; NSEC],
            act_b: [false; NSEC],
            qa: [Fifo::new(4 * BA, BA), Fifo::new(4 * BA, BA)],
            qb: [Fifo::new(3 * BB, BB), Fifo::new(3 * BB, BB)],
            w_old: [0.0; NSEC],
            w_new: [0.0; NSEC],
            g_old: 1.0,
            g_new: 1.0,
            tmp: [vec![0.0; BB], vec![0.0; BB]],
            primed: false,
        }
    }

    /// 控制块开始时调用：块末尾的声源方位角，和说话人音量增益（忽远忽近时房间声跟着说话人的音量走）。
    pub fn set_target(&mut self, az: f32, talker_gain: f32) {
        let w = sector_weights(az);
        if self.primed {
            self.w_old = self.w_new;
            self.g_old = self.g_new;
        } else {
            self.w_old = w;
            self.g_old = talker_gain;
            self.primed = true;
        }
        self.w_new = w;
        self.g_new = talker_gain;
    }

    /// 处理控制块内从第 `phase` 个样本开始的一段，结果乘 `gain` 后累加进 out。
    pub fn process(&mut self, x: &[f32], phase: usize, gain: f32, out_l: &mut [f32], out_r: &mut [f32]) {
        let mut secs = [0usize; NSEC];
        let mut ns = 0;
        for s in 0..NSEC {
            if self.w_old[s] != 0.0 || self.w_new[s] != 0.0 {
                secs[ns] = s;
                ns += 1;
            }
        }
        for (i, &xi) in x.iter().enumerate() {
            let wp = (phase + i) as f32 / CTRL as f32;
            let v = xi * (self.g_old + (self.g_new - self.g_old) * wp);
            if v != 0.0 {
                for &s in &secs[..ns] {
                    let w = self.w_old[s] + (self.w_new[s] - self.w_old[s]) * wp;
                    if w != 0.0 {
                        self.in_a[s * BA + self.fill_a] = v * w;
                        self.in_b[s * BB + self.fill_b] = v * w;
                        self.act_a[s] = true;
                        self.act_b[s] = true;
                    }
                }
            }
            self.fill_a += 1;
            self.fill_b += 1;
            if self.fill_a == BA {
                let [tl, tr] = &mut self.tmp;
                self.a.process(&self.in_a, &self.act_a, tl, tr);
                for k in 0..BA {
                    self.qa[0].push(tl[k]);
                    self.qa[1].push(tr[k]);
                }
                for s in 0..NSEC {
                    if self.act_a[s] {
                        self.in_a[s * BA..(s + 1) * BA].fill(0.0);
                        self.act_a[s] = false;
                    }
                }
                self.fill_a = 0;
            }
            if self.fill_b == BB {
                let [tl, tr] = &mut self.tmp;
                self.b.process(&self.in_b, &self.act_b, tl, tr);
                for k in 0..BB {
                    self.qb[0].push(tl[k]);
                    self.qb[1].push(tr[k]);
                }
                for s in 0..NSEC {
                    if self.act_b[s] {
                        self.in_b[s * BB..(s + 1) * BB].fill(0.0);
                        self.act_b[s] = false;
                    }
                }
                self.fill_b = 0;
            }
            out_l[i] += gain * (self.qa[0].pop() + self.qb[0].pop());
            out_r[i] += gain * (self.qa[1].pop() + self.qb[1].pop());
        }
    }
}

/// `a` 与 `b` 卷积后前 `keep` 个样本的能量（`b` 为空表示不卷积）。
pub fn conv_energy(a: &[f32], b: Option<&[f32]>, keep: usize) -> f64 {
    match b {
        None => a.iter().take(keep).map(|&v| (v as f64) * (v as f64)).sum(),
        Some(b) => {
            let n = (a.len() + b.len() - 1).min(keep);
            let mut e = 0f64;
            for i in 0..n {
                let k0 = (i + 1).saturating_sub(a.len());
                let k1 = (i + 1).min(b.len());
                let mut s = 0f32;
                for k in k0..k1 {
                    s += b[k] * a[i - k];
                }
                e += (s as f64) * (s as f64);
            }
            e
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::tables::tables;

    /// 两级分块卷积应当与直接卷积逐样本一致（不多不少的时延）。
    #[test]
    fn room_matches_direct_convolution() {
        let tab = tables();
        let mut room = Room::new(tab);
        let n = 6000;
        let mut x = vec![0f32; n];
        let mut seed = 12345u32;
        for v in x.iter_mut().take(3000) {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
        }
        let az = 100.0; // 扇区 3 权重 2/3、扇区 4 权重 1/3
        let (mut ol, mut or) = (vec![0f32; n], vec![0f32; n]);
        let mut i = 0;
        let mut phase = 0;
        while i < n {
            if phase == 0 {
                room.set_target(az, 1.0);
            }
            let m = (CTRL - phase).min(n - i).min(37); // 故意用不整齐的小段
            room.process(&x[i..i + m], phase, 1.0, &mut ol[i..i + m], &mut or[i..i + m]);
            phase = (phase + m) % CTRL;
            i += m;
        }
        let w = sector_weights(az);
        for &t in &[200usize, 700, 1500, 2999, 4000, 5500] {
            let mut refl = 0f64;
            for (s, &ws) in w.iter().enumerate() {
                if ws == 0.0 {
                    continue;
                }
                let ir = tab.room_ir(s, 0);
                for k in 0..=t.min(ir.len() - 1) {
                    refl += (ws * ir[k]) as f64 * x[t - k] as f64;
                }
            }
            assert!((ol[t] as f64 - refl).abs() < 2e-4, "t={t}: {} vs {}", ol[t], refl);
        }
    }
}
