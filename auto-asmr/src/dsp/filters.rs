//! 基础滤波器：双二阶、一阶低通、FIR（时域）、最小相位 FIR 设计（倒谱法）、分数延时用的加窗 sinc 表。

use super::FS;
use rustfft::{num_complex::Complex32, FftPlanner};
use std::f32::consts::PI;

/// 8 路累加的点积，编译器可以自动向量化。
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// 双二阶（转置直接 II 型）。
#[derive(Clone, Copy, Debug)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Default for Biquad {
    fn default() -> Self {
        Biquad { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0, z1: 0.0, z2: 0.0 }
    }
}

impl Biquad {
    pub fn from_coeffs(b: [f32; 3], a: [f32; 3]) -> Self {
        Biquad { b0: b[0] / a[0], b1: b[1] / a[0], b2: b[2] / a[0], a1: a[1] / a[0], a2: a[2] / a[0], z1: 0.0, z2: 0.0 }
    }

    /// 二阶低通（q = 0.7071 即 2 阶巴特沃斯，与 scipy.signal.butter(2, fc) 相同）。
    pub fn lowpass(fc: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * fc / FS;
        let (s, c) = w0.sin_cos();
        let al = s / (2.0 * q);
        Self::from_coeffs([(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0], [1.0 + al, -2.0 * c, 1.0 - al])
    }

    /// 任意采样率下的二阶低通（控制信号用，比如每个控制块更新一次的曲线）。
    pub fn lowpass_at(fc: f32, q: f32, fs: f32) -> Self {
        let w0 = 2.0 * PI * fc / fs;
        let (s, c) = w0.sin_cos();
        let al = s / (2.0 * q);
        Self::from_coeffs([(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0], [1.0 + al, -2.0 * c, 1.0 - al])
    }

    /// 把状态设成"输入一直是 v"的稳态（直流增益为 1 的滤波器），避免从 0 起步的瞬变。
    pub fn prime(&mut self, v: f32) {
        self.z1 = v * (1.0 - self.b0);
        self.z2 = v * (self.b2 - self.a2);
    }

    pub fn highpass(fc: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * fc / FS;
        let (s, c) = w0.sin_cos();
        let al = s / (2.0 * q);
        Self::from_coeffs([(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0], [1.0 + al, -2.0 * c, 1.0 - al])
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

}

pub const BUTTER_Q: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// 一阶巴特沃斯低通（双线性变换，与 scipy.signal.butter(1, fc) 相同）。
#[derive(Clone, Copy, Default)]
pub struct OnePoleLp {
    b: f32,
    a1: f32,
    x1: f32,
    y1: f32,
}

impl OnePoleLp {
    pub fn new(fc: f32) -> Self {
        let k = (PI * fc / FS).tan();
        OnePoleLp { b: k / (1.0 + k), a1: (k - 1.0) / (k + 1.0), x1: 0.0, y1: 0.0 }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b * (x + self.x1) - self.a1 * self.y1;
        self.x1 = x;
        self.y1 = y;
        y
    }
}

/// 指数平滑系数：时间常数 tau 秒、每步 dt 秒。
#[inline]
pub fn ema_coef(tau: f32, dt: f32) -> f32 {
    1.0 - (-dt / tau.max(1e-6)).exp()
}

/// 时域 FIR：环形缓冲写两份，取连续切片做点积。
#[derive(Clone)]
pub struct Fir {
    taps: Vec<f32>,
    buf: Vec<f32>,
    pos: usize,
}

impl Fir {
    pub fn new(taps: &[f32]) -> Self {
        let n = taps.len().max(1);
        let mut t = taps.to_vec();
        t.resize(n, 0.0);
        Fir { taps: t, buf: vec![0.0; 2 * n], pos: 0 }
    }

    pub fn len(&self) -> usize {
        self.taps.len()
    }

    /// 换一组等长的系数，保留历史（不产生咔哒声以外的瞬变由调用方负责交叉淡化）。
    pub fn set_taps(&mut self, taps: &[f32]) {
        if taps.len() == self.taps.len() {
            self.taps.copy_from_slice(taps);
        } else {
            *self = Fir::new(taps);
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let n = self.taps.len();
        self.pos = if self.pos == 0 { n - 1 } else { self.pos - 1 };
        self.buf[self.pos] = x;
        self.buf[self.pos + n] = x;
        dot(&self.taps, &self.buf[self.pos..self.pos + n])
    }
}

/// 幅度谱（nfft/2+1 点）-> 最小相位 FIR（ntap 点），倒谱法；末尾 1/8 做余弦淡出。与 Python 版相同。
pub fn minphase_from_mag(mag: &[f32], nfft: usize, ntap: usize) -> Vec<f32> {
    assert_eq!(mag.len(), nfft / 2 + 1);
    let mut planner = FftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(nfft);
    let inv = planner.plan_fft_inverse(nfft);
    let mx = mag.iter().cloned().fold(0f32, f32::max).max(1e-30);
    let mut buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); nfft];
    for k in 0..=nfft / 2 {
        let lm = mag[k].max(1e-7 * mx).ln();
        buf[k] = Complex32::new(lm, 0.0);
        if k > 0 && k < nfft / 2 {
            buf[nfft - k] = Complex32::new(lm, 0.0);
        }
    }
    inv.process(&mut buf);
    let scale = 1.0 / nfft as f32;
    for (i, v) in buf.iter_mut().enumerate() {
        let w = if i == 0 || i == nfft / 2 {
            1.0
        } else if i < nfft / 2 {
            2.0
        } else {
            0.0
        };
        *v = Complex32::new(v.re * scale * w, 0.0);
    }
    fwd.process(&mut buf);
    for v in buf.iter_mut() {
        *v = v.exp();
    }
    inv.process(&mut buf);
    let mut h: Vec<f32> = buf[..ntap].iter().map(|v| v.re * scale).collect();
    let nf = ntap / 8;
    for i in 0..nf {
        let t = i as f32 / (nf.max(2) - 1) as f32;
        h[ntap - nf + i] *= 0.5 * (1.0 + (PI * t).cos());
    }
    h
}

/// FIR 的幅度谱（nfft/2+1 点），测试里核对均衡曲线用。
#[cfg(test)]
pub fn fir_mag(h: &[f32], nfft: usize) -> Vec<f32> {
    let mut planner = FftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(nfft);
    let mut buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); nfft];
    for (b, &v) in buf.iter_mut().zip(h.iter()) {
        b.re = v;
    }
    fwd.process(&mut buf);
    buf[..=nfft / 2].iter().map(|v| v.norm()).collect()
}

/// 在对数频率上对 (频率, dB) 表做线性插值（两端保持），相当于 np.interp(log f, log F, dB)。
pub fn interp_log(f: f32, freqs: &[f32], db: &[f32]) -> f32 {
    let lf = f.max(1e-3).ln();
    if lf <= freqs[0].ln() {
        return db[0];
    }
    for i in 1..freqs.len() {
        let (l0, l1) = (freqs[i - 1].ln(), freqs[i].ln());
        if lf <= l1 {
            return db[i - 1] + (db[i] - db[i - 1]) * (lf - l0) / (l1 - l0);
        }
    }
    db[db.len() - 1]
}

/// 零阶修正贝塞尔函数 I0（Kaiser 窗用）。
fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let hx = (x as f64) / 2.0;
    for k in 1..30 {
        term *= (hx / k as f64) * (hx / k as f64);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum as f32
}

/// Kaiser 窗加窗 sinc 的多相表：`phases` 个相位 × `taps` 点，用于分数延时和重采样。
/// 第 p 行、第 k 点 = sinc(cutoff·(f + m))·cutoff·窗，f = p/phases，m = taps/2 − 1 − k。
pub struct SincTable {
    pub taps: usize,
    pub phases: usize,
    table: Vec<f32>,
}

impl SincTable {
    pub fn new(taps: usize, phases: usize, beta: f32, cutoff: f32) -> Self {
        let mut table = vec![0f32; (phases + 1) * taps];
        let half = taps as f32 / 2.0;
        let i0b = bessel_i0(beta);
        for p in 0..=phases {
            let f = p as f32 / phases as f32;
            let row = &mut table[p * taps..(p + 1) * taps];
            let mut sum = 0.0;
            for (k, v) in row.iter_mut().enumerate() {
                let m = (taps / 2) as f32 - 1.0 - k as f32;
                let t = f + m; // 目标时刻相对该样本的偏移
                let x = PI * cutoff * t;
                let s = if x.abs() < 1e-6 { 1.0 } else { x.sin() / x };
                let u = t / half;
                let w = if u.abs() >= 1.0 { 0.0 } else { bessel_i0(beta * (1.0 - u * u).sqrt()) / i0b };
                *v = s * cutoff * w;
                sum += *v;
            }
            if cutoff >= 0.999 {
                // 纯延时：把直流增益归一到 1
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
        SincTable { taps, phases, table }
    }

    /// 取小数部分为 frac（0..1）的一行系数，相邻两行线性插值后与 `x`（taps 点，下标增大 = 更旧的样本）做点积。
    #[inline]
    pub fn interp(&self, x: &[f32], frac: f32) -> f32 {
        let fp = frac.clamp(0.0, 1.0) * self.phases as f32;
        let p0 = (fp as usize).min(self.phases - 1);
        let a = fp - p0 as f32;
        let r0 = &self.table[p0 * self.taps..(p0 + 1) * self.taps];
        let r1 = &self.table[(p0 + 1) * self.taps..(p0 + 2) * self.taps];
        (1.0 - a) * dot(r0, x) + a * dot(r1, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butter_matches_scipy() {
        // scipy.signal.butter(2, 40, 'high', fs=48000) 的系数
        let hp = Biquad::highpass(40.0, BUTTER_Q);
        assert!((hp.b0 - 0.996_304_7).abs() < 1e-5);
        assert!((hp.a1 + 1.992_595_8).abs() < 1e-5);
        assert!((hp.a2 - 0.992_623_0).abs() < 1e-5);
    }

    #[test]
    fn minphase_flat_is_delta() {
        let mag = vec![1.0f32; 257];
        let h = minphase_from_mag(&mag, 512, 64);
        assert!((h[0] - 1.0).abs() < 1e-4);
        assert!(h[1..].iter().all(|v| v.abs() < 1e-4));
    }

    #[test]
    fn sinc_delay_is_accurate() {
        // 1 kHz 正弦延时 0.3 个样本
        let t = SincTable::new(24, 512, 6.5, 1.0);
        let w = 2.0 * PI * 1000.0 / FS;
        let x: Vec<f32> = (0..24).map(|j| ((100.0 + 11.0 - j as f32) * w).sin()).collect(); // x[j] = s(n - i + 11 - j)，n − i = 100
        let y = t.interp(&x, 0.3);
        assert!((y - ((100.0 - 0.3) * w).sin()).abs() < 2e-3);
    }
}
