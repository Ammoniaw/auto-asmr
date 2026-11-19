//! 人声预处理：噪声门（响应阈值）、输入降噪（RNNoise）、去齿音、轻声化。都不改变嗓音本身。

use super::filters::{ema_coef, interp_log, minphase_from_mag, Biquad, BUTTER_Q};
use super::{db_to_lin, FS};
use nnnoiseless::DenoiseState;

const DT: f32 = 1.0 / FS;

/// 噪声门：输入低于阈值时平滑地关掉（开 3 ms、保持 0.2 s、关 0.15 s），带 3 dB 回差。
pub struct Gate {
    env: f32,
    open: bool,
    hold: f32,
    g: f32,
}

impl Gate {
    pub fn new() -> Self {
        Gate { env: 0.0, open: true, hold: 0.0, g: 1.0 }
    }

    #[inline]
    pub fn process(&mut self, x: f32, thr_db: f32) -> f32 {
        if thr_db <= -89.0 {
            self.g = 1.0;
            return x;
        }
        let a = x.abs();
        self.env = if a > self.env { a } else { self.env + ema_coef(0.05, DT) * (a - self.env) };
        let thr = db_to_lin(thr_db);
        if self.env > thr {
            self.open = true;
            self.hold = 0.2;
        } else if self.env < thr * 0.708 {
            self.hold -= DT;
            if self.hold <= 0.0 {
                self.open = false;
            }
        }
        let target = if self.open { 1.0 } else { 0.0 };
        let tau = if target > self.g { 0.003 } else { 0.15 };
        self.g += ema_coef(tau, DT) * (target - self.g);
        x * self.g
    }
}

/// RNNoise 一帧 480 个样本，内部还有一帧重叠，所以打开降噪会增加 960 个样本（20 ms）的延时。
pub const DENOISE_DELAY: usize = 2 * DenoiseState::FRAME_SIZE;
const FRAME: usize = DenoiseState::FRAME_SIZE;

/// 输入降噪：RNNoise（nnnoiseless，纯 CPU 的小型循环网络）。`mix` 控制强度，干声路径对齐后混合。
pub struct Denoiser {
    st: Box<DenoiseState<'static>>,
    inb: [f32; FRAME],
    outb: [f32; FRAME],
    fill: usize,
    dry: Vec<f32>,
    dpos: usize,
    first: bool,
}

impl Denoiser {
    pub fn new() -> Self {
        Denoiser { st: DenoiseState::new(), inb: [0.0; FRAME], outb: [0.0; FRAME], fill: 0, dry: vec![0.0; DENOISE_DELAY], dpos: 0, first: true }
    }

    /// 输入一个样本，返回 20 ms 之前那个样本降噪后的结果。
    #[inline]
    pub fn process(&mut self, x: f32, mix: f32) -> f32 {
        let wet = self.outb[self.fill];
        self.inb[self.fill] = x * 32767.0; // RNNoise 按 16 位整数的量级工作
        self.fill += 1;
        if self.fill == FRAME {
            self.st.process_frame(&mut self.outb, &self.inb);
            if self.first {
                self.outb.fill(0.0); // 第一帧是淡入的残留，丢掉
                self.first = false;
            }
            for v in self.outb.iter_mut() {
                *v /= 32767.0;
            }
            self.fill = 0;
        }
        let dry = self.dry[self.dpos];
        self.dry[self.dpos] = x;
        self.dpos = (self.dpos + 1) % DENOISE_DELAY;
        dry + (wet - dry) * mix.clamp(0.0, 1.0)
    }
}

/// 去齿音：4.5 kHz 以上的齿音带能量接近整体能量时（"s / sh / x"），按 3:1 动态压低，最多 `max_db`。
/// 用 4 阶 Linkwitz–Riley 分频（低 + 高 = 全通），零延时，其他时刻只多一个全通相移。
pub struct DeEsser {
    lp: [Biquad; 2],
    hp: [Biquad; 2],
    det_lp: Biquad,
    eb: f32,
    ef: f32,
    pk: f32,
    gr: f32,
}

impl DeEsser {
    pub fn new() -> Self {
        DeEsser {
            lp: [Biquad::lowpass(4500.0, BUTTER_Q); 2],
            hp: [Biquad::highpass(4500.0, BUTTER_Q); 2],
            det_lp: Biquad::lowpass(11000.0, BUTTER_Q),
            eb: 0.0,
            ef: 0.0,
            pk: 1e-6,
            gr: 0.0,
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32, max_db: f32) -> f32 {
        let lo = self.lp[0].process(x);
        let lo = self.lp[1].process(lo);
        let hi = self.hp[0].process(x);
        let hi = self.hp[1].process(hi);
        let band = self.det_lp.process(hi);
        let c = ema_coef(0.002, DT);
        self.eb += c * (band * band - self.eb);
        self.ef += c * (x * x - self.ef);
        self.pk = (self.pk * (1.0 - ema_coef(3.0, DT))).max(self.ef);
        let mut target = 0.0;
        if self.ef > self.pk * 1e-6 && self.ef > 1e-12 {
            let over = 10.0 * ((self.eb + 1e-14) / (self.ef + 1e-14)).log10() + 8.0;
            target = (over * (2.0 / 3.0)).clamp(0.0, max_db.max(0.0));
        }
        let tau = if target > self.gr { 0.001 } else { 0.02 };
        self.gr += ema_coef(tau, DT) * (target - self.gr);
        lo + hi * db_to_lin(-self.gr)
    }
}

// ANSI S3.5-1997 表 3：不同说话力度的标准语音频谱（1/3 倍频程，dB）
const ANSI_F: [f32; 18] = [160., 200., 250., 315., 400., 500., 630., 800., 1000., 1250., 1600., 2000., 2500., 3150., 4000., 5000., 6300., 8000.];
const ANSI_NORMAL: [f32; 18] = [32.41, 34.48, 34.75, 33.98, 34.59, 34.27, 32.06, 28.3, 25.01, 23.0, 20.15, 17.32, 13.18, 11.55, 9.33, 5.31, 2.59, 1.13];
const ANSI_RAISED: [f32; 18] = [33.81, 33.92, 38.98, 38.57, 39.11, 40.15, 38.78, 36.37, 33.86, 31.89, 28.58, 25.32, 22.35, 20.15, 16.78, 11.47, 7.67, 5.07];

pub const SOFT_TAPS: usize = 512;

/// 轻声化的均衡器（最小相位 FIR）：按 ANSI S3.5"正常"与"提高"力度的频谱形状差反向施加 `amount` 倍——
/// 中频略收、低频更饱满、最高频气声略多（Brungart & Scott 2001：说话力度是语音距离感的主要线索）。
pub fn soft_voice_fir(amount: f32) -> Vec<f32> {
    let mut diff = [0f32; 18];
    let mut mean = 0.0;
    for i in 0..18 {
        diff[i] = ANSI_RAISED[i] - ANSI_NORMAL[i];
        mean += diff[i] / 18.0;
    }
    let mut shape = [0f32; 18];
    for i in 0..18 {
        shape[i] = (-(diff[i] - mean)).clamp(-3.0, 3.0);
    }
    let mut sm = [0f32; 18];
    for i in 0..18usize {
        let a = shape[i.saturating_sub(1)];
        let b = shape[(i + 1).min(17)];
        sm[i] = (a + shape[i] + b) / 3.0;
    }
    let nfft = 4096;
    let mag: Vec<f32> = (0..=nfft / 2)
        .map(|k| {
            let f = (k as f32 * FS / nfft as f32).max(20.0);
            db_to_lin(amount * interp_log(f, &ANSI_F, &sm))
        })
        .collect();
    minphase_from_mag(&mag, nfft, SOFT_TAPS)
}

// 近耳音色校正（对齐 KU100 近耳 / 居中差），--ku100-eq 时使用
const EAR_EQ_F: [f32; 9] = [500., 2000., 4000., 6000., 8000., 10000., 12500., 16000., 22000.];
const EAR_EQ_DB: [f32; 9] = [0.0, 0.0, -1.5, -4.0, -5.0, -5.5, -4.0, -3.0, -3.0];

pub const EAR_EQ_TAPS: usize = 64;

/// 贴耳时近耳的均衡（最小相位 FIR，64 点）：临场感（3.5 kHz）+ 空气感（9 kHz 以上）−"暗"（5 kHz）。
/// 全为 0 时返回 None。
pub fn ear_eq_fir(presence_db: f32, dark_db: f32, ku100_eq: bool) -> Option<Vec<f32>> {
    if presence_db == 0.0 && dark_db == 0.0 && !ku100_eq {
        return None;
    }
    let nfft = 512;
    let mag: Vec<f32> = (0..=nfft / 2)
        .map(|k| {
            let f = (k as f32 * FS / nfft as f32).max(1.0);
            let mut d = if ku100_eq { interp_log(f, &EAR_EQ_F, &EAR_EQ_DB) } else { 0.0 };
            d += presence_db * (-0.5 * ((f / 3500.0).log2() / 0.7).powi(2)).exp() + 0.7 * presence_db / (1.0 + (9000.0 / f).powi(2));
            d -= dark_db * (-0.5 * ((f / 5000.0).log2() / 0.6).powi(2)).exp();
            db_to_lin(d)
        })
        .collect();
    Some(minphase_from_mag(&mag, nfft, EAR_EQ_TAPS))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::filters::fir_mag;

    #[test]
    fn soft_voice_shape() {
        let h = soft_voice_fir(1.0);
        let m = fir_mag(&h, 4096);
        let db = |f: f32| 20.0 * m[(f / FS * 4096.0).round() as usize].log10();
        // Python 版实测：低频约 +2.7 dB，1–2.5 kHz 约 −2.5…−2.8 dB，最高频约 +1.5 dB
        assert!(db(200.0) > 1.8 && db(200.0) < 3.2, "200 Hz {}", db(200.0));
        assert!(db(1600.0) < -2.0 && db(1600.0) > -3.2, "1.6 kHz {}", db(1600.0));
        assert!(db(8000.0) > 0.8 && db(8000.0) < 2.2, "8 kHz {}", db(8000.0));
    }

    /// 确定性的噪声：white = 白噪声；否则一阶低通过的"风扇 / 空调"式低频噪声。有效值 `rms_db` dBFS。
    fn noise(n: usize, rms_db: f32, white: bool) -> Vec<f32> {
        let mut seed = 12345u32;
        let mut lp = 0.0f32;
        let mut x: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let w = (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
                lp += 0.05 * (w - lp);
                if white {
                    w
                } else {
                    lp
                }
            })
            .collect();
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
        let g = crate::dsp::db_to_lin(rms_db) / rms;
        x.iter_mut().for_each(|v| *v *= g);
        x
    }

    fn rms_db(x: &[f32]) -> f32 {
        10.0 * (x.iter().map(|v| (v * v) as f64).sum::<f64>() / x.len() as f64 + 1e-30).log10() as f32
    }

    fn run_denoiser(x: &[f32], mix: f32) -> Vec<f32> {
        let mut d = Denoiser::new();
        x.iter().map(|&v| d.process(v, mix)).collect()
    }

    fn attenuation(x: &[f32], db: f32) -> f32 {
        let p = crate::dsp::params::Params { denoise_db: db, ..Default::default() };
        let y = run_denoiser(x, p.denoise_mix());
        rms_db(&x[3 * 48000..]) - rms_db(&y[3 * 48000..])
    }

    #[test]
    fn denoise_strength_is_a_fixed_number_of_db() {
        use crate::dsp::params::DENOISE_MAX_DB;
        const DENOISE_STEPS: [f32; 5] = [6.0, 12.0, 18.0, 24.0, 30.0];
        // 稳定的低频底噪（风扇 / 空调一类），降噪器自己能把它压得很干净：设多少分贝就降多少分贝
        let fan = noise(6 * 48000, -40.0, false);
        for db in DENOISE_STEPS {
            let att = attenuation(&fan, db);
            assert!((att - db).abs() < 0.6, "set {db} dB, got {att} dB");
        }
        // 拖到头（100）不设上限，比 30 dB 干净得多；60 以上降噪器自己已经到底，差别很小
        assert!(attenuation(&fan, DENOISE_MAX_DB) > 40.0);
        assert!(attenuation(&fan, 60.0) > 40.0);
        // 0 就是原样通过
        assert!(attenuation(&fan, 0.0).abs() < 0.01);
        // 嘶嘶的白噪声，降噪器自己只能压低二十多分贝：实际降的不会超过设定值，小档位仍然基本到位
        let hiss = noise(6 * 48000, -55.0, true);
        for db in DENOISE_STEPS {
            let att = attenuation(&hiss, db);
            assert!(att < db + 0.3 && att > db.min(24.0) - 3.0, "hiss: set {db} dB, got {att} dB");
        }
    }

    #[test]
    fn denoiser_keeps_the_voice_and_stays_aligned() {
        // 合成的"人声"（带谐波的浊音 + 一点噪声）：降噪后音量基本不变；
        // 原声那一路和降噪后那一路严格对齐（差一个样本，按比例混合时就会出梳状滤波）
        let v = crate::cli::test_voice(6.0);
        let p = crate::dsp::params::Params { denoise_db: 12.0, ..Default::default() };
        for mix in [p.denoise_mix(), 1.0] {
            let y = run_denoiser(&v, mix);
            let (i0, n) = (2 * 48000usize, 3 * 48000usize);
            let a = &v[i0..i0 + n];
            let corr = |lag: i32| -> f64 { a.iter().enumerate().map(|(i, &u)| u as f64 * y[(i0 + DENOISE_DELAY + i) as i32 as usize + lag as usize] as f64).sum() };
            let best = (-8i32..=8).max_by(|&p, &q| corr(p).total_cmp(&corr(q))).unwrap();
            assert_eq!(best, 0);
            let change = rms_db(&y[i0 + DENOISE_DELAY..i0 + DENOISE_DELAY + n]) - rms_db(a);
            assert!(change.abs() < 1.5, "voice level changed by {change} dB");
        }
    }

    #[test]
    fn deesser_is_transparent_on_vowels() {
        let mut d = DeEsser::new();
        let w = 2.0 * std::f32::consts::PI * 300.0 / FS;
        let (mut ein, mut eout) = (0.0, 0.0);
        for n in 0..48000 {
            let x = 0.3 * (w * n as f32).sin();
            let y = d.process(x, 6.0);
            if n > 4800 {
                ein += x * x;
                eout += y * y;
            }
        }
        assert!((10.0 * (eout / ein).log10()).abs() < 0.05);
    }
}
