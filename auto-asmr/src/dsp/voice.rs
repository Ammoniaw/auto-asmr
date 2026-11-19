//! 人声预处理：噪声门（响应阈值）、输入降噪（RNNoise）、去齿音、轻声化、贴耳气声。都不改变嗓音本身。

use super::filters::{ema_coef, interp_log, minphase_from_mag, Biquad, OnePoleLp, BUTTER_Q};
use super::{db_to_lin, CTRL_DT, FS};
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

/// 耳甲腔深度共振的频率（Shaw 1979：堵住耳道口时 4.3 kHz，各个方向来的声音都能激发）。
pub const CONCHA_F: f32 = 4300.0;

/// 贴耳时近耳的均衡（最小相位 FIR，64 点）：临场感（3.5 kHz）+ 空气感（9 kHz 以上）−"暗"（5 kHz）
/// + 耳甲共鸣（4.3 kHz，约 0.8 倍频程宽）。全为 0 时返回 None。
pub fn ear_eq_fir(presence_db: f32, dark_db: f32, ku100_eq: bool, concha_db: f32) -> Option<Vec<f32>> {
    if presence_db == 0.0 && dark_db == 0.0 && !ku100_eq && concha_db == 0.0 {
        return None;
    }
    let nfft = 512;
    let mag: Vec<f32> = (0..=nfft / 2)
        .map(|k| {
            let f = (k as f32 * FS / nfft as f32).max(1.0);
            let mut d = if ku100_eq { interp_log(f, &EAR_EQ_F, &EAR_EQ_DB) } else { 0.0 };
            d += presence_db * (-0.5 * ((f / 3500.0).log2() / 0.7).powi(2)).exp() + 0.7 * presence_db / (1.0 + (9000.0 / f).powi(2));
            d -= dark_db * (-0.5 * ((f / 5000.0).log2() / 0.6).powi(2)).exp();
            d += concha_db * (-0.5 * ((f / CONCHA_F).log2() / 0.33).powi(2)).exp();
            db_to_lin(d)
        })
        .collect();
    Some(minphase_from_mag(&mag, nfft, EAR_EQ_TAPS))
}

/// 远耳高频遮挡的转折频率。
pub const FAR_SHADOW_F: f32 = 6000.0;

/// 贴耳时远耳的高频遮挡（最小相位 FIR，64 点）：球头模型低估了近处声源在远耳的高频头影
/// （与 BEM 近距仿真比，离耳 3.8 cm 时远耳 6 kHz 以上偏亮 2–9 dB）。`db` 是最高频压低的量；0 时返回 None。
pub fn far_shadow_fir(db: f32) -> Option<Vec<f32>> {
    if db <= 0.0 {
        return None;
    }
    let nfft = 512;
    let mag: Vec<f32> = (0..=nfft / 2)
        .map(|k| {
            let f = (k as f32 * FS / nfft as f32).max(1.0);
            db_to_lin(-db / (1.0 + (FAR_SHADOW_F / f).powi(2)))
        })
        .collect();
    Some(minphase_from_mag(&mag, nfft, EAR_EQ_TAPS))
}

/// 气流脉冲的标定增益（`puff` = 1 时）。
pub const PUFF_K: f32 = 1.5;
/// 气声带下限：F3 以上送气噪声取代谐波（Klatt & Klatt 1990）。
pub const BREATH_FC: f32 = 2500.0;

/// 贴耳气声（与 Python 版 `breath_fx` 相同）：嘴贴近耳朵时把嗓音自己的气声提上来，并加上气流扑到耳朵上的低频脉冲。
/// 两样都只用嗓音里已有的送气声，不生成任何噪声、不改变嗓音；离开耳边时都为 0。
///
/// * 气声增强：气声带 = 2.5 kHz 以上。只把"轻"的部分提上来（频带内向上压缩）：电平比响的齿音低 16 dB 以上的
///   送气声 / 气息 / 轻辅音提得最多，接近齿音电平的不提，低到底噪附近（低 36–48 dB）的也不提。
/// * 气流扑耳：气声带瞬时功率在 50–130 Hz 的起伏（除以功率的平方根，幅度与气声成正比），幅度跟着人声的电平走，
///   送气越强脉冲越强，爆破音开头最明显。加在直达声的输入上，由近场 HRIR 送到贴着的那只耳朵。
///
/// 电平每个控制块（1.33 ms）检测一次，下一个控制块生效，没有额外延时。
pub struct BreathFx {
    hp: Biquad,
    s_lp: OnePoleLp,
    w_hp: Biquad,
    w_lp: [Biquad; 2],
    acc_b: f64,
    acc_x: f64,
    n: usize,
    env: f64,
    lref: f64,
    envx: f64,
    vref: f64,
    /// 当前可用的提升比例（0–1）。
    g: f32,
    /// "响的齿音"的功率、"响的人声"的幅度。
    pref: f32,
    vamp: f32,
    lift: (f32, f32),
    puff: (f32, f32),
    primed: bool,
}

/// "响的"电平跟踪的初值（dBFS）：从很低开始，听到第一个响的音就跟上去，与输入音量无关。
const BREATH_REF0: f64 = -90.0;

impl BreathFx {
    pub fn new() -> Self {
        BreathFx {
            hp: Biquad::highpass(BREATH_FC, BUTTER_Q),
            s_lp: OnePoleLp::new(30.0),
            w_hp: Biquad::highpass(50.0, BUTTER_Q),
            w_lp: [Biquad::lowpass(130.0, BUTTER_Q); 2],
            acc_b: 0.0,
            acc_x: 0.0,
            n: 0,
            env: 0.0,
            lref: BREATH_REF0,
            envx: 0.0,
            vref: BREATH_REF0,
            g: 0.0,
            pref: 10f64.powf(BREATH_REF0 / 10.0) as f32,
            vamp: 10f64.powf(BREATH_REF0 / 20.0) as f32,
            lift: (0.0, 0.0),
            puff: (0.0, 0.0),
            primed: false,
        }
    }

    /// 控制块开始：`cl` = 块末尾的贴耳程度（0–1），算出这一块里气声增益和气流增益走到哪里。
    pub fn begin_block(&mut self, cl: f32, lift_db: f32, puff: f32) {
        let t = if lift_db > 0.0 { cl * (db_to_lin(lift_db * self.g) - 1.0) } else { 0.0 };
        let u = PUFF_K * puff.max(0.0) * cl;
        if self.primed {
            self.lift = (self.lift.1, t);
            self.puff = (self.puff.1, u);
        } else {
            self.lift = (t, t);
            self.puff = (u, u);
            self.primed = true;
        }
    }

    /// 处理一个样本：`a` = 人声，`wp` = 在控制块里的进度（0–1）。返回（加强后的人声，气流脉冲）。
    #[inline]
    pub fn process(&mut self, a: f32, wp: f32) -> (f32, f32) {
        let b = self.hp.process(a);
        let q = b * b;
        self.acc_b += q as f64;
        self.acc_x += (a * a) as f64;
        self.n += 1;
        let y = a + (self.lift.0 + (self.lift.1 - self.lift.0) * wp) * b;
        let s = self.s_lp.process(q).max(0.0);
        let w = self.w_hp.process(q);
        let w = self.w_lp[0].process(w);
        let w = self.w_lp[1].process(w);
        let gp = self.puff.0 + (self.puff.1 - self.puff.0) * wp;
        if gp <= 0.0 {
            return (y, 0.0);
        }
        // 气声功率的低频起伏 / 功率的平方根 = 与气声幅度成正比；再除以响的齿音幅度，得到与音色明暗无关的量
        let raw = w / ((s + 1e-3 * self.pref) * self.pref + 1e-30).sqrt();
        (y, gp * self.vamp * raw.tanh())
    }

    /// 控制块结束：用这一块的均方更新电平、"响的"电平和提升比例。
    pub fn end_block(&mut self) {
        if self.n == 0 {
            return;
        }
        let dt = CTRL_DT as f64 * self.n as f64 / super::CTRL as f64;
        let (ca, cr) = (1.0 - (-dt / 0.003).exp(), 1.0 - (-dt / 0.03).exp());
        // "响的"电平：50 ms 跟上去；有声音时每秒落 0.5 dB，静下来（低 30 dB 以上）就保持
        let (cu, down) = (1.0 - (-dt / 0.05).exp(), 0.5 * dt);
        let m = self.acc_b / self.n as f64;
        self.env += (if m > self.env { ca } else { cr }) * (m - self.env);
        let lb = 10.0 * (self.env + 1e-12).log10();
        self.lref = if lb > self.lref {
            self.lref + cu * (lb - self.lref)
        } else if lb > self.lref - 30.0 {
            self.lref - down
        } else {
            self.lref
        };
        let m = self.acc_x / self.n as f64;
        self.envx += (if m > self.envx { ca } else { cr }) * (m - self.envx);
        let lx = 10.0 * (self.envx + 1e-12).log10();
        self.vref = if lx > self.vref {
            self.vref + cu * (lx - self.vref)
        } else if lx > self.vref - 30.0 {
            self.vref - down
        } else {
            self.vref
        };
        let d = self.lref - lb;
        let u = ((d - 4.0) / 12.0).clamp(0.0, 1.0);
        let v = ((d - 36.0) / 12.0).clamp(0.0, 1.0);
        self.g = (u * u * (3.0 - 2.0 * u) * (1.0 - v * v * (3.0 - 2.0 * v))) as f32;
        self.pref = 10f64.powf(self.lref / 10.0) as f32;
        self.vamp = 10f64.powf(self.vref / 20.0) as f32;
        self.acc_b = 0.0;
        self.acc_x = 0.0;
        self.n = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::filters::fir_mag;

    /// 远耳高频遮挡：6 kHz 处压一半，最高频接近设定值，500 Hz 基本不动。
    #[test]
    fn far_shadow_shape() {
        let h = far_shadow_fir(5.0).unwrap();
        let m = fir_mag(&h, 4096);
        let db = |f: f32| 20.0 * m[(f / FS * 4096.0).round() as usize].log10();
        assert!(db(500.0).abs() < 0.3, "500 Hz {}", db(500.0));
        assert!((db(6000.0) + 2.5).abs() < 0.4, "6 kHz {}", db(6000.0));
        assert!(db(16000.0) < -4.0 && db(16000.0) > -5.2, "16 kHz {}", db(16000.0));
        assert!(far_shadow_fir(0.0).is_none());
    }

    /// 耳甲共鸣：4.3 kHz 处约 +4 dB，1 kHz 和 12 kHz 基本不动。
    #[test]
    fn concha_peak() {
        let h = ear_eq_fir(0.0, 0.0, false, 4.0).unwrap();
        let m = fir_mag(&h, 4096);
        let db = |f: f32| 20.0 * m[(f / FS * 4096.0).round() as usize].log10();
        assert!((db(4300.0) - 4.0).abs() < 0.4, "4.3 kHz {}", db(4300.0));
        assert!(db(1000.0).abs() < 0.4 && db(12000.0).abs() < 0.4, "1k {} 12k {}", db(1000.0), db(12000.0));
    }

    /// 跑一段信号：返回（提升后 2.5 kHz 以上相对原来的增益 dB，气流脉冲有效值相对输入有效值 dB）。
    fn run_breath(x: &[f32], cl: f32, lift_db: f32, puff: f32, skip: usize) -> (f32, f32) {
        let mut fx = BreathFx::new();
        let mut hp_in = Biquad::highpass(BREATH_FC, BUTTER_Q);
        let mut hp_out = Biquad::highpass(BREATH_FC, BUTTER_Q);
        let (mut ei, mut eo, mut ep, mut ex) = (0f64, 0f64, 0f64, 0f64);
        for (k, blk) in x.chunks(super::super::CTRL).enumerate() {
            fx.begin_block(cl, lift_db, puff);
            for (i, &a) in blk.iter().enumerate() {
                let (y, p) = fx.process(a, i as f32 / super::super::CTRL as f32);
                let (bi, bo) = (hp_in.process(a), hp_out.process(y));
                if k * super::super::CTRL + i >= skip {
                    ei += (bi * bi) as f64;
                    eo += (bo * bo) as f64;
                    ep += (p * p) as f64;
                    ex += (a * a) as f64;
                }
            }
            fx.end_block();
        }
        ((10.0 * (eo / ei.max(1e-30)).log10()) as f32, (10.0 * (ep / ex.max(1e-30)).log10()) as f32)
    }

    /// 气声增强：响的齿音不提，比它低 20 dB 的气声提到接近设定值，离开耳边（贴耳程度 0）时不动，底噪不提。
    #[test]
    fn breath_lift_only_soft() {
        let n = FS as usize * 4;
        let loud = noise(n, -20.0, true);
        // 前 2 s 响（让"响的齿音"电平跟上去），后 2 s 低 20 dB / 低 60 dB；从变轻 0.6 s 后开始量
        let mk = |drop_db: f32| -> Vec<f32> { loud.iter().enumerate().map(|(i, &v)| if i < n / 2 { v } else { v * db_to_lin(-drop_db) }).collect() };
        let half = n / 2 + FS as usize * 6 / 10;
        let (g_loud, _) = run_breath(&loud, 1.0, 6.0, 0.0, half);
        let (g_soft, _) = run_breath(&mk(20.0), 1.0, 6.0, 0.0, half);
        let (g_floor, _) = run_breath(&mk(60.0), 1.0, 6.0, 0.0, half);
        let (g_away, _) = run_breath(&mk(20.0), 0.0, 6.0, 0.0, half);
        assert!(g_loud.abs() < 0.5, "响的不提：{g_loud}");
        assert!(g_soft > 4.5 && g_soft < 6.2, "轻的提到接近 6 dB：{g_soft}");
        assert!(g_floor.abs() < 0.5, "底噪不提：{g_floor}");
        assert!(g_away.abs() < 1e-3, "离开耳边不动：{g_away}");
    }

    /// 气流扑耳：能量集中在 50–130 Hz，强度随 puff 线性变化，离开耳边时为 0。
    #[test]
    fn puff_is_low_frequency() {
        let n = FS as usize * 2;
        // 每 0.25 s 一段 60 ms 的"送气"噪声
        let x: Vec<f32> = noise(n, -20.0, true).iter().enumerate().map(|(i, &v)| if i % 12000 < 2880 { v } else { 0.0 }).collect();
        let mut fx = BreathFx::new();
        let (mut lo, mut hi) = (Biquad::lowpass(200.0, BUTTER_Q), Biquad::highpass(400.0, BUTTER_Q));
        let (mut e_all, mut e_lo, mut e_hi) = (0f64, 0f64, 0f64);
        for blk in x.chunks(super::super::CTRL) {
            fx.begin_block(1.0, 0.0, 1.0);
            for (i, &a) in blk.iter().enumerate() {
                let (y, p) = fx.process(a, i as f32 / super::super::CTRL as f32);
                assert_eq!(y, a, "不开气声增强时人声不变");
                let (l, h) = (lo.process(p), hi.process(p));
                e_all += (p * p) as f64;
                e_lo += (l * l) as f64;
                e_hi += (h * h) as f64;
            }
            fx.end_block();
        }
        assert!(e_all > 0.0);
        assert!(e_lo / e_all > 0.8, "200 Hz 以下占 {}", e_lo / e_all);
        assert!(e_hi / e_all < 0.01, "400 Hz 以上占 {}", e_hi / e_all);
        let (_, p1) = run_breath(&x, 1.0, 0.0, 1.0, 0);
        let (_, p2) = run_breath(&x, 1.0, 0.0, 2.0, 0);
        let (_, p0) = run_breath(&x, 0.0, 0.0, 1.0, 0);
        assert!((p2 - p1 - 6.02).abs() < 0.1, "puff 加倍 +6 dB：{}", p2 - p1);
        assert!(p1 > -30.0 && p1 < 0.0, "脉冲电平相对输入 {p1} dB");
        assert!(p0 < -200.0, "离开耳边没有脉冲：{p0}");
    }

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
