//! 背景音：实时生成的雨声（"雨夜"）。不用录音素材，按雨滴落地的物理过程现算；实时（主播）、试听和文件转换
//! 用的是同一套代码。音量为 0 时整个模块不参与，输出与没有这个模块时逐字节相同。人声本身一点不动。
//!
//! 组成（做法和文献见 doc/effects.md「背景音：雨声」）：
//!
//! 1. **整片雨声**。雨量 R（mm/h）下雨滴直径按 Marshall–Palmer 谱 N(D) = 8000·e^(−ΛD) /m³/mm（Λ = 4.1·R^−0.21 /mm），
//!    下落速度 v(D) = 9.65 − 10.3·e^(−0.6D) m/s（Atlas 等 1973）。雨滴均匀落在以听者为中心的三个圆环里
//!    （近 0.5–3 m、中 3–12 m、远 12–50 m，耳朵离地 1.2 m），每一滴从它自己的方向（KU100 HRIR，含仰角）、
//!    按 1/距离 衰减地到达两耳，叠加起来就是包围着人的雨声；两耳之间的相关、远近的明暗都是自然形成的。
//!    一次撞击是一个很短的压力脉冲（Miklavcic 等 2004）：开头一个尖峰（接触瞬间），再跟一个宽度等于接触时间
//!    D/v 的单周期正弦（雨滴减速的力的变化率），峰值 ∝ D·v³ / 距离——大雨滴更响、更低沉，小雨滴是轻轻的"嗒"。
//!    真实的雨滴每秒几十万个，每个圆环只逐个渲染最大的几千个（它们决定"噼啪"的颗粒感），更小的那些合成成
//!    能量相同、正负号随机的稀疏脉冲（"天鹅绒噪声"，每秒两千个以上听起来就是平滑的沙沙声）。
//!    McDermott & Simoncelli（2011）：雨这类纹理靠"稀疏"（零星而集中的能量）和各频带同步起伏才像，
//!    只把频谱做对的噪声认不出是雨——所以这里是一滴一滴叠出来的，不是滤波白噪声。
//!    雨量还会以十几秒的节奏轻轻起伏（±25%），不是一成不变的。
//! 2. **流动**（参考一段 8D 环绕雨声，见 doc/effects.md）：雨点落在哪个方向不是一直均匀的，有一个"重心"
//!    绕着头慢慢转（平均 26 秒一圈，快慢随机变化），那边落得密一些。流动 50 时 1–4 kHz 左右差的起伏
//!    （标准差约 1.5 dB）和参考一样；0 = 不动。
//! 3. **落地声**：身边 0.5–3 m 的地上（泥地、草、积水）一个个听得出的雨点，低沉的"噗"——150–600 Hz 的一团
//!    闷响，几十毫秒，开头带一点点溅水的"沙"；也按 KU100 HRIR 摆在四周地面上的各个方向。频谱和长短按参考
//!    视频里的落地声做，滑块 60 时和参考一样明显，默认 30（低 6 dB）。雨越大落得越多。
//! 4. **屋檐滴水**：1–3 处固定的位置，各有各的节奏（带一点随机和慢慢的漂移），"嗒"——一下撞击加一小团
//!    溅开的水，各处溅水的中心频率不同。多响跟着落地声。
//! 5. **隔窗**：外面的雨声（连同落地声）按墙和玻璃的隔声（越高的频率挡得越多）低通，再漏一点进来；
//!    屋檐滴水挡一半。
//! 6. **响度**：按整片雨声的期望功率（每一种雨滴的频谱 × 方向分布下的 HRIR × 各级滤波 × 回放补偿 × K 加权）
//!    解析地算出增益，让雨声的积分响度 = 人声的目标响度 + 音量（dB）+ 雨量带来的 ±5 dB，不用先跑一遍去测。
//!    落地声和滴水按整片雨声的功率定各自的响度，所以调它们不会改变"沙沙"的大小。
//!
//! 7. **雷声**（可选，默认没有）：见 [`crate::dsp::thunder`]，在这里和雨声一起出、一起闷、一起定增益。
//!
//! 没有气泡声（水洼里的"叮"、滴水的"咕嘟"）和窗玻璃上的"叮咚"：第一版有，试听觉得敲玻璃一样吵、水泡声
//! 恶心（2026-11-20），都去掉了。

use super::dynamics::KWeight;
use super::filters::{Biquad, Fir, BUTTER_Q};
use super::params::Params;
use super::rng::Rng;
use super::tables::{tables, Hrir};
use super::thunder::{self, Thunder};
use super::{CTRL, FS, NTAP};
use rustfft::{num_complex::Complex32, FftPlanner};
use std::f32::consts::{PI, TAU};
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------- 方向网格和脉冲

/// 方位每 5° 一个方向（SOFA 方位角：逆时针，0 = 正前，90 = 左）。
pub(super) const N_AZ: usize = 72;
pub(super) const AZ_STEP: f32 = 5.0;
/// 仰角 5 档：地面上的雨滴在耳朵下方（近处更靠下），窗户大约和耳朵齐平，树叶在上面。
const ELS: [f32; 5] = [-55.0, -35.0, -18.0, -5.0, 15.0];
const N_EL: usize = ELS.len();
const N_DIR: usize = N_AZ * N_EL;
/// 窗户方向（隔窗时外面的雨从窗户进来）、屋檐滴水落地处的仰角档。
const EL_WINDOW: usize = 3;
const EL_LOW: usize = 1;

/// 撞击脉冲宽度（= 接触时间 D/v）的 4 档，毫秒：0.4 mm 的小雨滴约 0.26 ms，6 mm 的大雨滴约 0.64 ms。
const TAUS_MS: [f32; 4] = [0.26, 0.33, 0.43, 0.56];
/// 落在软的表面（树叶、草）上：表面会让开，接触时间长约 3.5 倍，声音更低（0.5–1.5 kHz 的"啪嗒"；
/// 热带雨林录音里雨打植物最明显的是 0.6–1.2 kHz 和 4.4–5.6 kHz 两段：Metcalf 等）。
/// 脉冲表前 4 档是硬表面，后 4 档是软表面。
const SOFT_STRETCH: f32 = 2.5;
const N_HARD: usize = TAUS_MS.len();
const N_PULSE: usize = 2 * N_HARD;
const PULSE_LEN: usize = 96;
/// 脉冲开头的尖峰（接触瞬间的水锤压力，只有几微秒：Miklavcic 等 2004），相对单周期正弦的峰值：硬表面、软表面。
/// 它决定 5 kHz 以上的"沙沙"：取 2.5 时 8 kHz、16 kHz 两个倍频程比总能量低约 14–16 dB。
const CRACK: f32 = 3.0;
const CRACK_SOFT: f32 = 1.5;
/// 脉冲里单极成分（半个正弦）的大小：硬表面、软表面。
const MONO_HARD: f32 = 0.0;
const MONO_SOFT: f32 = 1.1;
/// 落在软表面上的比例和它的强度（相对硬表面）。
const P_SOFT: f32 = 0.4;
const SOFT_GAIN: f32 = 0.7;

/// HRIR（含两耳时延和分数延时插值的余量）的长度；卷上脉冲之后的长度。
pub(super) const H_LEN: usize = 192;
const B_LEN: usize = H_LEN + PULSE_LEN;
/// 定响度用的频率网格。
const NFFT: usize = 512;
const NB: usize = NFFT / 2 + 1;
/// 各条总线的环形缓冲（样本）；一个事件最长 GRAIN_MAX + H_LEN，提前不到一个控制块安排，放得下。
pub(super) const RING: usize = 16384;
pub(super) const GRAIN_MAX: usize = 8192;

// ---------------------------------------------------------------- 雨滴物理

/// Marshall–Palmer 谱的 N₀（/m³/mm）。
const N0: f64 = 8000.0;
/// 小于这个直径（mm）的雨滴几乎没有声音；大于 6 mm 的在空中就碎了。
const D_LO: f32 = 0.4;
const D_MAX: f32 = 6.0;
/// 耳朵离地多高（米）。
pub(super) const H_EAR: f32 = 1.2;
/// 三个圆环（米）、每个圆环逐个渲染的大雨滴和"底子"每秒各多少个、圆环按距离的低通（远处的雨被空气、地面和
/// 植被吃掉高频），以及远处两个圆环被建筑、地形、植被挡掉之后剩下的功率比例（空旷平地上每翻一倍距离贡献一样多，
/// 实际上远处总有遮挡）。
const RINGS: [(f32, f32); 3] = [(0.5, 3.0), (3.0, 12.0), (12.0, 50.0)];
const CAP_TOP: [f64; 3] = [2800.0, 2400.0, 2400.0];
const CAP_FLOOR: [f64; 3] = [1600.0, 2400.0, 3000.0];
const LP_FC: [f32; 3] = [10000.0, 8000.0, 6000.0];
const RING_W: [f32; 3] = [1.0, 0.7, 0.35];
/// 每一滴撞击强度的随机起伏（撞击角度、溅开的样子）：对数正态的 σ。
const ZETA_SIGMA: f32 = 0.4;
/// 雨量的慢起伏：时间常数（秒）和对数标准差。
const LFO_TAU: f32 = 14.0;
const LFO_SIGMA: f32 = 0.25;
/// 流动：重心转一圈的平均秒数、转速慢慢变化的时间常数（秒）和幅度（对数标准差）、流动 100 时的集中度
/// （von Mises κ：方向 θ 上落得多密 ∝ e^{κ·cos(θ − 重心)}；κ = FLOW_KAPPA ×（流动/100）²）。
/// 按参考的 8D 雨声定：流动 50（κ = 0.5）时 1–4 kHz 左右差（0.5 s）的标准差约 1.5 dB、自相关约 4–5 s 降到一半，
/// 和参考一样；100（κ = 2）时左右摆动约 ±5–7 dB。
const FLOW_PERIOD: f32 = 26.0;
const FLOW_TAU: f32 = 20.0;
const FLOW_SPEED_SIGMA: f32 = 0.45;
const FLOW_KAPPA: f32 = 2.0;
/// 落地声：雨量 0 / 100 时每秒几个（中间按雨量线性）。
const LAND_RATE: (f32, f32) = (1.0, 5.0);
/// 落地声的大小：一个"噗"的能量 = 整片雨声 LAND_T 秒的能量再高（低）多少 dB——滑块 LAND_REF 时为 LAND_DB，
/// 每 1 格 LAND_STEP dB。LAND_DB 按参考视频定：150–600 Hz 每秒约 0.8 个比背景高 8 dB 以上、峰值中位约 9 dB。
const LAND_T: f32 = 0.08;
const LAND_DB: f32 = -5.5;
const LAND_REF: f32 = 60.0;
const LAND_STEP: f32 = 0.2;
/// 落地声的远近：落在 0.7–3 m 的地上（耳朵离地 1.2 m）。
const LAND_R: (f32, f32) = (0.7, 3.0);
/// 屋檐滴水一下的能量比一个落地声高（低）多少 dB（"嗒"短而亮，同样的能量听起来更突出）。
const DRIP_REL_DB: f32 = 2.0;

/// 下落速度（m/s），直径 D（mm）。
#[inline]
fn v_term(d: f32) -> f32 {
    (9.65 - 10.3 * (-0.6 * d).exp()).max(0.0)
}

/// 一滴撞击的强度（未乘距离衰减和随机起伏）：∝ D·v³。
#[inline]
fn w_raw(d: f32) -> f32 {
    1e-3 * d * v_term(d).powi(3)
}

/// 直径 D 的雨滴的脉冲宽度落在哪一档。
#[inline]
fn bucket(d: f32) -> usize {
    let tau = d / v_term(d).max(0.5); // 毫秒（mm / (m/s) = ms）
    let mut best = 0;
    for (i, &t) in TAUS_MS.iter().enumerate() {
        if (tau / t).ln().abs() < (tau / TAUS_MS[best]).ln().abs() {
            best = i;
        }
    }
    best
}

/// 落在离脚下水平距离 r（米）处的雨滴在耳朵看来的仰角档（−atan(1.2 / r)：1.2 m 以内 −55°，2.4 m 以内 −35°……）。
#[inline]
fn el_of_r(r: f32) -> usize {
    if r < 1.2 {
        0
    } else if r < 2.41 {
        1
    } else if r < 5.89 {
        2
    } else {
        3
    }
}

// ---------------------------------------------------------------- 预先算好的方向表

struct Bank {
    /// `[N_DIR][2][H_LEN]`：1.5 m 处的 KU100 HRIR，两耳时延（分数延时）已经做进去。
    hrir: Vec<f32>,
    /// `[N_PULSE][N_DIR][2][B_LEN]`：撞击脉冲卷上 HRIR。
    pulse: Vec<f32>,
    /// `[N_DIR][2][NB]`：HRIR 的功率谱 |H|²。
    hpow: Vec<f32>,
    /// `[N_PULSE][NB]`：单位峰值脉冲的功率谱。
    ppow: Vec<f32>,
}

impl Bank {
    #[inline]
    fn hrir(&self, dir: usize, ear: usize) -> &[f32] {
        let b = (dir * 2 + ear) * H_LEN;
        &self.hrir[b..b + H_LEN]
    }

    #[inline]
    fn pulse(&self, p: usize, dir: usize, ear: usize) -> &[f32] {
        let b = ((p * N_DIR + dir) * 2 + ear) * B_LEN;
        &self.pulse[b..b + B_LEN]
    }

    fn build() -> Bank {
        let tab = tables();
        let (mut h, mut tmp) = (Hrir::default(), Hrir::default());
        let mut raw = Vec::with_capacity(N_DIR);
        let (mut tmin, mut tmax) = (f32::MAX, f32::MIN);
        for &el in ELS.iter() {
            for a in 0..N_AZ {
                tab.lookup3(a as f32 * AZ_STEP, el, 1.5, &mut h, &mut tmp);
                tmin = tmin.min(h.tau[0]).min(h.tau[1]);
                tmax = tmax.max(h.tau[0]).max(h.tau[1]);
                raw.push(h.clone());
            }
        }
        assert!(NTAP as f32 + (tmax - tmin) + 18.0 <= H_LEN as f32, "两耳时延太大：{tmin} … {tmax}");
        let mut hrir = vec![0f32; N_DIR * 2 * H_LEN];
        for (d, hr) in raw.iter().enumerate() {
            for e in 0..2 {
                let b = (d * 2 + e) * H_LEN;
                frac_delay(&hr.h[e], hr.tau[e] - tmin + 8.0, &mut hrir[b..b + H_LEN]);
            }
        }
        let shapes: Vec<[f32; PULSE_LEN]> = TAUS_MS.iter().map(|&t| pulse_shape(t, CRACK, MONO_HARD)).chain(TAUS_MS.iter().map(|&t| pulse_shape(t * SOFT_STRETCH, CRACK_SOFT, MONO_SOFT))).collect();
        let mut pulse = vec![0f32; N_PULSE * N_DIR * 2 * B_LEN];
        for (p, sh) in shapes.iter().enumerate() {
            for d in 0..N_DIR {
                for e in 0..2 {
                    let hs = &hrir[(d * 2 + e) * H_LEN..(d * 2 + e + 1) * H_LEN];
                    let b = ((p * N_DIR + d) * 2 + e) * B_LEN;
                    let out = &mut pulse[b..b + B_LEN];
                    for (i, &x) in sh.iter().enumerate() {
                        if x != 0.0 {
                            for (o, &hv) in out[i..i + H_LEN].iter_mut().zip(hs) {
                                *o += x * hv;
                            }
                        }
                    }
                }
            }
        }
        let mut planner = FftPlanner::<f32>::new();
        let mut hpow = vec![0f32; N_DIR * 2 * NB];
        for (i, chunk) in hrir.chunks_exact(H_LEN).enumerate() {
            let s = power_spectrum(&mut planner, chunk, NFFT);
            hpow[i * NB..(i + 1) * NB].copy_from_slice(&s);
        }
        let mut ppow = vec![0f32; N_PULSE * NB];
        for (p, sh) in shapes.iter().enumerate() {
            ppow[p * NB..(p + 1) * NB].copy_from_slice(&power_spectrum(&mut planner, sh, NFFT));
        }
        Bank { hrir, pulse, hpow, ppow }
    }
}

fn bank() -> &'static Bank {
    static B: OnceLock<Bank> = OnceLock::new();
    B.get_or_init(Bank::build)
}

/// |FFT(x)|²，前 n/2+1 个点（n ≥ x.len()）。
fn power_spectrum(planner: &mut FftPlanner<f32>, x: &[f32], n: usize) -> Vec<f32> {
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<Complex32> = (0..n).map(|i| Complex32::new(x.get(i).copied().unwrap_or(0.0), 0.0)).collect();
    fft.process(&mut buf);
    buf[..n / 2 + 1].iter().map(|c| c.norm_sqr()).collect()
}

/// 把 n 点（n = NFFT·r）功率谱的第 k 档附近 r 个点平均，变成 NFFT 点网格上的值（帕塞瓦尔下能量不变）。
fn fold(s: &[f32], k: usize, r: usize) -> f32 {
    if r <= 1 {
        return s[k];
    }
    let c = (k * r) as isize;
    let half = (r / 2) as isize;
    let mut acc = 0.0;
    for j in (c - half)..(c - half + r as isize) {
        acc += s[j.unsigned_abs().min(s.len() - 1)];
    }
    acc / r as f32
}

/// 分数延时（16 点加汉宁窗的 sinc）：把 h 往后挪 `delay`（≥ 8）个样本写进 out。
pub(super) fn frac_delay(h: &[f32], delay: f32, out: &mut [f32]) {
    const HW: i32 = 8;
    out.fill(0.0);
    for (k, &v) in h.iter().enumerate() {
        if v == 0.0 {
            continue;
        }
        let x = k as f32 + delay;
        let i0 = x.floor() as i32;
        for j in (i0 - HW + 1)..=(i0 + HW) {
            if j < 0 || j as usize >= out.len() {
                continue;
            }
            let t = j as f32 - x;
            let s = if t.abs() < 1e-6 { 1.0 } else { (PI * t).sin() / (PI * t) };
            out[j as usize] += v * s * (0.5 + 0.5 * (PI * t / HW as f32).cos());
        }
    }
}

/// 一次撞击的压力脉冲：开头的尖峰 + 宽 τ 的单周期正弦（雨滴减速的力的变化率，双极、没有净面积）
/// + `mono` × 宽 τ 的半个正弦（单极：软的表面被压下去、挤开空气，带来 100–500 Hz 的"噗"）。
fn pulse_shape(tau_ms: f32, crack: f32, mono: f32) -> [f32; PULSE_LEN] {
    let t = tau_ms * 1e-3 * FS;
    assert!(t < PULSE_LEN as f32);
    let mut p = [0f32; PULSE_LEN];
    p[0] = crack;
    for (i, v) in p.iter_mut().enumerate() {
        let x = (i as f32 + 0.5) / t;
        if x < 1.0 {
            *v += (TAU * x).sin() + mono * (PI * x).sin();
        }
    }
    p
}

/// 逆 CDF 表：`dens` 在 [a, b] 上的分布。表按 t = −ln(1 − u)（对数生存函数）均匀取点，
/// 指数衰减的尾巴在 t 上接近直线，插值不会把最大的那一小撮雨滴摊大（按 u 均匀取点时会明显高估大雨滴）。
const NCDF: usize = 256;
const T_MAX: f64 = 14.0;
type Cdf = [f32; NCDF + 1];

fn inv_cdf(a: f32, b: f32, dens: impl Fn(f32) -> f64) -> Cdf {
    const N: usize = 8192;
    let dx = (b - a) / N as f32;
    let mut cum = vec![0f64; N + 1];
    let mut prev = dens(a);
    for i in 1..=N {
        let cur = dens(a + i as f32 * dx);
        cum[i] = cum[i - 1] + 0.5 * (prev + cur) * dx as f64;
        prev = cur;
    }
    let total = cum[N].max(1e-300);
    let mut out = [a; NCDF + 1];
    let mut j = 0;
    for (i, o) in out.iter_mut().enumerate() {
        let t = T_MAX * i as f64 / NCDF as f64;
        let target = total * (1.0 - (-t).exp());
        while j < N - 1 && cum[j + 1] < target {
            j += 1;
        }
        let (c0, c1) = (cum[j], cum[j + 1]);
        let f = if c1 > c0 { ((target - c0) / (c1 - c0)).clamp(0.0, 1.0) } else { 0.0 };
        *o = a + (j as f32 + f as f32) * dx;
    }
    out
}

#[inline]
fn sample_cdf(t: &Cdf, u: f32) -> f32 {
    let x = (-((1.0 - u as f64).max(1e-12)).ln() / T_MAX * NCDF as f64) as f32;
    if x >= NCDF as f32 {
        return t[NCDF];
    }
    let i = x as usize;
    t[i] + (t[i + 1] - t[i]) * (x - i as f32)
}

/// 方位的分布（按 72 个方向，和为 1）：以 `center`（SOFA 度）为中心、±`half` 度以内，再按"偏向一侧的程度" `k`（0–1）
/// 往中心集中：(1 + k·cos Δ)^(1+3k)。
fn az_weights(center: f32, half: f32, k: f32) -> [f32; N_AZ] {
    let mut w = [0f32; N_AZ];
    let q = 1.0 + 3.0 * k;
    for (i, v) in w.iter_mut().enumerate() {
        let d = ((i as f32 * AZ_STEP - center + 180.0).rem_euclid(360.0) - 180.0).abs();
        let inside = ((half + AZ_STEP - d) / AZ_STEP).clamp(0.0, 1.0);
        *v = inside * ((1.0 + k * d.to_radians().cos()) / (1.0 + k)).max(0.0).powf(q);
    }
    let s: f32 = w.iter().sum();
    if s <= 0.0 {
        w[((center.rem_euclid(360.0) / AZ_STEP).round() as usize) % N_AZ] = 1.0;
    } else {
        w.iter_mut().for_each(|v| *v /= s);
    }
    w
}

#[inline]
pub(super) fn sample_az(cdf: &[f32; N_AZ], u: f32) -> usize {
    cdf.partition_point(|&c| c < u).min(N_AZ - 1)
}

#[inline]
fn az_index(az_sofa: f32) -> usize {
    ((az_sofa.rem_euclid(360.0) / AZ_STEP).round() as usize) % N_AZ
}

// ---------------------------------------------------------------- 一份设置（在界面线程里算好）

/// 由参数算好的雨声设置：各圆环要渲染的雨滴、方向分布、各种事件的多少、滤波器和定响度用的期望功率。
/// 在界面线程里算（约 1 ms），交给音频线程。
pub struct RainSetup {
    /// 比人声的目标响度低多少（dB）、雨量带来的偏移（dB）。
    pub level_db: f32,
    pub trim_db: f32,
    top_rate: [f32; 3],
    top_cdf: [Cdf; 3],
    floor_rate: [f32; 3],
    floor_cdf: [Cdf; 3],
    floor_w: [f32; 3],
    az_cdf: [f32; N_AZ],
    center: f32,
    /// 外面的雨从窗户方向进来的比例（隔窗时）。
    win_p: f32,
    indoor: f32,
    /// 流动：重心的集中度（von Mises κ，0 = 不动）。
    flow: f32,
    /// 落地声每秒几个；落地声为 0 时 0。
    land_rate: f32,
    /// 屋檐滴水（0–1）；落地声为 0 时也是 0（滴水的响度跟着落地声）。
    drip: f32,
    /// 整片雨声在增益 1 时的期望 K 加权功率（两耳之和，每样本）。
    p_unit: f32,
    /// 一个落地声（距离取平均）、一下滴水在增益 1 时的能量（两耳之和）。
    land_energy: f32,
    drip_energy: f32,
    /// 雷声（0–1，0 = 没有）。
    thunder: f32,
}

impl RainSetup {
    /// 音量为 0 时 None（不生成雨声）。`playback`：回放补偿 FIR（雨声和人声过同一个）。
    pub fn new(p: &Params, playback: Option<&[f32]>) -> Option<RainSetup> {
        if p.bg_level <= 0.0 {
            return None;
        }
        let bank = bank();
        let x = (p.rain / 100.0).clamp(0.0, 1.0);
        let rain_mmh = 0.3 * 200f32.powf(x);
        let lam = 4.1 * rain_mmh.powf(-0.21);
        let m = (p.bg_indoor / 100.0).clamp(0.0, 1.0);
        let pan = (p.bg_pan / 100.0).clamp(-1.0, 1.0);
        let width = (p.bg_width / 100.0).clamp(0.0, 1.0);
        let center = -90.0 * pan;
        let azw = az_weights(center, 5.0 + 175.0 * width, pan.abs());
        let mut az_cdf = [0f32; N_AZ];
        let mut acc = 0.0;
        for (c, w) in az_cdf.iter_mut().zip(azw.iter()) {
            acc += w;
            *c = acc;
        }
        az_cdf[N_AZ - 1] = 1.0;
        let win_p = 0.7 * m;

        // 直径上的积分（梯形）：雨滴通量 φ(D) = e^(−ΛD)·v(D)，功率权重 φ·w²
        const ND: usize = 800;
        let dd = (D_MAX - D_LO) / ND as f32;
        let ds: Vec<f32> = (0..=ND).map(|i| D_LO + i as f32 * dd).collect();
        let phi = |d: f32| (-(lam as f64) * d as f64).exp() * v_term(d) as f64;
        let pw = |d: f32| phi(d) * (w_raw(d) as f64).powi(2);
        let mut cum_phi = vec![0f64; ND + 1];
        for i in 1..=ND {
            cum_phi[i] = cum_phi[i - 1] + 0.5 * (phi(ds[i - 1]) + phi(ds[i])) * dd as f64;
        }
        // 从 d 到 D_MAX 的通量积分
        let tail = |d: f32| -> f64 {
            let x = ((d - D_LO) / dd).clamp(0.0, ND as f32);
            let i = (x as usize).min(ND - 1);
            let c = cum_phi[i] + (cum_phi[i + 1] - cum_phi[i]) * (x - i as f32) as f64;
            cum_phi[ND] - c
        };
        // 按脉冲档、[a, b] 区间的功率积分
        let pow_b = |a: f32, b: f32| -> [f64; N_HARD] {
            let mut out = [0f64; N_HARD];
            for i in 0..ND {
                let (d0, d1) = (ds[i], ds[i + 1]);
                let mid = 0.5 * (d0 + d1);
                if mid < a || mid > b {
                    continue;
                }
                out[bucket(mid)] += 0.5 * (pw(d0) + pw(d1)) * dd as f64;
            }
            out
        };
        let e_zeta2 = (2.0 * (ZETA_SIGMA as f64).powi(2)).exp();

        let mut top_rate = [0f32; 3];
        let mut floor_rate = [0f32; 3];
        let mut floor_w = [0f32; 3];
        let mut top_cdf = [[0f32; NCDF + 1]; 3];
        let mut floor_cdf = [[0f32; NCDF + 1]; 3];
        // 每个圆环、每个仰角档、每个脉冲档的功率（每秒，未乘方向和滤波）
        let mut pow_cel = [[[0f64; N_PULSE]; N_EL]; 3];
        // 硬 / 软表面：同一档雨滴，一部分落在软表面上（脉冲更宽、强度 × SOFT_GAIN）
        let split = |b: usize, v: f64| -> [(usize, f64); 2] { [(b, v * (1.0 - P_SOFT) as f64), (b + N_HARD, v * (P_SOFT * SOFT_GAIN * SOFT_GAIN) as f64)] };
        for c in 0..3 {
            let (ri, ro) = RINGS[c];
            let area = PI as f64 * (ro as f64 * ro as f64 - ri as f64 * ri as f64);
            // 按 r 的积分：E[1/R² · 1(仰角档)] 和各仰角档的面积占比
            let mut e_r2 = [0f64; N_EL];
            let mut a_el = [0f64; N_EL];
            const NR: usize = 400;
            let dr = (ro - ri) / NR as f32;
            for j in 0..NR {
                let r = ri + (j as f32 + 0.5) * dr;
                let ring = 2.0 * PI as f64 * r as f64 * dr as f64 / area;
                let el = el_of_r(r);
                e_r2[el] += ring / (r as f64 * r as f64 + (H_EAR * H_EAR) as f64);
                a_el[el] += ring;
            }
            // 隔窗：一部分改从窗户方向（−5°）来
            let mix = |v: [f64; N_EL]| -> [f64; N_EL] {
                let s: f64 = v.iter().sum();
                let mut o = v.map(|x| x * (1.0 - win_p as f64));
                o[EL_WINDOW] += s * win_p as f64;
                o
            };
            let (e_r2, a_el) = (mix(e_r2).map(|v| v * RING_W[c] as f64), mix(a_el));
            let inv_r2: f64 = e_r2.iter().sum();
            let total = area * N0 * tail(D_LO);
            let d_r = if total <= CAP_TOP[c] {
                D_LO
            } else {
                // 二分：area·N0·∫_{d_r}^{D_MAX} φ = CAP_TOP
                let (mut lo, mut hi) = (D_LO, D_MAX);
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    if area * N0 * tail(mid) > CAP_TOP[c] {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                0.5 * (lo + hi)
            };
            top_rate[c] = (area * N0 * tail(d_r)) as f32;
            top_cdf[c] = inv_cdf(d_r, D_MAX, phi);
            let ptop = pow_b(d_r, D_MAX);
            for el in 0..N_EL {
                for b in 0..N_HARD {
                    for (k, v) in split(b, area * N0 * ptop[b] * e_zeta2 * e_r2[el]) {
                        pow_cel[c][el][k] += v;
                    }
                }
            }
            if d_r > D_LO + 1e-3 {
                let pf = pow_b(D_LO, d_r);
                let p_small = area * N0 * pf.iter().sum::<f64>() * e_zeta2 * inv_r2; // inv_r2 里已经乘了 RING_W
                floor_rate[c] = CAP_FLOOR[c] as f32;
                floor_w[c] = (p_small / CAP_FLOOR[c]).sqrt() as f32;
                floor_cdf[c] = inv_cdf(D_LO, d_r, pw);
                let sum_pf: f64 = pf.iter().sum::<f64>().max(1e-300);
                for el in 0..N_EL {
                    for b in 0..N_HARD {
                        for (k, v) in split(b, p_small * (pf[b] / sum_pf) * a_el[el]) {
                            pow_cel[c][el][k] += v;
                        }
                    }
                }
            } else {
                floor_cdf[c] = [D_LO; NCDF + 1];
            }
        }

        // 频率网格上的各级滤波 |·|²
        let freqs: Vec<f32> = (0..NB).map(|k| k as f32 * FS / NFFT as f32).collect();
        let mf = Muffle::new(m);
        let hp = Biquad::highpass(60.0, BUTTER_Q);
        let kw = KWeight::new();
        let pb2: Vec<f32> = match playback {
            Some(h) => {
                let n = h.len().next_power_of_two().max(NFFT);
                let mut planner = FftPlanner::<f32>::new();
                let s = power_spectrum(&mut planner, h, n);
                (0..NB).map(|k| fold(&s, k, n / NFFT)).collect()
            }
            None => vec![1.0; NB],
        };
        let common: Vec<f32> = freqs
            .iter()
            .zip(&pb2)
            .map(|(&f, &pb)| mf.mag2(f) * hp.mag2(f) * kw.mag2(f) * pb)
            .collect();
        let lp2: Vec<Vec<f32>> = LP_FC.iter().map(|&fc| { let b = Biquad::lowpass(fc, BUTTER_Q); freqs.iter().map(|&f| b.mag2(f)).collect() }).collect();
        // 方向平均的 HRIR 功率谱（每个仰角档、每只耳朵）
        let mut dav = vec![0f32; N_EL * 2 * NB];
        for el in 0..N_EL {
            for (a, &w) in azw.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let d = el * N_AZ + a;
                for e in 0..2 {
                    let src = &bank.hpow[(d * 2 + e) * NB..(d * 2 + e + 1) * NB];
                    let dst = &mut dav[(el * 2 + e) * NB..(el * 2 + e + 1) * NB];
                    for (o, &s) in dst.iter_mut().zip(src) {
                        *o += w * s;
                    }
                }
            }
        }
        let wk = |k: usize| if k == 0 || k == NB - 1 { 1.0f64 } else { 2.0 };
        let mut p_imp = 0f64;
        for c in 0..3 {
            for el in 0..N_EL {
                for b in 0..N_PULSE {
                    let pw = pow_cel[c][el][b];
                    if pw <= 0.0 {
                        continue;
                    }
                    let pp = &bank.ppow[b * NB..(b + 1) * NB];
                    let mut s = 0f64;
                    for e in 0..2 {
                        let dv = &dav[(el * 2 + e) * NB..(el * 2 + e + 1) * NB];
                        for k in 0..NB {
                            s += wk(k) * (pp[k] * dv[k] * lp2[c][k] * common[k]) as f64;
                        }
                    }
                    p_imp += pw * s;
                }
            }
        }
        p_imp /= FS as f64 * NFFT as f64;
        let p_unit = p_imp as f32;

        // 落地声：滑块 0 = 没有；1–100 每格 LAND_STEP dB，LAND_REF 时和参考视频一样明显
        let land = p.rain_near.clamp(0.0, 100.0);
        let thunder = (p.bg_thunder / 100.0).clamp(0.0, 1.0);
        if thunder > 0.0 {
            thunder::prepare();
        }
        let land_energy = if land > 0.0 { p_unit * LAND_T * FS * 10f32.powf((LAND_DB + LAND_STEP * (land - LAND_REF)) / 10.0) } else { 0.0 };
        Some(RainSetup {
            level_db: p.bg_db().unwrap_or(-40.0),
            trim_db: 10.0 * x - 5.0,
            top_rate,
            top_cdf,
            floor_rate,
            floor_cdf,
            floor_w,
            az_cdf,
            center,
            win_p,
            indoor: m,
            flow: FLOW_KAPPA * (p.bg_flow / 100.0).clamp(0.0, 1.0).powi(2),
            land_rate: if land > 0.0 { LAND_RATE.0 + (LAND_RATE.1 - LAND_RATE.0) * x } else { 0.0 },
            drip: if land > 0.0 { (p.rain_drip / 100.0).clamp(0.0, 1.0) } else { 0.0 },
            p_unit,
            land_energy,
            drip_energy: land_energy * 10f32.powf(DRIP_REL_DB / 10.0),
            thunder,
        })
    }

    /// 屋檐滴水有几处（0–3）和最快那处的间隔（秒）。
    fn drip_spots(&self) -> (usize, f32) {
        if self.drip <= 0.0 {
            return (0, 0.0);
        }
        let n = if self.drip < 0.34 {
            1
        } else if self.drip < 0.67 {
            2
        } else {
            3
        };
        (n, 2.8 * 0.25f32.powf(self.drip))
    }
}

/// 隔窗的滤波：墙和玻璃按质量定律隔声（一阶低通，每倍频程约 6 dB），双层玻璃 / 墙体在一百多赫兹的共振处透得多一点
/// （峰值），再漏一点原声进来。
#[derive(Clone, Copy)]
struct Muffle {
    lp: Biquad,
    pk: Biquad,
    leak: f32,
}

impl Muffle {
    /// 隔窗 m（0–1）：一阶低通的拐点从 20 kHz 降到 900 Hz，125 Hz 的共振最多 +14 dB，漏进来的原声从 100% 降到 2%。
    /// 这几个数按关着窗的夜雨录音校过：隔窗 100 时 125 Hz–4 kHz 各倍频程与录音相差 3 dB 以内。
    fn new(m: f32) -> Muffle {
        let m = m.clamp(0.0, 1.0);
        let fc = 20000.0 * (900.0f32 / 20000.0).powf(m);
        let leak = (1.0 - m).powi(2) + 0.02 * m;
        Muffle { lp: Biquad::lowpass1(fc.min(0.45 * FS)), pk: Biquad::peaking(125.0, 0.9, 14.0 * m), leak }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        (1.0 - self.leak) * self.lp.process(self.pk.process(x)) + self.leak * x
    }

    fn retune(&mut self, o: &Muffle) {
        self.lp.retune(&o.lp);
        self.pk.retune(&o.pk);
        self.leak = o.leak;
    }

    /// 频率 f 处的功率增益。
    fn mag2(&self, f: f32) -> f32 {
        let (a, b) = self.lp.response(f);
        let (c, d) = self.pk.response(f);
        let (re, im) = (a * c - b * d, a * d + b * c);
        let (re, im) = ((1.0 - self.leak) * re + self.leak, (1.0 - self.leak) * im);
        re * re + im * im
    }
}

/// 一句话说明（写进转换完成后的报告）；没开背景音时为空。
pub fn describe(p: &Params, english: bool) -> String {
    let Some(db) = p.bg_db() else { return String::new() };
    let mmh = 0.3 * 200f32.powf((p.rain / 100.0).clamp(0.0, 1.0));
    if english {
        format!("Background: rain, {:.0} dB below the voice ({mmh:.1} mm/h, flow {:.0}, window {:.0}, landing drops {:.0}, drips {:.0}, thunder {:.0})", -db, p.bg_flow, p.bg_indoor, p.rain_near, p.rain_drip, p.bg_thunder)
    } else {
        format!("背景音：雨声，比人声低 {:.0} dB（雨量约 {mmh:.1} mm/h，流动 {:.0}，隔窗 {:.0}，落地声 {:.0}，屋檐滴水 {:.0}，雷声 {:.0}）", -db, p.bg_flow, p.bg_indoor, p.rain_near, p.rain_drip, p.bg_thunder)
    }
}

// ---------------------------------------------------------------- 运行时

/// 屋檐下的一处滴水：下一滴的时间、慢慢漂移的节奏，溅水的中心频率。
#[derive(Clone, Copy)]
struct Drip {
    next: f64,
    drift: f32,
    f0: f32,
}

/// 各处滴水相对"雨声中心方向"的方位（度）、仰角档、距离（米）、节奏倍数、溅水中心频率的范围（Hz）。
const DRIP_SPOTS: [(f32, usize, f32, f32, f32, f32); 3] = [(50.0, EL_LOW, 1.0, 1.0, 1250.0, 1600.0), (-65.0, 2, 1.8, 1.43, 1900.0, 2500.0), (160.0, EL_LOW, 2.6, 0.71, 2800.0, 3600.0)];

/// 雨声发生器。`render` 写出加了增益的雨声（两耳），调用方加到人声上。
pub struct Rain {
    s: Arc<RainSetup>,
    off: bool,
    rng: Rng,
    /// 各用各的随机数：整片雨声、落地声、屋檐滴水、流动——调一个滑块不会打乱别的层。
    rng_ev: Rng,
    rng_drip: Rng,
    rng_flow: Rng,
    pos: u64,
    next_top: [f64; 3],
    next_floor: [f64; 3],
    next_land: f64,
    drips: [Drip; 3],
    bed: [[Vec<f32>; 2]; 3],
    out: [Vec<f32>; 2],
    half: [Vec<f32>; 2],
    lp: [[Biquad; 2]; 3],
    mf: [Muffle; 2],
    hf: [Muffle; 2],
    m_cur: f32,
    hp: [Biquad; 2],
    pb: Option<[Fir; 2]>,
    pb_src: Option<Arc<Vec<f32>>>,
    gain: f32,
    gain_t: f32,
    /// 关掉那一刻的增益：淡到比它低 80 dB 就算淡完。
    off_from: f32,
    lfo: f32,
    lfo_sigma: f32,
    rho: f32,
    /// 流动：重心的方向（弧度，SOFA：逆时针，0 = 正前）、往哪边转（±1）、转速的慢变化，
    /// 以及这一块里各个方位的接受概率 e^{κ(cos Δ − 1)}。
    phi: f32,
    spin: f32,
    flow_ou: f32,
    acc: [f32; N_AZ],
    grain: Vec<f32>,
    bin: [Vec<f32>; 2],
    /// 雷声（见 [`crate::dsp::thunder`]）和它自己的隔窗、高通（30 Hz：雷的隆隆声在 60 Hz 以下还有不少）。
    thunder: Thunder,
    th_mf: [Muffle; 2],
    th_hp: [Biquad; 2],
    /// 统计：生成了多少个落地声、滴水。
    pub n_land: u64,
    pub n_drips: u64,
}

impl Rain {
    pub fn new(s: Arc<RainSetup>, seed: u64, playback: Option<&Arc<Vec<f32>>>) -> Self {
        let rng_ev = Rng::new(seed ^ 0x0052_4149_4E45_5645);
        let mut rng_drip = Rng::new(seed ^ 0x0052_4149_4E44_5250);
        let mut rng_flow = Rng::new(seed ^ 0x0052_4149_4E46_4C4F);
        let mut drips = [Drip { next: 0.0, drift: 0.0, f0: 0.0 }; 3];
        for (d, spot) in drips.iter_mut().zip(DRIP_SPOTS.iter()) {
            d.f0 = rng_drip.uniform(spot.4, spot.5);
            d.next = f64::INFINITY;
        }
        let phi = rng_flow.uniform(0.0, TAU);
        let spin = if rng_flow.unit() < 0.5 { -1.0 } else { 1.0 };
        let ring = || vec![0f32; RING];
        let mf = Muffle::new(s.indoor);
        let hf = Muffle::new(0.5 * s.indoor);
        let mut r = Rain {
            s: s.clone(),
            off: false,
            rng: Rng::new(seed ^ 0x0052_4149_4E42_4544),
            rng_ev,
            rng_drip,
            rng_flow,
            pos: 0,
            next_top: [0.0; 3],
            next_floor: [0.0; 3],
            next_land: 0.0,
            drips,
            bed: [[ring(), ring()], [ring(), ring()], [ring(), ring()]],
            out: [ring(), ring()],
            half: [ring(), ring()],
            lp: LP_FC.map(|fc| [Biquad::lowpass(fc, BUTTER_Q); 2]),
            mf: [mf; 2],
            hf: [hf; 2],
            m_cur: s.indoor,
            hp: [Biquad::highpass(60.0, BUTTER_Q); 2],
            pb: None,
            pb_src: None,
            gain: 0.0,
            gain_t: 0.0,
            off_from: 0.0,
            lfo: 0.0,
            lfo_sigma: LFO_SIGMA,
            rho: 1.0,
            phi,
            spin,
            flow_ou: 0.0,
            acc: [1.0; N_AZ],
            grain: vec![0.0; GRAIN_MAX],
            bin: [vec![0.0; GRAIN_MAX + H_LEN], vec![0.0; GRAIN_MAX + H_LEN]],
            thunder: Thunder::new(seed),
            th_mf: [mf; 2],
            th_hp: [Biquad::highpass(30.0, BUTTER_Q); 2],
            n_land: 0,
            n_drips: 0,
        };
        r.thunder.set_amount(0.0, s.thunder, 0);
        r.set_playback(playback);
        r.redraw_all();
        r
    }

    /// 换一份设置（音频线程里调用，不分配内存）：事件的多少变了就从现在起按新的重新排；增益、隔窗慢慢跟过去。
    pub fn set_setup(&mut self, s: Arc<RainSetup>) {
        let old = std::mem::replace(&mut self.s, s);
        self.off = false;
        let changed = |a: f32, b: f32| (a - b).abs() > 0.05 * a.max(b);
        let now = self.pos as f64;
        for c in 0..3 {
            if changed(old.top_rate[c], self.s.top_rate[c]) {
                self.next_top[c] = now + self.gap(self.s.top_rate[c]);
            }
            if changed(old.floor_rate[c], self.s.floor_rate[c]) {
                self.next_floor[c] = now + self.gap(self.s.floor_rate[c]);
            }
        }
        if changed(old.land_rate, self.s.land_rate) {
            self.next_land = now + self.gap_ev(self.s.land_rate);
        }
        if old.drip_spots() != self.s.drip_spots() {
            self.redraw_drips();
        }
        if old.thunder != self.s.thunder {
            self.thunder.set_amount(old.thunder, self.s.thunder, self.pos);
        }
    }

    /// 统计：打了几次雷。
    pub fn thunder_strikes(&self) -> u64 {
        self.thunder.n_strikes
    }

    /// 回放补偿 FIR（与人声相同）。
    pub fn set_playback(&mut self, pb: Option<&Arc<Vec<f32>>>) {
        let same = match (&self.pb_src, pb) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        if !same {
            self.pb = pb.map(|h| [Fir::new(h), Fir::new(h)]);
            self.pb_src = pb.cloned();
        }
    }

    /// 关掉：约 0.12 秒的时间常数淡出，淡完（比关掉时低 80 dB）之后 [`Rain::silent`] 为真，调用方可以丢掉它。
    pub fn set_off(&mut self) {
        if !self.off {
            self.off = true;
            self.off_from = self.gain.max(1e-12);
        }
    }

    pub fn silent(&self) -> bool {
        self.off && self.gain <= self.off_from * 1e-4
    }

    /// 测试用：去掉雨量的慢起伏（标定和事件多少按期望值核对）。
    #[cfg(test)]
    pub fn steady(mut self) -> Self {
        self.lfo_sigma = 0.0;
        self
    }

    /// 文件转换用：增益直接取目标值（不从 0 淡入；开头和结尾的淡入淡出由调用方做）。
    pub fn prime_gain(&mut self, ref_lufs: f32) {
        self.gain = self.target_gain(ref_lufs);
        self.gain_t = self.gain;
    }

    fn target_gain(&self, ref_lufs: f32) -> f32 {
        if self.off {
            return 0.0;
        }
        let l = ref_lufs + self.s.level_db + self.s.trim_db;
        (10f32.powf((l + 0.691) / 10.0) / self.s.p_unit.max(1e-30)).sqrt()
    }

    fn redraw_all(&mut self) {
        let now = self.pos as f64;
        for c in 0..3 {
            self.next_top[c] = now + self.gap(self.s.top_rate[c]);
            self.next_floor[c] = now + self.gap(self.s.floor_rate[c]);
        }
        self.next_land = now + self.gap_ev(self.s.land_rate);
        self.redraw_drips();
    }

    fn redraw_drips(&mut self) {
        let (n, period) = self.s.drip_spots();
        for k in 0..3 {
            self.drips[k].next = if k < n { self.pos as f64 + self.rng_drip.unit() as f64 * (period * DRIP_SPOTS[k].3 * FS) as f64 } else { f64::INFINITY };
        }
    }

    /// 泊松过程的下一个间隔（样本）：每秒 `rate` 个（再乘上雨量的慢起伏）。
    #[inline]
    fn gap(&mut self, rate: f32) -> f64 {
        let r = rate * self.rho;
        if r <= 1e-6 {
            return f64::INFINITY;
        }
        -((1.0 - self.rng.unit()).max(1e-12) as f64).ln() * FS as f64 / r as f64
    }

    #[inline]
    fn gap_ev(&mut self, rate: f32) -> f64 {
        let r = rate * self.rho;
        if r <= 1e-6 {
            return f64::INFINITY;
        }
        -((1.0 - self.rng_ev.unit()).max(1e-12) as f64).ln() * FS as f64 / r as f64
    }

    /// 生成 `out_l.len()` 个样本的雨声（覆盖写）。`ref_lufs`：人声的目标响度（雨声比它低"音量"那么多 dB）。
    /// 结果与每次给多长无关（内部按绝对时间每 64 个样本安排一次）。
    pub fn render(&mut self, ref_lufs: f32, out_l: &mut [f32], out_r: &mut [f32]) {
        let n = out_l.len().min(out_r.len());
        let kg = 1.0 - (-1.0 / (if self.off { 0.12 } else { 0.25 } * FS)).exp();
        for i in 0..n {
            if self.pos % CTRL as u64 == 0 {
                self.control(ref_lufs);
            }
            let idx = (self.pos as usize) & (RING - 1);
            // 雷声（有雷或回声还没散时才走这条路；没打雷时输出和没有雷声这个功能时逐样本相同）
            let live = self.thunder.live(self.pos);
            let th = if live { self.thunder.sample(idx) } else { [0.0; 2] };
            let mut y = [0f32; 2];
            for (e, ye) in y.iter_mut().enumerate() {
                let mut b = 0.0;
                for c in 0..3 {
                    let v = std::mem::take(&mut self.bed[c][e][idx]);
                    b += self.lp[c][e].process(v);
                }
                b += std::mem::take(&mut self.out[e][idx]);
                let mb = self.mf[e].process(b);
                let h = std::mem::take(&mut self.half[e][idx]);
                let mh = self.hf[e].process(h);
                let mut v = self.hp[e].process(mb + mh);
                if live {
                    v += self.th_hp[e].process(self.th_mf[e].process(th[e]));
                }
                if let Some(pb) = &mut self.pb {
                    v = pb[e].process(v);
                }
                *ye = v;
            }
            self.gain += kg * (self.gain_t - self.gain);
            out_l[i] = self.gain * y[0];
            out_r[i] = self.gain * y[1];
            self.pos += 1;
        }
    }

    /// 每 64 个样本：雨量的慢起伏、增益和隔窗的目标，安排接下来 64 个样本里的所有雨滴和事件。
    fn control(&mut self, ref_lufs: f32) {
        let a = (-(CTRL as f32 / FS) / LFO_TAU).exp();
        self.lfo = a * self.lfo + (1.0 - a * a).sqrt() * self.rng.normal();
        let sg = self.lfo_sigma;
        self.rho = (sg * self.lfo - 0.5 * sg * sg).exp();
        self.gain_t = self.target_gain(ref_lufs);
        // 隔窗慢慢跟过去（0.15 s），每块重算一次系数
        let target = self.s.indoor;
        if (target - self.m_cur).abs() > 1e-4 {
            let k = 1.0 - (-(CTRL as f32 / FS) / 0.15).exp();
            self.m_cur += k * (target - self.m_cur);
            if (target - self.m_cur).abs() < 1e-3 {
                self.m_cur = target;
            }
            let (mf, hf) = (Muffle::new(self.m_cur), Muffle::new(0.5 * self.m_cur));
            for e in 0..2 {
                self.mf[e].retune(&mf);
                self.hf[e].retune(&hf);
                self.th_mf[e].retune(&mf);
            }
        }
        let t1 = (self.pos + CTRL as u64) as f64;
        let s = self.s.clone();
        // 流动：重心平均 FLOW_PERIOD 秒转一圈，转速随机慢慢变；这一块里各方位按 e^{κ(cos Δ − 1)} 接受
        if s.flow > 0.0 {
            let dt = CTRL as f32 / FS;
            let a = (-dt / FLOW_TAU).exp();
            self.flow_ou = a * self.flow_ou + (1.0 - a * a).sqrt() * self.rng_flow.normal();
            self.phi = (self.phi + self.spin * TAU / FLOW_PERIOD * (FLOW_SPEED_SIGMA * self.flow_ou).exp() * dt).rem_euclid(TAU);
            for (i, v) in self.acc.iter_mut().enumerate() {
                *v = (s.flow * (((i as f32 * AZ_STEP).to_radians() - self.phi).cos() - 1.0)).exp();
            }
        }
        for c in 0..3 {
            while self.next_top[c] < t1 {
                let t = self.next_top[c].max(self.pos as f64) as u64;
                self.impulse(&s, c, false, t);
                self.next_top[c] += self.gap(s.top_rate[c]);
            }
            while self.next_floor[c] < t1 {
                let t = self.next_floor[c].max(self.pos as f64) as u64;
                self.impulse(&s, c, true, t);
                self.next_floor[c] += self.gap(s.floor_rate[c]);
            }
        }
        while self.next_land < t1 {
            let t = self.next_land.max(self.pos as f64) as u64;
            self.land(&s, t);
            self.next_land += self.gap_ev(s.land_rate);
        }
        self.thunder.control(s.thunder, &s.az_cdf, s.p_unit, self.pos, t1);
        let (n_spots, period) = s.drip_spots();
        for k in 0..n_spots {
            while self.drips[k].next < t1 {
                let t = self.drips[k].next.max(self.pos as f64) as u64;
                self.drip(&s, k, t);
                // 节奏：各处自己的间隔 × 慢慢漂移（AR(1)，稳态 ±20%）× 每一滴 ±10% 的随机
                let (n1, n2) = (self.rng_drip.normal(), self.rng_drip.normal());
                let d = &mut self.drips[k];
                d.drift = 0.85 * d.drift + 0.527 * 0.2 * n1;
                let jitter = (1.0 + 0.1 * n2).clamp(0.7, 1.3);
                d.next += (period * DRIP_SPOTS[k].3 * d.drift.exp() * jitter * FS) as f64;
            }
        }
    }

    /// 整片雨声里的一滴（或一个"底子"脉冲）：大小、落点、方向、强度，脉冲卷 HRIR 加进它那个圆环的总线。
    fn impulse(&mut self, s: &RainSetup, c: usize, floor: bool, t: u64) {
        let d = sample_cdf(if floor { &s.floor_cdf[c] } else { &s.top_cdf[c] }, self.rng.unit());
        let (ri, ro) = RINGS[c];
        let r = (ri * ri + self.rng.unit() * (ro * ro - ri * ri)).sqrt();
        let el = if self.rng.unit() < s.win_p { EL_WINDOW } else { el_of_r(r) };
        let az = draw_az(&mut self.rng, &s.az_cdf, &self.acc, s.flow > 0.0);
        let w = if floor {
            if self.rng.unit() < 0.5 {
                -s.floor_w[c]
            } else {
                s.floor_w[c]
            }
        } else {
            w_raw(d) * self.rng.lognormal(1.0, ZETA_SIGMA) / (r * r + H_EAR * H_EAR).sqrt() * RING_W[c].sqrt()
        };
        let soft = self.rng.unit() < P_SOFT;
        let (p, w) = if soft { (bucket(d) + N_HARD, w * SOFT_GAIN) } else { (bucket(d), w) };
        let dir = el * N_AZ + az;
        let bank = bank();
        for e in 0..2 {
            ring_add(&mut self.bed[c][e], t, bank.pulse(p, dir, e), w);
        }
    }

    /// 身边地上的一个落地声：落在 0.7–3 m 处（按面积均匀），从那个方向、那个仰角来（隔窗时一部分从窗户来，
    /// 和外面的雨一起闷掉）；远近、轻重各有不同，平均响度按设置。
    fn land(&mut self, s: &RainSetup, t: u64) {
        self.n_land += 1;
        let n = self.thump_grain();
        let (ri, ro) = LAND_R;
        let r = (ri * ri + self.rng_ev.unit() * (ro * ro - ri * ri)).sqrt();
        let el = if self.rng_ev.unit() < s.win_p { EL_WINDOW } else { el_of_r(r) };
        let az = draw_az(&mut self.rng_ev, &s.az_cdf, &self.acc, s.flow > 0.0);
        let len = self.binauralize(n, el * N_AZ + az);
        // 能量 ∝ 1/(r² + 1.2²)，按 0.7–3 m 上的平均归一
        let h2 = H_EAR * H_EAR;
        let mean = ((ro * ro + h2) / (ri * ri + h2)).ln() / (ro * ro - ri * ri);
        let target = s.land_energy / (r * r + h2) / mean * self.rng_ev.lognormal(1.0, 0.3);
        let g = (target / self.bin_energy(len).max(1e-20)).sqrt();
        for e in 0..2 {
            ring_add(&mut self.out[e], t, &self.bin[e][..len], g);
        }
    }

    /// 屋檐的一滴：落在石阶 / 积水的地上，"嗒"。
    fn drip(&mut self, s: &RainSetup, k: usize, t: u64) {
        self.n_drips += 1;
        let spot = DRIP_SPOTS[k];
        let f0 = self.drips[k].f0 * (1.0 + 0.05 * self.rng_drip.normal());
        let n = self.drip_grain(f0);
        let az = az_index(s.center + spot.0);
        let len = self.binauralize(n, spot.1 * N_AZ + az);
        let target = s.drip_energy * (1.4 * 1.4) / (spot.2 * spot.2) * self.rng_drip.lognormal(1.0, 0.25);
        let g = (target / self.bin_energy(len).max(1e-20)).sqrt();
        for e in 0..2 {
            ring_add(&mut self.half[e], t, &self.bin[e][..len], g);
        }
    }

    /// 落地的"噗"：雨点落在泥地、草、积水上——一团 150–500 Hz 的闷响（带通噪声，中心 170–300 Hz，150 Hz 以下
    /// 四阶切掉），4 ms 起音、时间常数 35–60 ms 衰减，开头带一点点溅水的"沙"（1.5–5 kHz、3 ms，很轻）。
    /// 按参考视频里落地声的频谱和长短做：250 Hz 倍频程最突出，1 kHz 以上几乎没有，约 80 ms 后听不出来。
    /// 是噪声不是正弦，没有音调。
    fn thump_grain(&mut self) -> usize {
        let len = ((0.16 * FS) as usize).min(GRAIN_MAX);
        let fc = (170f32.ln() + self.rng_ev.unit() * (300f32 / 170.0).ln()).exp();
        let tau = self.rng_ev.uniform(0.035, 0.06) * FS;
        let splash = self.rng_ev.uniform(0.03, 0.08);
        let mut bp = Biquad::bandpass(fc, 1.0);
        let mut hp = [Biquad::highpass(160.0, BUTTER_Q); 2];
        let mut lp = Biquad::lowpass(700.0, BUTTER_Q);
        let (mut hs, mut ls) = (Biquad::highpass(1500.0, BUTTER_Q), Biquad::lowpass(5000.0, BUTTER_Q));
        let (att, ts, fade) = (0.004 * FS, 0.003 * FS, 0.01 * FS);
        for (n, o) in self.grain[..len].iter_mut().enumerate() {
            let x = n as f32;
            let mut env = if x < att { 0.5 - 0.5 * (PI * x / att).cos() } else { (-(x - att) / tau).exp() };
            let left = (len - n) as f32;
            if left < fade {
                env *= 0.5 - 0.5 * (PI * left / fade).cos();
            }
            let b0 = hp[0].process(lp.process(bp.process(env * (2.0 * self.rng_ev.unit() - 1.0))));
            let body = hp[1].process(b0);
            let sp = ls.process(hs.process(splash * (-x / ts).exp() * (2.0 * self.rng_ev.unit() - 1.0)));
            *o = body + sp;
        }
        len
    }

    /// 滴水的"嗒"：一下撞击（0.3–0.5 ms 的半个正弦）+ 一小团溅开的水（中心 f0 的带通噪声，4–8 ms）
    /// + 一点点"噗"（250–400 Hz，12 ms）。也是噪声，没有音调。
    fn drip_grain(&mut self, f0: f32) -> usize {
        let len = ((0.06 * FS) as usize).min(GRAIN_MAX);
        self.grain[..len].fill(0.0);
        tap(&mut self.grain, self.rng_drip.uniform(0.3, 0.5), 0.5);
        let mut bp = Biquad::bandpass(f0, 1.2);
        let mut lo = Biquad::bandpass(self.rng_drip.uniform(250.0, 400.0), 0.9);
        let (ts, tl, att) = (self.rng_drip.uniform(0.004, 0.008) * FS, 0.012 * FS, 0.0005 * FS);
        for (n, o) in self.grain[..len].iter_mut().enumerate() {
            let x = n as f32;
            let w = (x / att).min(1.0) * (2.0 * self.rng_drip.unit() - 1.0);
            *o += bp.process(w * (-x / ts).exp()) + 0.5 * lo.process(w * (-x / tl).exp());
        }
        len
    }

    /// 单声道的事件卷上一个方向的 HRIR，写进 `bin`，返回长度。
    fn binauralize(&mut self, n: usize, dir: usize) -> usize {
        let bank = bank();
        let len = (n + H_LEN - 1).min(GRAIN_MAX + H_LEN);
        for e in 0..2 {
            let h = bank.hrir(dir, e);
            let out = &mut self.bin[e][..len];
            out.fill(0.0);
            for (k, &hv) in h.iter().enumerate() {
                if hv == 0.0 {
                    continue;
                }
                for (o, &g) in out[k..k + n].iter_mut().zip(&self.grain[..n]) {
                    *o += hv * g;
                }
            }
        }
        len
    }

    fn bin_energy(&self, len: usize) -> f32 {
        self.bin.iter().map(|b| b[..len].iter().map(|v| v * v).sum::<f32>()).sum()
    }
}

/// 一下很短的"嗒"：宽 `ms` 毫秒的半个正弦，峰值 `amp`，加到 out 开头。返回长度。
fn tap(out: &mut [f32], ms: f32, amp: f32) -> usize {
    let n = ((ms * 1e-3 * FS).round() as usize).max(2);
    for (i, v) in out[..n].iter_mut().enumerate() {
        *v += amp * (PI * (i as f32 + 0.5) / n as f32).sin();
    }
    n
}

/// 抽一个方位：按 `cdf`；流动时再按 `acc`（各方位的接受概率）拒绝重抽，最多 16 次。
#[inline]
fn draw_az(rng: &mut Rng, cdf: &[f32; N_AZ], acc: &[f32; N_AZ], flow: bool) -> usize {
    let mut az = sample_az(cdf, rng.unit());
    if flow {
        for _ in 0..16 {
            if rng.unit() < acc[az] {
                break;
            }
            az = sample_az(cdf, rng.unit());
        }
    }
    az
}

/// 把 `src × g` 加进环形缓冲，从绝对时间 t 开始。
#[inline]
pub(super) fn ring_add(ring: &mut [f32], t: u64, src: &[f32], g: f32) {
    let start = (t as usize) & (RING - 1);
    let first = (RING - start).min(src.len());
    for (d, s) in ring[start..start + first].iter_mut().zip(&src[..first]) {
        *d += g * s;
    }
    for (d, s) in ring[..src.len() - first].iter_mut().zip(&src[first..]) {
        *d += g * s;
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// 只生成雨声：`secs` 秒，每次给 `chunk` 个样本；人声目标响度 −16 LUFS。没有雨量的慢起伏（按期望值核对）。
    pub fn render(p: &Params, secs: f32, chunk: usize) -> (Vec<f32>, Vec<f32>) {
        render_with(p, secs, chunk, true)
    }

    pub fn render_with(p: &Params, secs: f32, chunk: usize, steady: bool) -> (Vec<f32>, Vec<f32>) {
        let s = Arc::new(RainSetup::new(p, Some(&tables().cdfc)).expect("音量为 0"));
        let pb = Arc::new(tables().cdfc.clone());
        let mut rain = Rain::new(s, p.seed, Some(&pb));
        if steady {
            rain = rain.steady();
        }
        rain.prime_gain(-16.0);
        let n = (secs * FS) as usize;
        let (mut l, mut r) = (vec![0f32; n], vec![0f32; n]);
        let mut i = 0;
        while i < n {
            let m = chunk.min(n - i);
            rain.render(-16.0, &mut l[i..i + m], &mut r[i..i + m]);
            i += m;
        }
        (l, r)
    }

    pub use crate::dsp::testsig::lufs;

    fn energy(x: &[f32]) -> f64 {
        x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() + 1e-30
    }

    fn bed(rain: f32, indoor: f32, pan: f32, width: f32) -> Params {
        Params { bg_level: 60.0, rain, bg_indoor: indoor, bg_pan: pan, bg_width: width, bg_flow: 0.0, rain_near: 0.0, rain_drip: 0.0, ..Params::default() }
    }

    #[test]
    fn loudness_follows_the_volume_setting() {
        // 只有整片雨声时：积分响度 = −16 + 音量（−16 dB）+ 雨量的偏移，各种设置下误差在 0.8 dB 以内
        // （大雨滴是重尾分布，40 秒的实测会在期望值上下零点几 dB；这里去掉了雨量的慢起伏）
        for (rain, indoor, pan, width) in [(10.0, 0.0, 0.0, 100.0), (35.0, 60.0, 0.0, 100.0), (60.0, 0.0, -80.0, 40.0), (95.0, 100.0, 50.0, 100.0)] {
            let p = bed(rain, indoor, pan, width);
            let (l, r) = render(&p, 40.0, 4096);
            let want = -16.0 + p.bg_db().unwrap() + 10.0 * rain / 100.0 - 5.0;
            let got = lufs(&l, &r);
            assert!((got - want).abs() < 0.8, "雨量 {rain} 隔窗 {indoor} 左右 {pan} 包围 {width}：{got:.2} LUFS，应为 {want:.2}");
        }
    }

    #[test]
    fn output_does_not_depend_on_block_size() {
        // 带着雷（打开后 3–10 秒内来第一声，所以渲染 14 秒）
        let p = Params { bg_level: 60.0, rain_near: 80.0, rain_drip: 90.0, bg_thunder: 100.0, ..Params::default() };
        let a = render_with(&p, 14.0, 64, false);
        let b = render_with(&p, 14.0, 4096, false);
        let c = render_with(&p, 14.0, 1000, false);
        assert!(a == b && b == c);
        assert!(a.0.iter().chain(&a.1).all(|v| v.is_finite()));
    }

    #[test]
    fn pan_width_and_window_shape_the_rain() {
        let band = |x: &[f32], lo: f32| -> f64 {
            // lo 以上的能量（一阶差分近似高通够用：只比较相对大小）
            let mut hp = Biquad::highpass(lo, BUTTER_Q);
            x.iter().map(|&v| (hp.process(v) as f64).powi(2)).sum::<f64>()
        };
        let corr = |l: &[f32], r: &[f32]| -> f64 {
            let (mut a, mut b) = (Biquad::highpass(500.0, BUTTER_Q), Biquad::highpass(500.0, BUTTER_Q));
            let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
            for (&x, &y) in l.iter().zip(r) {
                let (u, v) = (a.process(x) as f64, b.process(y) as f64);
                ab += u * v;
                aa += u * u;
                bb += v * v;
            }
            ab / (aa * bb).sqrt()
        };
        let db = |a: f64, b: f64| 10.0 * (a / b).log10();
        // 四周均匀时左右一样响、两耳几乎不相关（包围着人）；全在左边时左耳响得多
        let (l, r) = render(&bed(35.0, 0.0, 0.0, 100.0), 10.0, 4096);
        assert!(db(energy(&l), energy(&r)).abs() < 1.0);
        let around = corr(&l, &r);
        assert!(around.abs() < 0.2, "四周均匀时两耳相关 {around:.2}");
        let (ll, rl) = render(&bed(35.0, 0.0, -100.0, 100.0), 10.0, 4096);
        assert!(db(energy(&ll), energy(&rl)) > 6.0, "全在左边：左比右 {:.1} dB", db(energy(&ll), energy(&rl)));
        // 只从正前方来（包围感 0）：两耳几乎一样，相关很高
        let (lf, rf) = render(&bed(35.0, 0.0, 0.0, 0.0), 10.0, 4096);
        let front = corr(&lf, &rf);
        assert!(front > 0.6, "只从正前方来时两耳相关 {front:.2}");
        // 隔窗：4 kHz 以上的能量（相对总能量）至少少 10 dB（真实录音：屋里听雨约 −18 dB，室外平均约 −8 dB）
        let (lo, _) = render(&bed(35.0, 0.0, 0.0, 100.0), 10.0, 4096);
        let (li, _) = render(&bed(35.0, 100.0, 0.0, 100.0), 10.0, 4096);
        let hf = |x: &[f32]| db(band(x, 4000.0), energy(x));
        assert!(hf(&lo) - hf(&li) > 10.0, "隔窗前后 4 kHz 以上占比：{:.1} → {:.1} dB", hf(&lo), hf(&li));
    }

    #[test]
    fn light_rain_is_sparse_and_events_follow_their_sliders() {
        // 小雨的 2–4 kHz 包络是稀疏的（一滴一滴），峰度明显大于高斯噪声的约 3.2
        let (l, _) = render(&bed(10.0, 0.0, 0.0, 100.0), 20.0, 4096);
        let mut bp = Biquad::bandpass(2800.0, 1.4);
        let y: Vec<f64> = l.iter().map(|&v| bp.process(v) as f64).collect();
        let env: Vec<f64> = y.chunks(120).map(|c| (c.iter().map(|v| v * v).sum::<f64>() / c.len() as f64).sqrt()).collect();
        let n = env.len() as f64;
        let m = env.iter().sum::<f64>() / n;
        let var = env.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
        let kurt = env.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n / (var * var);
        assert!(kurt > 5.0, "小雨包络峰度 {kurt:.1}");
        // 落地声随雨量：雨量 35 时每秒约 2.4 个；屋檐滴水 100 → 3 处、每秒约 4.4 滴；落地声为 0 时两样都没有
        let p = Params { bg_level: 60.0, rain_near: 50.0, rain_drip: 100.0, ..Params::default() };
        let s = Arc::new(RainSetup::new(&p, None).unwrap());
        let mut rain = Rain::new(s, 1, None).steady();
        let (mut a, mut b) = (vec![0f32; 4096], vec![0f32; 4096]);
        for _ in 0..(60 * 48_000 / 4096) {
            rain.render(-16.0, &mut a, &mut b);
        }
        let secs = (60 * 48_000 / 4096 * 4096) as f32 / FS;
        let land = rain.n_land as f32 / secs;
        let drips = rain.n_drips as f32 / secs;
        assert!((land - 2.4).abs() < 0.5, "落地声每秒 {land:.2}");
        assert!((drips - 4.4).abs() < 0.8, "滴水每秒 {drips:.2}");
        let off = RainSetup::new(&Params { rain_near: 0.0, ..p.clone() }, None).unwrap();
        assert!(off.land_rate == 0.0 && off.drip_spots().0 == 0);
    }

    /// 每 10 ms 一帧、两耳取大的，比 1 秒滑动中位数高多少 dB（只看 `lo`–`hi` Hz）。
    fn excess(l: &[f32], r: &[f32], lo: f32, hi: f32) -> Vec<f32> {
        let frame = (0.01 * FS) as usize;
        let per_ear = |x: &[f32]| -> Vec<f32> {
            let (mut h0, mut h1) = (Biquad::highpass(lo, BUTTER_Q), Biquad::highpass(lo, BUTTER_Q));
            let (mut g0, mut g1) = (Biquad::lowpass(hi, BUTTER_Q), Biquad::lowpass(hi, BUTTER_Q));
            x.chunks_exact(frame)
                .map(|c| {
                    c.iter()
                        .map(|&v| {
                            let y = g1.process(g0.process(h1.process(h0.process(v))));
                            y * y
                        })
                        .sum::<f32>()
                })
                .collect()
        };
        let m: Vec<f32> = per_ear(l).iter().zip(per_ear(r)).map(|(a, b)| a.max(b) + 1e-30).collect();
        (0..m.len())
            .map(|i| {
                let mut w = m[i.saturating_sub(50)..(i + 51).min(m.len())].to_vec();
                w.sort_by(|a, b| a.total_cmp(b));
                10.0 * (m[i] / w[w.len() / 2]).log10()
            })
            .collect()
    }

    fn pct(v: &[f32], q: f32) -> f32 {
        let mut w = v.to_vec();
        w.sort_by(|a, b| a.total_cmp(b));
        w[((w.len() - 1) as f32 * q) as usize]
    }

    #[test]
    fn nothing_sticks_out_and_landing_drops_follow_their_slider() {
        // 第一版的雨夜：窗玻璃的"咚"和滴水的水泡一下比雨声高 15 dB 以上、每秒两三下，最高 26 dB，太吵。
        // 参考的 8D 雨声（见 doc/effects.md）：全频 99.9% 分位 6–8.5 dB；150–500 Hz（落地声）99.9% 分位 11–12 dB。
        let night = Params::default().bg_rain_night();
        let (l, r) = render_with(&night, 40.0, 4096, false);
        let full = excess(&l[FS as usize..], &r[FS as usize..], 100.0, 14000.0);
        let (p999, max) = (pct(&full, 0.999), pct(&full, 1.0));
        assert!(p999 < 8.0 && max < 11.0, "雨夜全频：99.9% 分位 {p999:.1} dB，最大 {max:.1} dB");
        // 落地声：默认 30 比参考轻，60 和参考差不多，100 更明显；150–500 Hz 的 99.9% 分位
        let low = |land: f32| {
            let (l, r) = render_with(&Params { rain_near: land, ..night.clone() }, 40.0, 4096, false);
            pct(&excess(&l[FS as usize..], &r[FS as usize..], 150.0, 500.0), 0.999)
        };
        let (a, b, c) = (low(30.0), low(60.0), low(100.0));
        assert!(a < 10.5 && (10.5..15.0).contains(&b) && c > b + 2.0, "落地声 30 / 60 / 100：{a:.1} / {b:.1} / {c:.1} dB");
    }

    #[test]
    fn flow_moves_the_rain_around_like_the_reference() {
        // 1–4 kHz 左右差（每 0.5 s）的标准差：参考的 8D 雨声约 1.6 dB。流动 0 几乎不动，50 和参考差不多，100 更大
        let ild_sd = |flow: f32| -> f32 {
            let (l, r) = render(&Params { bg_flow: flow, ..bed(35.0, 0.0, 0.0, 100.0) }, 90.0, 4096);
            let blk = (0.5 * FS) as usize;
            let band = |x: &[f32]| -> Vec<f32> {
                let (mut h0, mut h1) = (Biquad::highpass(1000.0, BUTTER_Q), Biquad::highpass(1000.0, BUTTER_Q));
                let (mut g0, mut g1) = (Biquad::lowpass(4000.0, BUTTER_Q), Biquad::lowpass(4000.0, BUTTER_Q));
                x.chunks_exact(blk)
                    .map(|c| {
                        c.iter()
                            .map(|&v| {
                                let y = g1.process(g0.process(h1.process(h0.process(v))));
                                y * y
                            })
                            .sum::<f32>()
                    })
                    .collect()
            };
            let ild: Vec<f32> = band(&l).iter().zip(band(&r)).map(|(a, b)| 10.0 * (a / b).log10()).collect();
            let m = ild.iter().sum::<f32>() / ild.len() as f32;
            (ild.iter().map(|v| (v - m).powi(2)).sum::<f32>() / ild.len() as f32).sqrt()
        };
        let (a, b, c) = (ild_sd(0.0), ild_sd(50.0), ild_sd(100.0));
        assert!(a < 0.7 && (1.0..2.3).contains(&b) && c > 3.0, "流动 0 / 50 / 100 的左右差标准差：{a:.2} / {b:.2} / {c:.2} dB");
    }

    /// 雷声这一层（同一个种子：有雷 − 没雷；`thunder` 是滑块 0–100）和雨声本身，各 `secs` 秒；外加打了几次雷。
    fn thunder_layer(x: f32, secs: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, u64) {
        let night = Params::default().bg_rain_night();
        let run = |thunder: f32| -> (Vec<f32>, Vec<f32>, u64) {
            let p = Params { bg_thunder: thunder, ..night.clone() };
            let s = Arc::new(RainSetup::new(&p, Some(&tables().cdfc)).unwrap());
            let pb = Arc::new(tables().cdfc.clone());
            let mut rain = Rain::new(s, p.seed, Some(&pb)).steady();
            rain.prime_gain(-16.0);
            let n = (secs * FS) as usize;
            let (mut l, mut r) = (vec![0f32; n], vec![0f32; n]);
            for i in (0..n).step_by(4096) {
                let m = 4096.min(n - i);
                rain.render(-16.0, &mut l[i..i + m], &mut r[i..i + m]);
            }
            (l, r, rain.thunder_strikes())
        };
        let (l1, r1, k) = run(x);
        let (l0, r0, k0) = run(0.0);
        assert_eq!(k0, 0, "雷声 0 时不该打雷");
        let tl = l1.iter().zip(&l0).map(|(a, b)| a - b).collect();
        let tr = r1.iter().zip(&r0).map(|(a, b)| a - b).collect();
        (tl, tr, l0, r0, k)
    }

    #[test]
    fn thunder_is_optional_and_leaves_the_rain_alone() {
        // 雷用自己的随机数：打开后第一声（3–10 秒）来之前，有雷和没雷逐样本相同；之后只是多了雷
        let (tl, tr, l0, _, k) = thunder_layer(100.0, 20.0);
        assert!(k >= 1, "20 秒里一声雷都没有");
        let first = tl.iter().zip(&tr).position(|(a, b)| *a != 0.0 || *b != 0.0).unwrap();
        assert!((3.0..10.5).contains(&(first as f32 / FS)), "第一声雷在 {:.1} s", first as f32 / FS);
        assert!(energy(&tl[first..]) > 1e-3 * energy(&l0[first..]));
    }

    #[test]
    fn thunder_rolls_low_and_follows_its_slider() {
        // 多少：滑块 100 平均 25 秒一次（两次之间至少 8 秒），300 秒里约 12 次
        let (tl, tr, l0, r0, k) = thunder_layer(100.0, 300.0);
        assert!((6..=18).contains(&k), "300 秒打了 {k} 次雷");
        // 低沉：250 Hz 以下占八成以上，1 kHz 以上不到 1%
        let low = |x: &[f32], lo: f32, hi: f32| -> f64 {
            let (mut h, mut g) = (Biquad::highpass(lo, BUTTER_Q), Biquad::lowpass(hi, BUTTER_Q));
            x.iter().map(|&v| (g.process(h.process(v)) as f64).powi(2)).sum::<f64>()
        };
        let tot = energy(&tl) + energy(&tr);
        let below = low(&tl, 10.0, 250.0) + low(&tr, 10.0, 250.0);
        let above = {
            let (mut a, mut b) = (Biquad::highpass(1000.0, BUTTER_Q), Biquad::highpass(1000.0, BUTTER_Q));
            tl.iter().map(|&v| (a.process(v) as f64).powi(2)).sum::<f64>() + tr.iter().map(|&v| (b.process(v) as f64).powi(2)).sum::<f64>()
        };
        assert!(below / tot > 0.8 && above / tot < 0.01, "250 Hz 以下 {:.0}%，1 kHz 以上 {:.2}%", 100.0 * below / tot, 100.0 * above / tot);
        // 会滚：第一声雷在它自己峰值 −20 dB 以内的时间跨度至少 4 秒（0.1 s 一格）
        let blk = (0.1 * FS) as usize;
        let env: Vec<f64> = tl.chunks(blk).zip(tr.chunks(blk)).take(250).map(|(a, b)| energy(a) + energy(b)).collect();
        let pk = env.iter().cloned().fold(0.0, f64::max);
        let on: Vec<usize> = (0..env.len()).filter(|&i| env[i] > pk * 0.01).collect();
        let span = (on.last().unwrap() - on.first().unwrap()) as f32 * 0.1;
        assert!(span >= 4.0, "第一声雷只滚了 {span:.1} s");
        // 响度（K 加权、3 秒）：滑块 100 时最响的一声比雨声高 0–14 dB；滑块 30 时最响的明显更轻
        let loud = |l: &[f32], r: &[f32]| -> Vec<f64> {
            let (mut a, mut b) = (KWeight::new(), KWeight::new());
            let e: Vec<f64> = l.chunks(blk * 5).zip(r.chunks(blk * 5)).map(|(x, y)| x.iter().map(|&v| (a.process(v) as f64).powi(2)).sum::<f64>() + y.iter().map(|&v| (b.process(v) as f64).powi(2)).sum::<f64>()).collect();
            e.windows(6).map(|w| w.iter().sum::<f64>()).collect()
        };
        let rain = { let mut v = loud(&l0, &r0); v.sort_by(|a, b| a.total_cmp(b)); v[v.len() / 2] };
        let peak = |l: &[f32], r: &[f32]| 10.0 * (loud(l, r).into_iter().fold(0.0, f64::max) / rain).log10();
        let p100 = peak(&tl, &tr);
        let (tl30, tr30, _, _, _) = thunder_layer(30.0, 300.0);
        let p30 = peak(&tl30, &tr30);
        assert!((0.0..14.0).contains(&p100) && p30 < p100 - 4.0, "最响的一声比雨声：雷声 100 {p100:+.1} dB，雷声 30 {p30:+.1} dB");
    }

    /// 倍频程能量（dB，相对 125 Hz 及以上各档的总和），中心 63, 125, …, 16000 Hz；两耳加起来。
    fn octaves(l: &[f32], r: &[f32]) -> [f32; 9] {
        const N: usize = 8192;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N);
        let win: Vec<f32> = (0..N).map(|i| 0.5 - 0.5 * (TAU * i as f32 / N as f32).cos()).collect();
        let mut psd = vec![0f64; N / 2 + 1];
        for x in [l, r] {
            for seg in x.chunks_exact(N).step_by(1) {
                let mut buf: Vec<Complex32> = seg.iter().zip(&win).map(|(&v, &w)| Complex32::new(v * w, 0.0)).collect();
                fft.process(&mut buf);
                for (p, c) in psd.iter_mut().zip(&buf[..N / 2 + 1]) {
                    *p += c.norm_sqr() as f64;
                }
            }
        }
        let centers = [63.0f32, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0];
        let mut e = [0f64; 9];
        for (k, p) in psd.iter().enumerate() {
            let f = k as f32 * FS / N as f32;
            if let Some(i) = centers.iter().position(|&c| f >= c / 2f32.sqrt() && f < c * 2f32.sqrt()) {
                e[i] += p;
            }
        }
        let tot: f64 = e[1..].iter().sum();
        e.map(|v| (10.0 * (v / tot + 1e-15).log10()) as f32)
    }

    #[test]
    fn spectrum_matches_real_rain_recordings() {
        // 真实录音（Freesound 试听版，见 doc/effects.md）的倍频程，相对 125 Hz 以上的总和：
        // 室外 4 段（开窗双耳、树林双耳、树林小雨、大雨）的平均；屋里是关着窗的夜雨那一段（贴着玻璃录的那段是密密的
        // 雨打玻璃，窗玻璃上的雨点去掉之后不再和它比）
        const OUTDOOR: [f32; 9] = [-11.3, -12.5, -11.5, -8.8, -6.3, -7.0, -9.1, -14.1, -22.0];
        const INDOOR: [f32; 9] = [-12.8, -3.6, -6.2, -9.4, -9.3, -11.4, -19.2, -25.9, -27.6];
        let cases = [
            ("室外，雨量 35", Params { rain_near: 0.0, ..bed(35.0, 0.0, 0.0, 100.0) }, OUTDOOR),
            ("关着窗，落地声 30", Params { rain_near: 30.0, ..bed(35.0, 100.0, 0.0, 100.0) }, INDOOR),
        ];
        for (name, p, want) in cases {
            let (l, r) = render_with(&p, 40.0, 4096, false);
            let got = octaves(&l, &r);
            // 250 Hz–4 kHz 每档在 ±3 dB 以内（125 Hz 录音里有风和远处的低频，8 kHz 以上是 MP3 的边，不比）
            for i in 2..=6 {
                assert!((got[i] - want[i]).abs() < 3.0, "{name}：第 {i} 档 {:.1} dB，录音 {:.1} dB（全部：{got:?}）", got[i], want[i]);
            }
        }
    }
}
