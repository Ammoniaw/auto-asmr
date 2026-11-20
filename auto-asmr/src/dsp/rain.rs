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
//! 2. **水洼里的"叮"**。一部分大雨滴落进水里卷进一个气泡，气泡按 Minnaert 频率振荡（f₀·r₀ ≈ 3 m/s），
//!    衰减率 β = π·f₀·δ（δ = 辐射阻尼 0.0138 + 热阻尼 4.5×10⁻⁴·√f₀），音调随气泡上浮往上滑：
//!    f(t) = f₀(1 + ξβt)，ξ = 0.1（van den Doel 2005 的听感实验；Moss 等 2009 转述）。空气里听到的这一声是
//!    气泡带动水面振动发出的（Phillips 等 2018）。雨越大、水洼越多，这种"叮"越多。
//! 3. **近处雨滴**：窗玻璃（几个挨得很近的模态，敲出"叮"带一点拍频）、窗台（木头的"嗒"或金属的"叮"）、
//!    树叶（短促的"啪"）、水洼（气泡）上听得清的一滴一滴。"隔窗"越大，越多落在玻璃和窗台上。
//! 4. **屋檐滴水**：1–3 处固定的位置，各有各的节奏（带一点随机和慢慢的漂移）和音高，"滴答"——
//!    大水滴落进水洼，气泡的音调上滑更明显（ξ 0.25–0.55，真实的水滴常有 2–3 倍的上滑：Zheng & James 2009）。
//! 5. **隔窗**：外面的雨声按墙和玻璃的隔声（越高的频率挡得越多）低通，再漏一点进来；窗玻璃上的雨点就在
//!    玻璃上，不受影响；屋檐滴水挡一半。
//! 6. **响度**：按整片雨声的期望功率（每一种雨滴的频谱 × 方向分布下的 HRIR × 各级滤波 × 回放补偿 × K 加权）
//!    解析地算出增益，让雨声的积分响度 = 人声的目标响度 + 音量（dB）+ 雨量带来的 ±5 dB，不用先跑一遍去测。
//!    近处雨滴和滴水按整片雨声的功率定各自的响度，所以调它们不会改变"沙沙"的大小。

use super::dynamics::KWeight;
use super::filters::{Biquad, Fir, BUTTER_Q};
use super::params::Params;
use super::rng::Rng;
use super::tables::{tables, Hrir};
use super::{CTRL, FS, NTAP};
use rustfft::{num_complex::Complex32, FftPlanner};
use std::f32::consts::{PI, TAU};
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------- 方向网格和脉冲

/// 方位每 5° 一个方向（SOFA 方位角：逆时针，0 = 正前，90 = 左）。
const N_AZ: usize = 72;
const AZ_STEP: f32 = 5.0;
/// 仰角 5 档：地面上的雨滴在耳朵下方（近处更靠下），窗户大约和耳朵齐平，树叶在上面。
const ELS: [f32; 5] = [-55.0, -35.0, -18.0, -5.0, 15.0];
const N_EL: usize = ELS.len();
const N_DIR: usize = N_AZ * N_EL;
/// 窗户方向（隔窗时外面的雨从窗户进来）、高处（树叶）的仰角档。
const EL_WINDOW: usize = 3;
const EL_HIGH: usize = 4;
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
const H_LEN: usize = 192;
const B_LEN: usize = H_LEN + PULSE_LEN;
/// 定响度用的频率网格。
const NFFT: usize = 512;
const NB: usize = NFFT / 2 + 1;
/// 各条总线的环形缓冲（样本）；一个事件最长 GRAIN_MAX + H_LEN，提前不到一个控制块安排，放得下。
const RING: usize = 4096;
const GRAIN_MAX: usize = 2048;

// ---------------------------------------------------------------- 雨滴物理

/// Marshall–Palmer 谱的 N₀（/m³/mm）。
const N0: f64 = 8000.0;
/// 小于这个直径（mm）的雨滴几乎没有声音；大于 6 mm 的在空中就碎了。
const D_LO: f32 = 0.4;
const D_MAX: f32 = 6.0;
/// 耳朵离地多高（米）。
const H_EAR: f32 = 1.2;
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
/// 水洼"叮"每秒最多多少个；每个"叮"的能量 = 整片雨声 1.5 ms 的能量。能量集中在一个窄频带和几毫秒里，
/// 在它那个频带里比"沙沙"高约 3 dB；再大就一个个跳出来，雨声变成"咕嘟咕嘟"（2–4 kHz 包络峰度从 5 涨到 20 以上）。
const PLINK_MAX: f32 = 25.0;
const PLINK_SHARE: f32 = 0.0015;
/// 近处雨滴最多每秒几滴；近处雨滴 / 滴水比整片雨声在同样时长里响多少（dB）和按多长算（秒）。
const NEAR_MAX: f32 = 12.0;
const NEAR_DB: f32 = 5.0;
/// 隔窗 100 时，窗玻璃、窗台上的雨点再响多少（dB）：它们不隔着墙，外面的雨被闷掉了，它们相对就更突出。
/// 屋里听雨打窗的录音里，响度起伏 3.7–5 dB、每秒 3–4 个"突出的一下"，这里按它定。
const IN_BOOST_DB: f32 = 8.0;
const NEAR_T: f32 = 0.03;
const DRIP_DB: f32 = 13.0;
const DRIP_T: f32 = 0.04;

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

/// 气泡的衰减率（1/s）：β = π·f₀·δ，δ = 辐射阻尼 + 热阻尼（Leighton 1994；van den Doel 2005）。
#[inline]
fn bubble_decay(f0: f32) -> f32 {
    PI * f0 * (0.0138 + 4.5e-4 * f0.sqrt())
}

/// 气泡"叮"：振幅 `amp`，频率从 f₀ 按 f₀(1 + ξβt) 往上滑，指数衰减，开头 0.4 ms 淡入。写进 `out`，返回长度。
fn bubble(out: &mut [f32], f0: f32, xi: f32, amp: f32) -> usize {
    let beta = bubble_decay(f0);
    let len = ((5.5 / beta * FS).ceil() as usize).clamp(8, out.len());
    let att = (0.0004 * FS) as usize;
    let dec = (-beta / FS).exp();
    let (mut ph, mut env) = (0f32, amp);
    for (n, o) in out[..len].iter_mut().enumerate() {
        let f = (f0 * (1.0 + xi * beta * n as f32 / FS)).min(0.45 * FS);
        ph += TAU * f / FS;
        if ph > TAU {
            ph -= TAU;
        }
        let a = if n < att { 0.5 - 0.5 * (PI * n as f32 / att as f32).cos() } else { 1.0 };
        *o = a * env * ph.sin();
        env *= dec;
    }
    len
}

/// 气泡大小随机：f₀ 在 [lo, hi] Hz 之间按对数均匀；振荡幅度 ε 按 ε^−1.5 取（Moss 等 2009）。
/// 返回 (f₀, 振幅 ∝ ε·r₀)；r₀ 按毫米算。
fn bubble_draw(rng: &mut Rng, lo: f32, hi: f32) -> (f32, f32) {
    let f0 = (lo.ln() + rng.unit() * (hi / lo).ln()).exp();
    // ε 取 3%–10%（Moss 等的 1%–10% 会偶尔冒出一个比平均响十几 dB 的"咕嘟"，在雨声里很突兀）
    let (a, b, k) = (0.03f32, 0.1f32, -0.5f32); // 1 − μ，μ = 1.5
    let eps = (a.powf(k) + rng.unit() * (b.powf(k) - a.powf(k))).powf(1.0 / k);
    (f0, eps * 3000.0 / f0)
}

/// 窗户的样子：几个模态（频率 Hz, 时间常数 s, 相对幅度）。整块玻璃的低模态（一两百到四五百赫兹的"咚"），
/// 挨得很近的一簇（Farnell 的玻璃窗模型约 1969–2007 Hz，敲出来带一点拍频），再加两个更高、衰减更快的模态。
/// 各模态的大小按两段屋里听雨打窗的录音校过（见 doc/effects.md）。每个种子一扇窗。
fn window_modes(rng: &mut Rng) -> [(f32, f32, f32); 8] {
    let f1 = rng.uniform(1800.0, 2300.0);
    let t1 = rng.uniform(0.012, 0.022);
    [
        (rng.uniform(175.0, 275.0), 0.014, 3.0),
        (rng.uniform(325.0, 475.0), 0.010, 2.4),
        (f1, t1, 0.5),
        (f1 * 0.9935, t1 * 0.9, 0.4),
        (f1 * 0.9895, t1 * 0.8, 0.3),
        (f1 * 0.981, t1 * 0.7, 0.25),
        (rng.uniform(3600.0, 4500.0), 0.008, 0.6),
        (rng.uniform(5600.0, 7000.0), 0.004, 0.5),
    ]
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
    /// `[NB]`：水洼"叮"（ε·r₀ 的分布，距离 1 m）的平均功率谱。
    bub: Vec<f32>,
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
        // 水洼"叮"的平均功率谱：按运行时同样的分布抽 128 个（固定种子），2048 点 FFT 再并成 NB 档
        let mut rng = Rng::new(0x5EED_B0B);
        let mut g = vec![0f32; GRAIN_MAX];
        let mut bub = vec![0f32; NB];
        const NBUB: usize = 128;
        for _ in 0..NBUB {
            let (f0, amp) = bubble_draw(&mut rng, 2000.0, 9000.0);
            g.fill(0.0);
            let n = bubble(&mut g, f0, 0.1, amp);
            let s = power_spectrum(&mut planner, &g[..n], GRAIN_MAX);
            for (k, v) in bub.iter_mut().enumerate() {
                *v += fold(&s, k, GRAIN_MAX / NFFT) / NBUB as f32;
            }
        }
        Bank { hrir, pulse, hpow, ppow, bub }
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
fn frac_delay(h: &[f32], delay: f32, out: &mut [f32]) {
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
fn sample_az(cdf: &[f32; N_AZ], u: f32) -> usize {
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
    plink_rate: f32,
    plink_scale: f32,
    az_cdf: [f32; N_AZ],
    center: f32,
    /// 外面的雨从窗户方向进来的比例（隔窗时）。
    win_p: f32,
    indoor: f32,
    near_rate: f32,
    /// 近处雨滴落在哪里的累积概率：玻璃、窗台、树叶、水洼。
    near_cum: [f32; 4],
    drip: f32,
    /// 整片雨声在增益 1 时的期望 K 加权功率（两耳之和，每样本）。
    p_unit: f32,
    near_energy: f32,
    drip_energy: f32,
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
        let mut el_near = [0f64; N_EL];
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
            if c == 0 {
                el_near = e_r2;
            }
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

        // 水洼里的"叮"：落在近处、直径 2.2 mm 以上的雨滴里，有水的地方（雨越大越多）一部分卷进气泡
        let f_water = 0.15 + 0.35 * x;
        let area0 = PI * (RINGS[0].1 * RINGS[0].1 - RINGS[0].0 * RINGS[0].0);
        let plink_rate = ((0.2 * f_water * area0) as f64 * N0 * tail(2.2)).min(PLINK_MAX as f64) as f32;
        let mut plink_scale = 0.0;
        let mut p_unit = p_imp;
        if plink_rate > 0.01 {
            let mut s = 0f64;
            for el in 0..N_EL {
                if el_near[el] <= 0.0 {
                    continue;
                }
                for e in 0..2 {
                    let dv = &dav[(el * 2 + e) * NB..(el * 2 + e + 1) * NB];
                    for k in 0..NB {
                        s += el_near[el] * wk(k) * (bank.bub[k] * dv[k] * common[k]) as f64;
                    }
                }
            }
            let p_pl = plink_rate as f64 * s / (FS as f64 * NFFT as f64);
            let kappa = (PLINK_SHARE * plink_rate) as f64;
            if p_pl > 0.0 {
                plink_scale = (kappa * p_imp / p_pl).sqrt() as f32;
                p_unit = p_imp * (1.0 + kappa);
            }
        }
        let p_unit = p_unit as f32;
        let near = (p.rain_near / 100.0).clamp(0.0, 1.0);
        let near_cum = {
            let w = [0.75 * m, 0.25 * m, 0.6 * (1.0 - m), 0.4 * (1.0 - m)];
            let s: f32 = w.iter().sum();
            let mut c = [0f32; 4];
            let mut a = 0.0;
            for (o, v) in c.iter_mut().zip(w) {
                a += v / s;
                *o = a;
            }
            c[3] = 1.0;
            c
        };
        Some(RainSetup {
            level_db: p.bg_db().unwrap_or(-40.0),
            trim_db: 10.0 * x - 5.0,
            top_rate,
            top_cdf,
            floor_rate,
            floor_cdf,
            floor_w,
            plink_rate,
            plink_scale,
            az_cdf,
            center,
            win_p,
            indoor: m,
            near_rate: NEAR_MAX * near,
            near_cum,
            drip: (p.rain_drip / 100.0).clamp(0.0, 1.0),
            p_unit,
            near_energy: p_unit * NEAR_T * FS * 10f32.powf(NEAR_DB / 10.0),
            drip_energy: p_unit * DRIP_T * FS * 10f32.powf(DRIP_DB / 10.0),
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

/// 隔窗的滤波：二阶低通（截止频率从 20 kHz 降到 600 Hz）+ 一点漏进来的原声。返回（低通, 漏进来的比例）。
/// 隔窗的滤波：墙和玻璃按质量定律隔声（一阶低通，每倍频程约 6 dB），双层玻璃 / 墙体在一百多赫兹的共振处透得多一点
/// （峰值），再漏一点原声进来。
#[derive(Clone, Copy)]
struct Muffle {
    lp: Biquad,
    pk: Biquad,
    leak: f32,
}

impl Muffle {
    /// 隔窗 m（0–1）：一阶低通的拐点从 20 kHz 降到 700 Hz，125 Hz 的共振最多 +14 dB，漏进来的原声从 100% 降到 2%。
    /// 这几个数按两段屋里听雨（关着窗）的录音校过：隔窗 100 时各倍频程与录音的平均相差 3 dB 以内。
    fn new(m: f32) -> Muffle {
        let m = m.clamp(0.0, 1.0);
        let fc = 20000.0 * (700.0f32 / 20000.0).powf(m);
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
        format!("Background: rain, {:.0} dB below the voice ({mmh:.1} mm/h, window {:.0}, near drops {:.0}, drips {:.0})", -db, p.bg_indoor, p.rain_near, p.rain_drip)
    } else {
        format!("背景音：雨声，比人声低 {:.0} dB（雨量约 {mmh:.1} mm/h，隔窗 {:.0}，近处雨滴 {:.0}，屋檐滴水 {:.0}）", -db, p.bg_indoor, p.rain_near, p.rain_drip)
    }
}

// ---------------------------------------------------------------- 运行时

/// 屋檐下的一处滴水：下一滴的时间、慢慢漂移的节奏、位置和音高。
#[derive(Clone, Copy)]
struct Drip {
    next: f64,
    drift: f32,
    f0: f32,
}

/// 各处滴水相对"雨声中心方向"的方位（度）、仰角档、距离（米）、节奏倍数、音高范围（Hz）。
const DRIP_SPOTS: [(f32, usize, f32, f32, f32, f32); 3] = [(50.0, EL_LOW, 1.0, 1.0, 1250.0, 1600.0), (-65.0, 2, 1.8, 1.43, 1900.0, 2500.0), (160.0, EL_LOW, 2.6, 0.71, 2800.0, 3600.0)];

/// 雨声发生器。`render` 写出加了增益的雨声（两耳），调用方加到人声上。
pub struct Rain {
    s: Arc<RainSetup>,
    off: bool,
    rng: Rng,
    rng_ev: Rng,
    pos: u64,
    next_top: [f64; 3],
    next_floor: [f64; 3],
    next_plink: f64,
    next_near: f64,
    drips: [Drip; 3],
    bed: [[Vec<f32>; 2]; 3],
    out: [Vec<f32>; 2],
    half: [Vec<f32>; 2],
    inn: [Vec<f32>; 2],
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
    window: [(f32, f32, f32); 8],
    grain: Vec<f32>,
    bin: [Vec<f32>; 2],
    /// 统计：生成了多少个近处雨滴、滴水、水洼"叮"。
    pub n_near: u64,
    pub n_drips: u64,
    pub n_plinks: u64,
}

impl Rain {
    pub fn new(s: Arc<RainSetup>, seed: u64, playback: Option<&Arc<Vec<f32>>>) -> Self {
        let mut rng_ev = Rng::new(seed ^ 0x0052_4149_4E45_5645);
        let window = window_modes(&mut rng_ev);
        let mut drips = [Drip { next: 0.0, drift: 0.0, f0: 0.0 }; 3];
        for (d, spot) in drips.iter_mut().zip(DRIP_SPOTS.iter()) {
            d.f0 = rng_ev.uniform(spot.4, spot.5);
            d.next = f64::INFINITY;
        }
        let ring = || vec![0f32; RING];
        let mf = Muffle::new(s.indoor);
        let hf = Muffle::new(0.5 * s.indoor);
        let mut r = Rain {
            s: s.clone(),
            off: false,
            rng: Rng::new(seed ^ 0x0052_4149_4E42_4544),
            rng_ev,
            pos: 0,
            next_top: [0.0; 3],
            next_floor: [0.0; 3],
            next_plink: 0.0,
            next_near: 0.0,
            drips,
            bed: [[ring(), ring()], [ring(), ring()], [ring(), ring()]],
            out: [ring(), ring()],
            half: [ring(), ring()],
            inn: [ring(), ring()],
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
            window,
            grain: vec![0.0; GRAIN_MAX],
            bin: [vec![0.0; GRAIN_MAX + H_LEN], vec![0.0; GRAIN_MAX + H_LEN]],
            n_near: 0,
            n_drips: 0,
            n_plinks: 0,
        };
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
        if changed(old.plink_rate, self.s.plink_rate) {
            self.next_plink = now + self.gap(self.s.plink_rate);
        }
        if changed(old.near_rate, self.s.near_rate) {
            self.next_near = now + self.gap_ev(self.s.near_rate);
        }
        if old.drip_spots() != self.s.drip_spots() {
            self.redraw_drips();
        }
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
        self.next_plink = now + self.gap(self.s.plink_rate);
        self.next_near = now + self.gap_ev(self.s.near_rate);
        self.redraw_drips();
    }

    fn redraw_drips(&mut self) {
        let (n, period) = self.s.drip_spots();
        for k in 0..3 {
            self.drips[k].next = if k < n { self.pos as f64 + self.rng_ev.unit() as f64 * (period * DRIP_SPOTS[k].3 * FS) as f64 } else { f64::INFINITY };
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
                let mut v = self.hp[e].process(mb + mh + std::mem::take(&mut self.inn[e][idx]));
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
            }
        }
        let t1 = (self.pos + CTRL as u64) as f64;
        let s = self.s.clone();
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
        while self.next_plink < t1 {
            let t = self.next_plink.max(self.pos as f64) as u64;
            self.plink(&s, t);
            self.next_plink += self.gap(s.plink_rate);
        }
        while self.next_near < t1 {
            let t = self.next_near.max(self.pos as f64) as u64;
            self.near(&s, t);
            self.next_near += self.gap_ev(s.near_rate);
        }
        let (n_spots, period) = s.drip_spots();
        for k in 0..n_spots {
            while self.drips[k].next < t1 {
                let t = self.drips[k].next.max(self.pos as f64) as u64;
                self.drip(&s, k, t);
                // 节奏：各处自己的间隔 × 慢慢漂移（AR(1)，稳态 ±20%）× 每一滴 ±10% 的随机
                let (n1, n2) = (self.rng_ev.normal(), self.rng_ev.normal());
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
        let u = self.rng.unit();
        let az = sample_az(&s.az_cdf, u);
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

    /// 水洼里的"叮"（属于整片雨声，按隔窗一起闷掉）。
    fn plink(&mut self, s: &RainSetup, t: u64) {
        self.n_plinks += 1;
        let (f0, amp) = bubble_draw(&mut self.rng, 2000.0, 9000.0);
        let (ri, ro) = RINGS[0];
        let r = (ri * ri + self.rng.unit() * (ro * ro - ri * ri)).sqrt();
        let el = if self.rng.unit() < s.win_p { EL_WINDOW } else { el_of_r(r) };
        let u = self.rng.unit();
        let az = sample_az(&s.az_cdf, u);
        let n = bubble(&mut self.grain, f0, 0.1, amp * s.plink_scale / (r * r + H_EAR * H_EAR).sqrt());
        let len = self.binauralize(n, el * N_AZ + az);
        for e in 0..2 {
            ring_add(&mut self.out[e], t, &self.bin[e][..len], 1.0);
        }
    }

    /// 近处听得清的一滴：窗玻璃、窗台、树叶或水洼。
    fn near(&mut self, s: &RainSetup, t: u64) {
        self.n_near += 1;
        let u = self.rng_ev.unit();
        let kind = s.near_cum.iter().position(|&c| u < c).unwrap_or(3);
        self.grain.fill(0.0);
        let n = match kind {
            0 => self.glass_grain(),
            1 => self.sill_grain(),
            2 => self.leaf_grain(),
            _ => {
                let f0 = self.rng_ev.uniform(1200.0, 4500.0);
                let xi = self.rng_ev.uniform(0.1, 0.35);
                let n = bubble(&mut self.grain, f0, xi, 1.0);
                // 落水时一点点溅水声
                for v in self.grain[..72].iter_mut() {
                    *v += 0.2 * (self.rng_ev.unit() - 0.5);
                }
                n.max(72)
            }
        };
        let el = match kind {
            0 | 1 => EL_WINDOW,
            2 => EL_HIGH,
            _ => EL_LOW,
        };
        let u = self.rng_ev.unit();
        let az = sample_az(&s.az_cdf, u);
        let dist = self.rng_ev.uniform(0.6, 2.0);
        let len = self.binauralize(n, el * N_AZ + az);
        // 距离 0.6–2 m 的 1/R² 平均是 0.833：平均响度按设置，远近、轻重再各有不同
        let target = s.near_energy / (dist * dist) / 0.833 * self.rng_ev.lognormal(1.0, 0.5);
        // 窗玻璃、窗台上的雨点不隔着墙：外面的雨被闷掉了，它们相对就更突出（隔窗 100 时 +8 dB）
        let target = if kind <= 1 { target * 10f32.powf(IN_BOOST_DB * s.indoor / 10.0) } else { target };
        let g = (target / self.bin_energy(len).max(1e-20)).sqrt();
        for e in 0..2 {
            let ring = if kind <= 1 { &mut self.inn[e] } else { &mut self.out[e] };
            ring_add(ring, t, &self.bin[e][..len], g);
        }
    }

    /// 屋檐的一滴：大水滴落进水洼，气泡的音调上滑明显；偶尔没卷进气泡，只有轻轻的"啪"。
    fn drip(&mut self, s: &RainSetup, k: usize, t: u64) {
        self.n_drips += 1;
        let spot = DRIP_SPOTS[k];
        self.grain.fill(0.0);
        let n = if self.rng_ev.unit() < 0.1 {
            tap(&mut self.grain, 0.35, 1.0)
        } else {
            let f0 = self.drips[k].f0 * (1.0 + 0.035 * self.rng_ev.normal());
            let xi = self.rng_ev.uniform(0.25, 0.55);
            let n = bubble(&mut self.grain, f0, xi, 1.0);
            tap(&mut self.grain, 0.2, 0.25);
            n
        };
        let az = az_index(s.center + spot.0);
        let len = self.binauralize(n, spot.1 * N_AZ + az);
        let target = s.drip_energy * (1.4 * 1.4) / (spot.2 * spot.2) * self.rng_ev.lognormal(1.0, 0.25);
        let g = (target / self.bin_energy(len).max(1e-20)).sqrt();
        for e in 0..2 {
            ring_add(&mut self.half[e], t, &self.bin[e][..len], g);
        }
    }

    /// 窗玻璃上的一滴：一下"嗒"再加窗户的几个模态，每次落点不同，各模态的分量也不同。
    fn glass_grain(&mut self) -> usize {
        let len = ((0.06 * FS) as usize).min(GRAIN_MAX);
        // "嗒"的宽度 0.2 ms（真实雨点打玻璃的上升时间约 0.1–0.3 ms；再窄就是刺耳的咔嗒）
        tap(&mut self.grain, 0.2, 2.0);
        let win = self.window;
        for &(f, tau, a) in win.iter() {
            let amp = a * self.rng_ev.uniform(0.2, 1.0) * 0.6;
            modal(&mut self.grain[..len], f, tau, amp);
        }
        len
    }

    /// 窗台：木头的短"嗒"。（原来还有 30% 是金属的"叮"，长长的 2.5–7.5 kHz 尾音在屋里听雨的录音里没有，去掉了。）
    fn sill_grain(&mut self) -> usize {
        let len = (0.03 * FS) as usize;
        tap(&mut self.grain, 0.2, 0.8);
        for _ in 0..3 {
            let (f, tau) = (self.rng_ev.uniform(700.0, 2600.0), self.rng_ev.uniform(0.003, 0.007));
            modal(&mut self.grain[..len], f, tau, 0.8 * self.rng_ev.uniform(0.4, 1.0));
        }
        len
    }

    /// 树叶上的一滴：一小段带通的噪声，短促地"啪"一下。
    fn leaf_grain(&mut self) -> usize {
        let len = (0.025 * FS) as usize;
        let tau = self.rng_ev.uniform(0.002, 0.006);
        let mut bp = Biquad::bandpass(self.rng_ev.uniform(700.0, 2500.0), self.rng_ev.uniform(0.7, 1.4));
        let att = (0.0003 * FS) as usize;
        for n in 0..len {
            let env = if n < att { n as f32 / att as f32 } else { (-((n - att) as f32 / FS) / tau).exp() };
            self.grain[n] = bp.process(env * (self.rng_ev.unit() - 0.5) * 2.0);
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

/// 一个衰减的模态（频率 f、时间常数 tau、幅度 a），加到 out 上。
fn modal(out: &mut [f32], f: f32, tau: f32, a: f32) {
    let (s, c) = (TAU * f / FS).sin_cos();
    let dec = (-1.0 / (tau * FS)).exp();
    // 复数旋转：z = a·e^{(−1/τ + j2πf)t}，取虚部
    let (mut re, mut im) = (a, 0.0f32);
    for v in out.iter_mut() {
        *v += im;
        let (r2, i2) = (re * c - im * s, re * s + im * c);
        re = r2 * dec;
        im = i2 * dec;
    }
}

/// 把 `src × g` 加进环形缓冲，从绝对时间 t 开始。
#[inline]
fn ring_add(ring: &mut [f32], t: u64, src: &[f32], g: f32) {
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
    use crate::dsp::dynamics::LoudnessMeter;

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

    pub fn lufs(l: &[f32], r: &[f32]) -> f32 {
        let mut m = LoudnessMeter::new();
        for (&a, &b) in l.iter().zip(r) {
            m.push(a, b);
        }
        m.integrated()
    }

    fn energy(x: &[f32]) -> f64 {
        x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() + 1e-30
    }

    fn bed(rain: f32, indoor: f32, pan: f32, width: f32) -> Params {
        Params { bg_level: 60.0, rain, bg_indoor: indoor, bg_pan: pan, bg_width: width, rain_near: 0.0, rain_drip: 0.0, ..Params::default() }
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
        let p = Params { bg_level: 60.0, rain_near: 80.0, rain_drip: 90.0, ..Params::default() };
        let a = render_with(&p, 4.0, 64, false);
        let b = render_with(&p, 4.0, 4096, false);
        let c = render_with(&p, 4.0, 1000, false);
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
        // 近处雨滴 50 → 每秒约 6 滴；屋檐滴水 100 → 3 处、每秒约 4.4 滴；雨越大水洼里的"叮"越多
        let p = Params { bg_level: 60.0, rain_near: 50.0, rain_drip: 100.0, ..Params::default() };
        let s = Arc::new(RainSetup::new(&p, None).unwrap());
        let mut rain = Rain::new(s, 1, None).steady();
        let (mut a, mut b) = (vec![0f32; 4096], vec![0f32; 4096]);
        for _ in 0..(60 * 48_000 / 4096) {
            rain.render(-16.0, &mut a, &mut b);
        }
        let secs = (60 * 48_000 / 4096 * 4096) as f32 / FS;
        let near = rain.n_near as f32 / secs;
        let drips = rain.n_drips as f32 / secs;
        assert!((near - 6.0).abs() < 0.8, "近处雨滴每秒 {near:.2}");
        assert!((drips - 4.4).abs() < 0.8, "滴水每秒 {drips:.2}");
        let plinks = |rain: f32| RainSetup::new(&Params { bg_level: 60.0, rain, ..Params::default() }, None).unwrap().plink_rate;
        assert!(plinks(10.0) < 1.0 && plinks(35.0) > plinks(10.0) && plinks(90.0) >= plinks(35.0));
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
        // 真实录音（Freesound 试听版，见 doc/effects.md）的倍频程平均，相对 125 Hz 以上的总和：
        // 室外 4 段（开窗双耳、树林双耳、树林小雨、大雨），屋里 2 段（关着窗的夜雨、雨打玻璃；125 Hz 只用前一段）
        const OUTDOOR: [f32; 9] = [-11.3, -12.5, -11.5, -8.8, -6.3, -7.0, -9.1, -14.1, -22.0];
        const INDOOR: [f32; 9] = [-31.2, -3.6, -7.2, -7.7, -7.9, -8.7, -14.3, -22.5, -28.4];
        let cases = [
            ("室外，雨量 35", Params { bg_level: 60.0, rain: 35.0, bg_indoor: 0.0, bg_width: 100.0, rain_near: 0.0, rain_drip: 0.0, ..Params::default() }, OUTDOOR),
            ("关着窗，近处雨滴 30", Params { bg_level: 60.0, rain: 35.0, bg_indoor: 100.0, bg_width: 100.0, rain_near: 30.0, rain_drip: 0.0, ..Params::default() }, INDOOR),
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
