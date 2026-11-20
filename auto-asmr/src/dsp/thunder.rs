//! 背景音里的雷声（可选，默认关；「雷声」滑块 0 = 没有）。不用录音素材，按闪电的物理过程现算，和雨声一起出
//! （实时、试听、文件转换都一样），走雨声的同一条路：同一个增益、隔窗一起闷掉、回放补偿。做法和文献见
//! doc/effects.md「雷声」。
//!
//! 1. **闪电通道**：曲折的随机游走——每段 8–25 m，段与段之间转一个 |N(0, 20°)| 的角（平均约 16°，Hill 1968），
//!    绕一个随机的、垂直于当前方向的轴转。云地闪：从云底（1.8–3 km 高）往下走到地面，再在云里水平走 2–5 km，
//!    主干上半段长出 2–5 根 0.2–1.5 km 的分叉；云内闪：在 4–6.5 km 高处水平走 4–9 km。
//! 2. **每一段的声音**：放电把通道瞬间加热，激波衰减成一个 N 形的压力波（Few 1969），宽 6–14 ms（每次闪电取一个值，
//!    频谱峰约 50–110 Hz）。一段直的通道从侧面看是同时到达的一声"啪"，斜着看就被拖开 L·|cos ψ| / c
//!    （Ribner & Roy 1982：雷的"炸"和"滚"都是通道的曲折造成的）。通道在米的尺度上也是弯的，所以每 2 m 算一颗
//!    "珠子"（Few 的 string of pearls），在拖开的时间里随机排开：低频合起来就是拖开的平均，高频不互相抵消。
//!    幅度 ∝ 段长 / 距离。
//! 3. **多次回击**：云地闪常有 2–4 次回击，间隔 40–120 ms，一次比一次弱（×0.4–0.7），只走主干。
//! 4. **传播**：每一段的声音按距离 / 343 m/s 到达——近的部分先到、远的部分后到，一次闪电滚 5–15 秒；空气吸收按距离
//!    低通（二阶，拐点 1000·(0.6 / 距离 km)^(2/3) Hz：1 km 约 710 Hz、3 km 约 340 Hz、8 km 约 180 Hz，ISO 9613-1
//!    的量级），远雷只剩闷闷的隆隆声。
//! 5. **方向**：每一段从它自己的方向来（KU100 HRIR，方位每 5°，仰角 0–75° 六档）——先从低处响起，再滚到头顶上、
//!    滚到远处。闪电落在哪个方位按雨声的「左右」「包围感」抽。
//! 6. **回声**：地形、建筑、云底的回声让雷"滚"得更连贯：一个 4 路的反馈延时网络（混响时间 3 s，每绕一圈更闷一点），
//!    两耳的回声互不相关。
//! 7. **多少、远近和响度**：滑块越大，雷越近、越频繁、越响——1 ≈ 平均 4 分钟一次、8 km 外；50 ≈ 77 秒、3.5 km；
//!    100 ≈ 25 秒、1.5 km（每次的距离再随机 ±40%，最近 0.8 km，免得出现吓人的炸雷）。每次闪电的总能量按距离
//!    （1/距离²）定，3 km 时 = 整片雨声 1 秒的能量再高 [`THUNDER_DB`] dB，再随机 ±2.5 dB；所以调雷声不会改变雨声本身。
//!    打开以后 3–10 秒内来第一声，之后两次之间至少隔 8 秒。
//!
//! 用自己的随机数：开不开雷、雷多雷少，都不改变雨声、落地声和滴水的任何一个样本（只是加上雷）。

use super::filters::{Biquad, BUTTER_Q};
use super::rain::{frac_delay, ring_add, sample_az, AZ_STEP, GRAIN_MAX, H_EAR, H_LEN, N_AZ, RING};
use super::rng::Rng;
use super::tables::{tables, Hrir};
use super::{FS, NTAP};
use std::f32::consts::TAU;
use std::sync::OnceLock;

/// 声速（m/s）。
const C_AIR: f32 = 343.0;
/// 天上的方向：仰角 0–75°，每 15° 一档；方位和雨声一样每 5° 一个。
const SKY_ELS: [f32; 6] = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0];
const N_SKY: usize = N_AZ * SKY_ELS.len();
/// 通道每段的长度（m）、段与段之间转角的标准差（度；|N(0, σ)| 的平均是 0.8σ ≈ 16°）、"珠子"的间距（m）。
const SEG: (f32, f32) = (8.0, 25.0);
const TURN_SIGMA: f32 = 20.0;
const PEARL: f32 = 2.0;
/// N 波的宽度（秒）：每次闪电在这个范围里取一个。
const T_N: (f32, f32) = (0.006, 0.014);
/// 滑块 1 / 100 时两次闪电的平均间隔（秒）和距离的中位数（km），中间按对数插；距离再乘对数正态（σ 0.35），
/// 限在 0.8–15 km。
const INTERVAL_S: (f32, f32) = (240.0, 25.0);
const DIST_KM: (f32, f32) = (8.0, 1.5);
const D_RANGE_KM: (f32, f32) = (0.8, 15.0);
/// 两次闪电之间至少隔几秒；打开以后第一声在几秒内来。
const MIN_GAP_S: f32 = 8.0;
const FIRST_S: (f32, f32) = (3.0, 10.0);
/// 云地闪的比例（其余是云内闪：只有远远的隆隆声）。
const P_CG: f32 = 0.45;
/// 3 km 外一次闪电的总能量 = 整片雨声 1 秒的能量再高多少 dB（按 1/距离² 换算到别的距离）。
pub const THUNDER_DB: f32 = 22.0;
const REF_M: f32 = 3000.0;
/// 回声：4 路延时（样本）、混响时间（秒）、每绕一圈的低通（Hz）、和直达声相加的比例。
const FDN_LEN: [usize; 4] = [2437, 3301, 4127, 5281];
const RT60: f32 = 3.0;
const DAMP_HZ: f32 = 700.0;
const WET: f32 = 0.5;
/// 一次闪电最多多少段（含回击）；每个控制块最多渲染几段（多出来的推到下一块，晚一两毫秒，听不出来），免得一块里算太多。
const MAX_PULSES: usize = 6144;
const PER_BLOCK: usize = 2;
/// 空气吸收低通的拖尾（样本）。
const TAIL: usize = 480;

// ---------------------------------------------------------------- 天上方向的 HRIR

struct Sky {
    /// `[N_SKY][2][H_LEN]`：1.5 m 处的 KU100 HRIR，两耳时延（分数延时）已经做进去。
    hrir: Vec<f32>,
}

fn sky() -> &'static Sky {
    static S: OnceLock<Sky> = OnceLock::new();
    S.get_or_init(|| {
        let tab = tables();
        let (mut h, mut tmp) = (Hrir::default(), Hrir::default());
        let mut raw = Vec::with_capacity(N_SKY);
        let (mut tmin, mut tmax) = (f32::MAX, f32::MIN);
        for &el in SKY_ELS.iter() {
            for a in 0..N_AZ {
                tab.lookup3(a as f32 * AZ_STEP, el, 1.5, &mut h, &mut tmp);
                tmin = tmin.min(h.tau[0]).min(h.tau[1]);
                tmax = tmax.max(h.tau[0]).max(h.tau[1]);
                raw.push(h.clone());
            }
        }
        assert!(NTAP as f32 + (tmax - tmin) + 18.0 <= H_LEN as f32, "两耳时延太大：{tmin} … {tmax}");
        let mut hrir = vec![0f32; N_SKY * 2 * H_LEN];
        for (d, hr) in raw.iter().enumerate() {
            for e in 0..2 {
                let b = (d * 2 + e) * H_LEN;
                frac_delay(&hr.h[e], hr.tau[e] - tmin + 8.0, &mut hrir[b..b + H_LEN]);
            }
        }
        Sky { hrir }
    })
}

/// 在界面线程里先把天上方向的 HRIR 算好（约几十毫秒），免得第一次打雷时音频线程去算。
pub fn prepare() {
    let _ = sky();
}

/// 滑块（0–1）对应的平均间隔（秒）和距离中位数（km）。
pub fn interval_s(x: f32) -> f32 {
    INTERVAL_S.0 * (INTERVAL_S.1 / INTERVAL_S.0).powf(x.clamp(0.0, 1.0))
}

pub fn distance_km(x: f32) -> f32 {
    DIST_KM.0 * (DIST_KM.1 / DIST_KM.0).powf(x.clamp(0.0, 1.0))
}

// ---------------------------------------------------------------- 小向量

type V3 = [f32; 3];

#[inline]
fn add(a: V3, b: V3, k: f32) -> V3 {
    [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]]
}

#[inline]
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn unit(a: V3) -> V3 {
    let n = dot(a, a).sqrt().max(1e-12);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// 把单位向量 u 绕一个随机的、垂直于 u 的轴转 `ang`（弧度）。
fn turn(rng: &mut Rng, u: V3, ang: f32) -> V3 {
    let r = [rng.normal(), rng.normal(), rng.normal()];
    let a = unit(add(r, u, -dot(r, u)));
    let c = [a[1] * u[2] - a[2] * u[1], a[2] * u[0] - a[0] * u[2], a[0] * u[1] - a[1] * u[0]];
    let (s, co) = ang.sin_cos();
    unit([u[0] * co + c[0] * s, u[1] * co + c[1] * s, u[2] * co + c[2] * s])
}

fn horizontal(rng: &mut Rng) -> V3 {
    let a = rng.uniform(0.0, TAU);
    [a.cos(), a.sin(), 0.0]
}

// ---------------------------------------------------------------- 回声

/// 4 路反馈延时网络（Householder 反馈矩阵，每路一阶低通），单声道进、两耳不相关的两路出。
struct Fdn {
    buf: [Vec<f32>; 4],
    i: [usize; 4],
    z: [f32; 4],
    g: [f32; 4],
    a: f32,
}

impl Fdn {
    fn new() -> Self {
        Fdn {
            buf: FDN_LEN.map(|n| vec![0.0; n]),
            i: [0; 4],
            z: [0.0; 4],
            g: FDN_LEN.map(|n| 10f32.powf(-3.0 * n as f32 / (FS * RT60))),
            a: 1.0 - (-TAU * DAMP_HZ / FS).exp(),
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> (f32, f32) {
        const SIGN: [f32; 4] = [1.0, -1.0, 1.0, -1.0];
        let mut y = [0f32; 4];
        for (k, yk) in y.iter_mut().enumerate() {
            let d = self.buf[k][self.i[k]];
            self.z[k] += self.a * (d - self.z[k]);
            *yk = self.g[k] * self.z[k];
        }
        let s = 0.5 * (y[0] + y[1] + y[2] + y[3]);
        for k in 0..4 {
            self.buf[k][self.i[k]] = y[k] - s + SIGN[k] * x;
            self.i[k] += 1;
            if self.i[k] == FDN_LEN[k] {
                self.i[k] = 0;
            }
        }
        (y[0] - y[2], y[1] - y[3])
    }
}

// ---------------------------------------------------------------- 运行时

/// 通道上的一段：到达时间（样本）、幅度、拖开多少样本、几颗珠子、方向、距离（m）、N 波宽（样本）。
#[derive(Clone, Copy, Default)]
struct Pulse {
    t: f64,
    amp: f32,
    smear: f32,
    pearls: u16,
    dir: u16,
    r: f32,
    tn: u16,
}

/// 雷声发生器（[`super::rain::Rain`] 里一个）。缓冲一次分配好，音频线程里不分配内存。
pub(super) struct Thunder {
    rng: Rng,
    /// 下一次闪电的时间（样本）；不打雷时是无穷大。
    next: f64,
    pulses: Vec<Pulse>,
    n: usize,
    cur: usize,
    /// 主干的拐点（分叉从这里长出去）。
    verts: Vec<V3>,
    nv: usize,
    ring: [Vec<f32>; 2],
    grain: Vec<f32>,
    bin: [Vec<f32>; 2],
    fdn: Fdn,
    /// 到这个时间（样本）为止还有雷声或回声；之后整条雷声通路可以跳过。
    live_until: u64,
    /// 统计：打了几次雷。
    pub(super) n_strikes: u64,
}

impl Thunder {
    pub(super) fn new(seed: u64) -> Self {
        Thunder {
            rng: Rng::new(seed ^ 0x0054_4855_4E44_4552),
            next: f64::INFINITY,
            pulses: vec![Pulse::default(); MAX_PULSES],
            n: 0,
            cur: 0,
            verts: vec![[0.0; 3]; 2048],
            nv: 0,
            ring: [vec![0.0; RING], vec![0.0; RING]],
            grain: vec![0.0; GRAIN_MAX],
            bin: [vec![0.0; GRAIN_MAX + H_LEN], vec![0.0; GRAIN_MAX + H_LEN]],
            fdn: Fdn::new(),
            live_until: 0,
            n_strikes: 0,
        }
    }

    /// 雷声滑块变了（0–1）：从关到开时 3–10 秒内来第一声；开着时变了，下一声不晚于新的平均间隔；关掉后不再有新的
    /// 闪电，正在滚的那一声和回声自然结束。
    pub(super) fn set_amount(&mut self, old: f32, new: f32, now: u64) {
        if new <= 0.0 {
            self.next = f64::INFINITY;
        } else if old <= 0.0 {
            self.next = now as f64 + (self.rng.uniform(FIRST_S.0, FIRST_S.1) * FS) as f64;
        } else {
            self.next = self.next.min(now as f64 + (interval_s(new) * FS) as f64);
        }
    }

    #[inline]
    pub(super) fn live(&self, pos: u64) -> bool {
        pos < self.live_until
    }

    /// 取出这一个样本的雷声（直达 + 回声，两耳），`idx` = 环形缓冲的位置。
    #[inline]
    pub(super) fn sample(&mut self, idx: usize) -> [f32; 2] {
        let l = std::mem::take(&mut self.ring[0][idx]);
        let r = std::mem::take(&mut self.ring[1][idx]);
        let (wl, wr) = self.fdn.process(0.5 * (l + r));
        [l + WET * wl, r + WET * wr]
    }

    /// 每个控制块：到时间了就打一次雷（算出整条通道），再把这一块里到达的各段渲染进环形缓冲。
    /// `x`：雷声滑块（0–1）；`az_cdf`：雨声的方位分布；`p_unit`：整片雨声在增益 1 时的功率（定雷的响度）。
    pub(super) fn control(&mut self, x: f32, az_cdf: &[f32; N_AZ], p_unit: f32, pos: u64, t1: f64) {
        if x > 0.0 {
            while self.next < t1 {
                let t = self.next.max(pos as f64);
                self.strike(x, az_cdf, p_unit, t);
                let mean = interval_s(x);
                let gap = MIN_GAP_S + (mean - MIN_GAP_S).max(1.0) * -(1.0 - self.rng.unit()).max(1e-9).ln();
                self.next += (gap * FS) as f64;
            }
        }
        let mut k = 0;
        while self.cur < self.n && self.pulses[self.cur].t < t1 && k < PER_BLOCK {
            let p = self.pulses[self.cur];
            self.cur += 1;
            k += 1;
            self.render(&p, p.t.max(pos as f64) as u64);
        }
    }

    /// 一次闪电：通道、各段的到达时间 / 方向 / 拖开多少，按距离定好总能量，再和还没放完的上一次合在一起按时间排好。
    fn strike(&mut self, x: f32, az_cdf: &[f32; N_AZ], p_unit: f32, t0: f64) {
        self.n_strikes += 1;
        self.pulses.copy_within(self.cur..self.n, 0);
        self.n -= self.cur;
        self.cur = 0;
        let first = self.n;
        let d = (distance_km(x) * self.rng.lognormal(1.0, 0.35)).clamp(D_RANGE_KM.0, D_RANGE_KM.1) * 1000.0;
        let bear = (sample_az(az_cdf, self.rng.unit()) as f32 * AZ_STEP + self.rng.uniform(-2.5, 2.5)).to_radians();
        let base = [d * bear.cos(), d * bear.sin(), 0.0];
        let tn = (self.rng.uniform(T_N.0, T_N.1) * FS) as u16;
        if self.rng.unit() < P_CG {
            // 云地闪：主干（记下拐点）、云里的一段、分叉、回击
            let h = self.rng.uniform(1800.0, 3000.0);
            let top = [base[0] + 300.0 * self.rng.normal(), base[1] + 300.0 * self.rng.normal(), h];
            self.nv = 0;
            let a = self.n;
            self.walk(top, [0.0, 0.0, -1.0], 2.0 * h + 2000.0, [0.0, 0.0, -1.0], true, tn, true);
            let b = self.n;
            let hd = horizontal(&mut self.rng);
            let len = self.rng.uniform(2000.0, 5000.0);
            self.walk(top, hd, len, hd, false, tn, false);
            let branches = self.rng.uniform(2.0, 6.0) as usize;
            for _ in 0..branches {
                if self.nv < 4 {
                    break;
                }
                let start = self.verts[(self.rng.unit() * (self.nv / 2) as f32) as usize];
                let dir = unit([self.rng.normal(), self.rng.normal(), -self.rng.normal().abs() - 0.3]);
                let len = self.rng.uniform(200.0, 1500.0);
                self.walk(start, dir, len, dir, true, tn, false);
            }
            let strokes = self.rng.uniform(1.0, 4.0) as usize;
            let (mut toff, mut amp) = (0.0f32, 1.0f32);
            for _ in 0..strokes {
                toff += self.rng.uniform(0.04, 0.12);
                amp *= self.rng.uniform(0.4, 0.7);
                for i in a..b {
                    if self.n >= MAX_PULSES {
                        break;
                    }
                    let mut q = self.pulses[i];
                    q.t += (toff * FS) as f64;
                    q.amp *= amp;
                    self.pulses[self.n] = q;
                    self.n += 1;
                }
            }
        } else {
            // 云内闪：高处水平的一长段，只有远远的隆隆声
            let h = self.rng.uniform(4000.0, 6500.0);
            let hd = horizontal(&mut self.rng);
            let len = self.rng.uniform(4000.0, 9000.0);
            self.walk([base[0], base[1], h], hd, len, hd, false, tn, false);
        }
        if self.n == first {
            return;
        }
        // 第一声在 t0；总能量按距离定（3 km 时 = 整片雨声 1 秒的能量再高 THUNDER_DB），轻重再有 ±2.5 dB 的不同
        let t_min = self.pulses[first..self.n].iter().map(|p| p.t).fold(f64::INFINITY, f64::min);
        let mut e_raw = 0f64;
        for p in self.pulses[first..self.n].iter() {
            let k = p.pearls as f32;
            let shape = 1.0 / k + (1.0 - 1.0 / k) / (1.0 + p.smear / p.tn as f32).powi(2);
            e_raw += (p.amp * p.amp * p.tn as f32 / 3.0 * shape) as f64;
        }
        let e_target = p_unit * FS * 10f32.powf(THUNDER_DB / 10.0) * (REF_M / d).powi(2) * self.rng.lognormal(1.0, 0.58);
        let g = (e_target as f64 / e_raw.max(1e-30)).sqrt() as f32;
        for p in self.pulses[first..self.n].iter_mut() {
            p.t = t0 + (p.t - t_min);
            p.amp *= g;
        }
        self.pulses[..self.n].sort_unstable_by(|a, b| a.t.total_cmp(&b.t));
    }

    /// 从 p 出发沿 u 走一条曲折的通道（总长 `len` m，每步往 `bias` 偏一点），每段记一个脉冲；`ground`：碰到地面就停；
    /// `record`：记下拐点（主干）。
    #[allow(clippy::too_many_arguments)]
    fn walk(&mut self, mut p: V3, mut u: V3, len: f32, bias: V3, ground: bool, tn: u16, record: bool) {
        let mut done = 0.0;
        while done < len && self.n < MAX_PULSES {
            let l = self.rng.uniform(SEG.0, SEG.1);
            let ang = (TURN_SIGMA * self.rng.normal()).abs().to_radians();
            u = unit(add(turn(&mut self.rng, u, ang), bias, 0.15));
            let q = add(p, u, l);
            self.push(p, q, u, l, tn);
            if record && self.nv < self.verts.len() {
                self.verts[self.nv] = q;
                self.nv += 1;
            }
            p = q;
            done += l;
            if ground && q[2] <= 0.0 {
                break;
            }
        }
    }

    fn push(&mut self, p: V3, q: V3, u: V3, l: f32, tn: u16) {
        // 相对耳朵（x 前、y 左、z 上；耳朵离地 1.2 m）
        let m = [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2]) - H_EAR];
        let r = dot(m, m).sqrt().max(1.0);
        let cos = (dot(u, m) / r).abs();
        let az = m[1].atan2(m[0]).to_degrees();
        let el = m[2].atan2(m[0].hypot(m[1])).to_degrees();
        let ai = ((az.rem_euclid(360.0) / AZ_STEP).round() as usize) % N_AZ;
        let ei = ((el / 15.0).round().max(0.0) as usize).min(SKY_ELS.len() - 1);
        self.pulses[self.n] = Pulse { t: (r / C_AIR * FS) as f64, amp: l / r, smear: l * cos / C_AIR * FS, pearls: ((l / PEARL).round() as u16).max(1), dir: (ei * N_AZ + ai) as u16, r, tn };
        self.n += 1;
    }

    /// 一段的声音：珠子们（N 波，各 1/k）在拖开的时间里随机排开 → 按距离的空气吸收 → 这个方向的 HRIR → 环形缓冲。
    fn render(&mut self, p: &Pulse, at: u64) {
        let tn = (p.tn as usize).max(2);
        let nd = (p.smear as usize).min(GRAIN_MAX - tn - TAIL);
        let len = tn + nd + TAIL;
        let a = 1.0 / p.pearls as f32;
        let inv = 2.0 / tn as f32;
        let g = &mut self.grain[..len];
        g.fill(0.0);
        for _ in 0..p.pearls {
            let off = (self.rng.unit() * nd as f32) as usize;
            for (j, v) in g[off..off + tn].iter_mut().enumerate() {
                *v += a * (1.0 - inv * j as f32);
            }
        }
        let f3 = (1000.0 * (600.0 / p.r).powf(2.0 / 3.0)).clamp(60.0, 0.45 * FS);
        let mut lp = Biquad::lowpass(f3, BUTTER_Q);
        for v in g.iter_mut() {
            *v = p.amp * lp.process(*v);
        }
        let sky = sky();
        let blen = len + H_LEN - 1;
        for e in 0..2 {
            let h = &sky.hrir[(p.dir as usize * 2 + e) * H_LEN..(p.dir as usize * 2 + e + 1) * H_LEN];
            let out = &mut self.bin[e][..blen];
            out.fill(0.0);
            for (k, &hv) in h.iter().enumerate() {
                if hv == 0.0 {
                    continue;
                }
                for (o, &x) in out[k..k + len].iter_mut().zip(&self.grain[..len]) {
                    *o += hv * x;
                }
            }
            ring_add(&mut self.ring[e], at, &self.bin[e][..blen], 1.0);
        }
        self.live_until = self.live_until.max(at + blen as u64 + (1.5 * RT60 * FS) as u64);
    }
}
