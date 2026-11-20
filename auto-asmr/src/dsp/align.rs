//! 多音频合成的自动对齐：同一首歌的几个版本（一段前奏长、一段前奏短，或者干脆是不同的编曲），
//! 自动找到它们"一起开始唱"的地方，从那里叠加；对不上的开头和结尾裁掉。
//!
//! 做法（以第 1 段为基准，其余各段分别和它比）：
//!
//! 1. **粗找**（20 ms 一帧）：每帧取色度（12 个音级的能量占比，和弦 / 旋律）和 24 个对数频带的能量（编曲的轮廓），
//!    减去各维的均值后在所有可能的偏移上做互相关（FFT）。得分按两段**全长**的能量归一化，而不是只按重叠部分：
//!    歌曲的副歌会重复，"拿第一段副歌去对第二段副歌"这种只重叠一小段的偏移在重叠部分里也很像，
//!    但整首歌对得上的只有真正的偏移。
//! 2. **细找**（2 ms 一帧）：在粗找结果 ±80 ms 内，用起音包络（16 个频带的能量上升）在重叠部分里均匀取的几段上再比一次，
//!    抛物线插值到样本。不同编曲的两个版本（波形不同）靠这一步，精度约 1–2 ms。
//! 3. **逐样本**：在几处 2 秒的窗口里直接比波形（±4 ms）。同一份音源剪出来的（一段带前奏一段不带、转过码）
//!    波形几乎一样，相关很高、几处的偏移一致，就用逐样本的结果；否则保留第 2 步的结果。
//!
//! 对齐后各段从各自的起点开始用，长度取共同的部分（最短的那段说了算），在裁开的地方淡入 / 淡出 10 ms，不会咔嗒。

use super::FS;
use realfft::{num_complex::Complex32, RealFftPlanner};

/// 粗找：每帧 20 ms，FFT 4096 点（约 11.7 Hz 一格，低音的音级也分得开）。
const HOP: usize = 960;
const NFFT: usize = 4096;
/// 细找：每帧 2 ms，FFT 2048 点。
const FHOP: usize = 96;
const FNFFT: usize = 2048;
const FBANDS: usize = 16;
/// 粗找结果附近细找的范围（帧，2 ms 一帧）：±80 ms。
const FSEARCH: usize = 40;
/// 细找最多取几段、每段多长（秒）。
const FSEGS: usize = 4;
const FSEG_S: f32 = 15.0;
/// 逐样本比波形：窗口长度（秒）、搜索范围（样本，±4 ms）、窗口数。
const WIN_S: f32 = 2.0;
const WSEARCH: usize = 192;
const WINS: usize = 3;
/// 裁开处的淡入淡出（样本，10 ms）。
pub const FADE: usize = 480;

/// 一段（第 k 段）相对第 1 段的对齐结果。
#[derive(Clone, Debug, PartialEq)]
pub struct PairInfo {
    /// 偏移（样本）：这一段的第 `t + offset` 个样本对应第 1 段的第 `t` 个样本。正数 = 这一段的前奏更长。
    pub offset: i64,
    /// 整体得分（色度 + 频带，0–2）和次好的偏移的得分：两者差得越多越可靠。
    pub score: f32,
    pub second: f32,
    /// 重叠部分里的相关（0–1）。
    pub local_r: f32,
    /// 逐样本（波形一致）。
    pub exact: bool,
    /// 细找时几段各自的偏移相差多少（ms）：大了说明两段速度不完全一样，叠加后会慢慢错开。
    pub drift_ms: f32,
    /// 是否可靠。
    pub reliable: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Alignment {
    /// 每段从第几个样本开始用（裁掉的开头）。
    pub starts: Vec<usize>,
    /// 每段原来的长度（样本）。
    pub lens: Vec<usize>,
    /// 对齐后共同部分的长度（样本）。
    pub len: usize,
    /// 第 2 段起各段相对第 1 段的情况（`pairs[k - 1]` 是第 k 段）。
    pub pairs: Vec<PairInfo>,
}

impl Alignment {
    /// 第 k 段对齐、裁剪后的样本：从 `starts[k]` 起取 `len` 个，在裁开的地方淡入 / 淡出。
    pub fn apply(&self, k: usize, x: &[f32]) -> Vec<f32> {
        let s = self.starts[k].min(x.len());
        let e = (s + self.len).min(x.len());
        let mut y = x[s..e].to_vec();
        let n = y.len();
        let f = FADE.min(n / 2);
        let ramp = |i: usize| 0.5 - 0.5 * (std::f32::consts::PI * (i as f32 + 0.5) / f as f32).cos();
        if s > 0 {
            for i in 0..f {
                y[i] *= ramp(i);
            }
        }
        if e < x.len() {
            for i in 0..f {
                y[n - 1 - i] *= ramp(i);
            }
        }
        y
    }

    /// 每段对齐后的开头和结尾：从原来的第几秒用到第几秒。
    pub fn spans(&self) -> Vec<(f64, f64)> {
        self.starts.iter().map(|&s| (s as f64 / FS as f64, (s + self.len) as f64 / FS as f64)).collect()
    }

    /// 给人看的说明：每段从哪里开始用、裁掉多少，结尾裁到哪里，以及要注意的地方。`names`：各段的文件名。
    pub fn describe(&self, names: &[String], english: bool) -> String {
        let mut lines = Vec::new();
        let total = self.len as f64 / FS as f64;
        lines.push(if english {
            format!("Auto-align: the files overlap for {} and are mixed from there", clock(total))
        } else {
            format!("自动对齐：共同部分 {}，从这里开始叠加", clock(total))
        });
        for (k, (a, b)) in self.spans().into_iter().enumerate() {
            let name = names.get(k).map(String::as_str).unwrap_or("");
            let full = self.lens[k] as f64 / FS as f64;
            let tail = (full - b).max(0.0);
            let head = if a < 0.005 {
                if english { "start kept".to_string() } else { "开头不裁".to_string() }
            } else if english {
                format!("starts at {} (first {:.2} s cut)", clock(a), a)
            } else {
                format!("从 {} 开始（裁掉开头 {:.2} s）", clock(a), a)
            };
            let end = if tail < 0.005 {
                if english { "end kept".to_string() } else { "结尾不裁".to_string() }
            } else if english {
                format!("ends at {} (last {:.2} s cut)", clock(b), tail)
            } else {
                format!("用到 {}（裁掉结尾 {:.2} s）", clock(b), tail)
            };
            lines.push(if english { format!("  {}. {name}: {head}, {end}", k + 1) } else { format!("  {}. {name}：{head}，{end}", k + 1) });
        }
        for (k, p) in self.pairs.iter().enumerate() {
            let k = k + 2;
            if !p.reliable {
                lines.push(if english {
                    format!("  ! File {k} does not clearly match file 1 (maybe not the same song); check the result by ear")
                } else {
                    format!("  ！第 {k} 段和第 1 段没有明显对得上的地方（可能不是同一首歌），请试听确认")
                });
            } else if p.drift_ms > 40.0 {
                lines.push(if english {
                    format!("  ! Files 1 and {k} differ slightly in tempo (offset varies by {:.0} ms), so they drift apart over time", p.drift_ms)
                } else {
                    format!("  ！第 1、{k} 段的速度不完全一样（前后偏移差 {:.0} ms），叠加后会慢慢错开", p.drift_ms)
                });
            }
        }
        lines.join("\n")
    }
}

/// 秒 → "分:秒.百分秒"。
pub fn clock(s: f64) -> String {
    let s = s.max(0.0);
    let m = (s / 60.0).floor();
    format!("{}:{:05.2}", m as u64, s - 60.0 * m)
}

/// 对齐几段 48 kHz 单声道音频（以第 1 段为基准）。少于两段、或者有一段太短（不到 3 秒）时报错。
pub fn align(tracks: &[&[f32]]) -> Result<Alignment, String> {
    if tracks.len() < 2 {
        return Err("自动对齐至少需要两段音频".into());
    }
    if tracks.iter().any(|x| x.len() < 3 * FS as usize) {
        return Err("自动对齐需要每段至少 3 秒".into());
    }
    let feats: Vec<Coarse> = std::thread::scope(|s| {
        let hs: Vec<_> = tracks.iter().map(|x| s.spawn(move || Coarse::new(x))).collect();
        hs.into_iter().map(|h| h.join().expect("对齐的特征计算出错")).collect()
    });
    let pairs: Vec<PairInfo> = std::thread::scope(|s| {
        let hs: Vec<_> = (1..tracks.len()).map(|k| {
            let (fa, fb, a, b) = (&feats[0], &feats[k], tracks[0], tracks[k]);
            s.spawn(move || pair(fa, fb, a, b))
        }).collect();
        hs.into_iter().map(|h| h.join().expect("对齐出错")).collect()
    });
    let mut offs: Vec<i64> = vec![0];
    offs.extend(pairs.iter().map(|p| p.offset));
    let lo = *offs.iter().min().unwrap();
    let starts: Vec<usize> = offs.iter().map(|&o| (o - lo) as usize).collect();
    let lens: Vec<usize> = tracks.iter().map(|x| x.len()).collect();
    let len = starts.iter().zip(&lens).map(|(&s, &l)| l.saturating_sub(s)).min().unwrap_or(0);
    if len < FS as usize {
        return Err("对齐后几段的共同部分不到 1 秒，没法叠加".into());
    }
    Ok(Alignment { starts, lens, len, pairs })
}

/// 粗找用的逐帧特征：色度（12）和对数频带能量（24），都已减去各维的均值。
struct Coarse {
    chroma: Vec<Vec<f32>>,
    bands: Vec<Vec<f32>>,
    frames: usize,
}

const NB: usize = 24;

impl Coarse {
    fn new(x: &[f32]) -> Self {
        let frames = x.len() / HOP + 1;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(NFFT);
        let win = hann(NFFT);
        let df = FS / NFFT as f32;
        // 每个频率格属于哪个音级（80 Hz–4 kHz）、哪个频带（60 Hz–10 kHz，对数等分）
        let pc: Vec<Option<usize>> = (0..=NFFT / 2)
            .map(|i| {
                let f = i as f32 * df;
                (80.0..4000.0).contains(&f).then(|| ((12.0 * (f / 440.0).log2() + 69.0).round() as i64).rem_euclid(12) as usize)
            })
            .collect();
        let band: Vec<Option<usize>> = (0..=NFFT / 2)
            .map(|i| {
                let f = i as f32 * df;
                (60.0..10000.0).contains(&f).then(|| (((f / 60.0).ln() / (10000.0f32 / 60.0).ln() * NB as f32) as usize).min(NB - 1))
            })
            .collect();
        let mut chroma = vec![vec![0f32; frames]; 12];
        let mut bands = vec![vec![0f32; frames]; NB];
        let mut buf = fft.make_input_vec();
        let mut spec = fft.make_output_vec();
        let mut scratch = fft.make_scratch_vec();
        for t in 0..frames {
            frame(x, t as i64 * HOP as i64 - (NFFT / 2) as i64, &win, &mut buf);
            let _ = fft.process_with_scratch(&mut buf, &mut spec, &mut scratch);
            for (i, c) in spec.iter().enumerate() {
                let p = c.norm_sqr();
                if let Some(k) = pc[i] {
                    chroma[k][t] += p;
                }
                if let Some(k) = band[i] {
                    bands[k][t] += p;
                }
            }
        }
        // 色度按每帧的总量归一化（静音帧加一个按整段平均能量定的底，接近 0）；频带取对数
        let tot: Vec<f32> = (0..frames).map(|t| (0..12).map(|k| chroma[k][t]).sum()).collect();
        let eps = 1e-3 * tot.iter().sum::<f32>() / frames as f32 + 1e-20;
        for c in chroma.iter_mut() {
            for (v, s) in c.iter_mut().zip(&tot) {
                *v /= s + eps;
            }
        }
        let btot: f32 = bands.iter().map(|b| b.iter().sum::<f32>()).sum::<f32>() / (frames * NB) as f32;
        let floor = 1e-4 * btot + 1e-20;
        for b in bands.iter_mut() {
            for v in b.iter_mut() {
                *v = (*v + floor).ln();
            }
        }
        for d in chroma.iter_mut().chain(bands.iter_mut()) {
            let m = d.iter().sum::<f32>() / frames as f32;
            for v in d.iter_mut() {
                *v -= m;
            }
        }
        Coarse { chroma, bands, frames }
    }
}

fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos()).collect()
}

/// 取从第 `start` 个样本开始的一帧（超出范围的补 0），乘窗。
fn frame(x: &[f32], start: i64, win: &[f32], out: &mut [f32]) {
    for (i, (o, w)) in out.iter_mut().zip(win).enumerate() {
        let j = start + i as i64;
        *o = if j >= 0 && (j as usize) < x.len() { x[j as usize] * w } else { 0.0 };
    }
}

/// 几维特征序列的互相关之和：`r[lag]` = Σ_d Σ_t a_d[t]·b_d[t + lag]，按两边全长的能量归一化。
/// 返回 (lag, 得分)，只含重叠不少于 `min_ov` 帧的偏移。
fn xcorr(a: &[Vec<f32>], b: &[Vec<f32>], ta: usize, tb: usize, min_ov: usize) -> Vec<(i64, f32)> {
    let n = (ta + tb).next_power_of_two();
    let mut planner = RealFftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    let mut acc = vec![Complex32::new(0.0, 0.0); n / 2 + 1];
    let mut ea = 0f64;
    let mut eb = 0f64;
    let mut ba = fwd.make_input_vec();
    let mut bb = fwd.make_input_vec();
    let mut sa = fwd.make_output_vec();
    let mut sb = fwd.make_output_vec();
    for (da, db) in a.iter().zip(b) {
        ba.fill(0.0);
        bb.fill(0.0);
        ba[..ta].copy_from_slice(&da[..ta]);
        bb[..tb].copy_from_slice(&db[..tb]);
        ea += da.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        eb += db.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        let _ = fwd.process(&mut ba, &mut sa);
        let _ = fwd.process(&mut bb, &mut sb);
        for ((o, x), y) in acc.iter_mut().zip(&sa).zip(&sb) {
            *o += x.conj() * y;
        }
    }
    let mut r = inv.make_output_vec();
    acc[0].im = 0.0;
    if let Some(last) = acc.last_mut() {
        last.im = 0.0;
    }
    let _ = inv.process(&mut acc, &mut r);
    let norm = (n as f64 * (ea * eb).sqrt()).max(1e-30);
    let mut out = Vec::new();
    for lag in -(ta as i64)..(tb as i64) {
        let t0 = 0i64.max(-lag);
        let t1 = (ta as i64).min(tb as i64 - lag);
        if t1 - t0 < min_ov as i64 {
            continue;
        }
        let v = r[lag.rem_euclid(n as i64) as usize] as f64 / norm;
        out.push((lag, v as f32));
    }
    out
}

/// 重叠部分里的相关（色度 + 频带，按重叠部分的能量归一化，0–1）。
fn local_corr(fa: &Coarse, fb: &Coarse, lag: i64) -> f32 {
    let t0 = 0i64.max(-lag) as usize;
    let t1 = (fa.frames as i64).min(fb.frames as i64 - lag).max(t0 as i64) as usize;
    let mut rs = 0.0;
    for (a, b) in [(&fa.chroma, &fb.chroma), (&fa.bands, &fb.bands)] {
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (da, db) in a.iter().zip(b.iter()) {
            for t in t0..t1 {
                let (x, y) = (da[t] as f64, db[(t as i64 + lag) as usize] as f64);
                ab += x * y;
                aa += x * x;
                bb += y * y;
            }
        }
        rs += ab / (aa * bb).sqrt().max(1e-30);
    }
    (rs / 2.0) as f32
}

/// 一段（b）相对基准（a）的对齐。
fn pair(fa: &Coarse, fb: &Coarse, a: &[f32], b: &[f32]) -> PairInfo {
    // 1. 粗找：重叠至少是短的那段的一半（且不少于 3 秒）
    let min_ov = (fa.frames.min(fb.frames) / 2).max((3.0 * FS) as usize / HOP);
    let rc = xcorr(&fa.chroma, &fb.chroma, fa.frames, fb.frames, min_ov);
    let rb = xcorr(&fa.bands, &fb.bands, fa.frames, fb.frames, min_ov);
    let scores: Vec<(i64, f32)> = rc.iter().zip(&rb).map(|(&(l, x), &(_, y))| (l, x + y)).collect();
    let &(lag, score) = scores.iter().max_by(|p, q| p.1.total_cmp(&q.1)).expect("至少有一个偏移");
    // 次好的：离最好的 1.5 秒以外
    let away = (1.5 * FS) as i64 / HOP as i64;
    let second = scores.iter().filter(|p| (p.0 - lag).abs() > away).map(|p| p.1).fold(0f32, f32::max);
    let local_r = local_corr(fa, fb, lag);
    // 2. 细找：在粗找结果 ±80 ms 内比起音包络
    let coarse = lag * HOP as i64;
    let (fine, drift_ms) = refine_flux(a, b, coarse);
    // 3. 逐样本：几处 2 秒的窗口直接比波形
    let (offset, exact) = match refine_wave(a, b, fine) {
        Some(o) => (o, true),
        None => (fine, false),
    };
    let reliable = exact || (local_r >= 0.25 && score >= 1.25 * second);
    PairInfo { offset, score, second, local_r, exact, drift_ms, reliable }
}

/// 起音包络：从第 `start` 个样本起、`n` 帧（2 ms 一帧），16 个对数频带的能量上升（对数差的正部），各维减去均值。
fn flux(x: &[f32], start: i64, n: usize) -> Vec<Vec<f32>> {
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FNFFT);
    let win = hann(FNFFT);
    let df = FS / FNFFT as f32;
    let band: Vec<Option<usize>> = (0..=FNFFT / 2)
        .map(|i| {
            let f = i as f32 * df;
            (60.0..10000.0).contains(&f).then(|| (((f / 60.0).ln() / (10000.0f32 / 60.0).ln() * FBANDS as f32) as usize).min(FBANDS - 1))
        })
        .collect();
    let mut buf = fft.make_input_vec();
    let mut spec = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();
    let mut lv = vec![vec![0f32; n + 1]; FBANDS];
    for t in 0..=n {
        frame(x, start + (t as i64 - 1) * FHOP as i64 - (FNFFT / 2) as i64, &win, &mut buf);
        let _ = fft.process_with_scratch(&mut buf, &mut spec, &mut scratch);
        for (i, c) in spec.iter().enumerate() {
            if let Some(k) = band[i] {
                lv[k][t] += c.norm_sqr();
            }
        }
    }
    let mean: f32 = lv.iter().map(|b| b.iter().sum::<f32>()).sum::<f32>() / ((n + 1) * FBANDS) as f32;
    let floor = 1e-4 * mean + 1e-20;
    lv.iter()
        .map(|b| {
            let mut d: Vec<f32> = (1..=n).map(|t| ((b[t] + floor).ln() - (b[t - 1] + floor).ln()).max(0.0)).collect();
            let m = d.iter().sum::<f32>() / n.max(1) as f32;
            d.iter_mut().for_each(|v| *v -= m);
            d
        })
        .collect()
}

/// 重叠部分里均匀取几段（每段最长 15 秒），返回各段在 a 里的起点（样本）和长度（样本）。
fn segments(la: usize, lb: usize, off: i64, n: usize, seg_s: f32) -> Vec<(usize, usize)> {
    let s0 = 0i64.max(-off) as usize;
    let s1 = (la as i64).min(lb as i64 - off).max(s0 as i64) as usize;
    let ov = s1 - s0;
    let seg = ((seg_s * FS) as usize).min(ov);
    if seg == 0 {
        return Vec::new();
    }
    let k = n.min(ov / seg).max(1);
    let gap = if k > 1 { (ov - seg) / (k - 1) } else { 0 };
    (0..k).map(|i| (s0 + i * gap, seg)).collect()
}

/// 细找：在 `coarse`（样本）附近 ±80 ms 里，用重叠部分里几段的起音包络找最好的偏移，抛物线插值到样本。
/// 同时返回各段单独找时的偏移相差多少（ms）。
fn refine_flux(a: &[f32], b: &[f32], coarse: i64) -> (i64, f32) {
    let segs = segments(a.len(), b.len(), coarse, FSEGS, FSEG_S);
    let m = FSEARCH;
    let mut total = vec![0f64; 2 * m + 1];
    let mut per_seg = Vec::new();
    for &(s, len) in &segs {
        let n = len / FHOP;
        if n < 50 {
            continue;
        }
        let fa = flux(a, s as i64, n);
        let fb = flux(b, s as i64 + coarse - (m * FHOP) as i64, n + 2 * m);
        let ea: f64 = fa.iter().flatten().map(|v| (*v as f64).powi(2)).sum();
        let mut r = vec![0f64; 2 * m + 1];
        for (d, rv) in r.iter_mut().enumerate() {
            let (mut ab, mut bb) = (0f64, 0f64);
            for (da, db) in fa.iter().zip(&fb) {
                for t in 0..n {
                    let y = db[t + d] as f64;
                    ab += da[t] as f64 * y;
                    bb += y * y;
                }
            }
            *rv = ab / (ea * bb).sqrt().max(1e-30);
        }
        for (t, v) in total.iter_mut().zip(&r) {
            *t += v;
        }
        per_seg.push(peak(&r));
    }
    if per_seg.is_empty() {
        return (coarse, 0.0);
    }
    let best = peak(&total);
    let off = coarse + ((best - m as f64) * FHOP as f64).round() as i64;
    let lo = per_seg.iter().cloned().fold(f64::MAX, f64::min);
    let hi = per_seg.iter().cloned().fold(f64::MIN, f64::max);
    (off, ((hi - lo) * FHOP as f64 / FS as f64 * 1000.0) as f32)
}

/// 最大值的位置，抛物线插值到小数。
fn peak(r: &[f64]) -> f64 {
    let i = r.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).map(|p| p.0).unwrap_or(0);
    if i == 0 || i + 1 >= r.len() {
        return i as f64;
    }
    let (y0, y1, y2) = (r[i - 1], r[i], r[i + 1]);
    let den = y0 - 2.0 * y1 + y2;
    if den.abs() < 1e-12 {
        i as f64
    } else {
        i as f64 + (0.5 * (y0 - y2) / den).clamp(-0.5, 0.5)
    }
}

/// 逐样本：在重叠部分里几处 2 秒的窗口（各取所在一段里最响的地方）直接比波形，±4 ms。
/// 每处都高度相关（|r| ≥ 0.6）且偏移一致（相差不超过 2 个样本）时返回这个偏移。
fn refine_wave(a: &[f32], b: &[f32], fine: i64) -> Option<i64> {
    let w = (WIN_S * FS) as usize;
    let segs = segments(a.len(), b.len(), fine, WINS, 12.0);
    if segs.len() < WINS.min(2) {
        return None;
    }
    let d = WSEARCH as i64;
    let mut found = Vec::new();
    for &(s, len) in &segs {
        if len < w {
            return None;
        }
        // 这一段里最响的 2 秒（0.25 秒一步）
        let step = (FS / 4.0) as usize;
        let s = (s..=s + len - w).step_by(step).max_by(|&p, &q| energy(&a[p..p + w]).total_cmp(&energy(&a[q..q + w])))?;
        let xa = &a[s..s + w];
        let ea = energy(xa);
        if ea < 1e-6 {
            return None;
        }
        let mut best = (0f64, 0i64);
        for lag in -d..=d {
            let o = s as i64 + fine + lag;
            if o < 0 || o as usize + w > b.len() {
                continue;
            }
            let xb = &b[o as usize..o as usize + w];
            let ab: f64 = xa.iter().zip(xb).map(|(x, y)| *x as f64 * *y as f64).sum();
            let r = ab / (ea * energy(xb)).sqrt().max(1e-30);
            if r.abs() > best.0 {
                best = (r.abs(), fine + lag);
            }
        }
        if best.0 < 0.6 {
            return None;
        }
        found.push(best.1);
    }
    let (lo, hi) = (*found.iter().min()?, *found.iter().max()?);
    (hi - lo <= 2).then(|| {
        found.sort_unstable();
        found[found.len() / 2]
    })
}

fn energy(x: &[f32]) -> f64 {
    x.iter().map(|v| (*v as f64).powi(2)).sum()
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// 一段像歌的测试信号：随机的音符（基频 110–440 Hz，带泛音）、随机时长（0.15–0.6 s），
    /// 加上随机间隔的打击声；不同的种子得到不同的"歌"。没有周期性，不会有多个一样好的偏移。
    pub fn test_song(seconds: f32, seed: u32) -> Vec<f32> {
        let n = (seconds * FS) as usize;
        let mut x = vec![0f32; n];
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
        let mut rnd = move || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32
        };
        let mut i = 0;
        while i < n {
            let dur = ((0.15 + 0.45 * rnd()) * FS) as usize;
            let f0 = 110.0 * 2f32.powf((rnd() * 24.0).floor() / 12.0);
            let amp = 0.05 + 0.1 * rnd();
            for j in 0..dur.min(n - i) {
                let t = j as f32 / FS;
                let env = (1.0 - j as f32 / dur as f32).powf(0.7) * (t / 0.01).min(1.0);
                let mut v = 0.0;
                for h in 1..6 {
                    v += (std::f32::consts::TAU * f0 * h as f32 * t).sin() / h as f32;
                }
                x[i + j] += amp * env * v;
            }
            i += dur;
        }
        let mut t = 0usize;
        while t < n {
            t += ((0.2 + 0.5 * rnd()) * FS) as usize;
            let a = 0.3 * rnd();
            for j in 0..((0.03 * FS) as usize).min(n.saturating_sub(t)) {
                x[t + j] += a * (rnd() - 0.5) * (-(j as f32) / (0.005 * FS)).exp();
            }
        }
        x
    }

    /// 简单的一阶低通 + 增益（模拟转码 / 不同的母带处理）。
    fn color(x: &[f32], k: f32, g: f32) -> Vec<f32> {
        let mut y = 0.0;
        x.iter().map(|&v| {
            y += k * (v - y);
            g * y
        }).collect()
    }

    #[test]
    fn same_recording_with_extra_intro_is_sample_exact() {
        // A = 7.37 秒前奏 + 歌；B = 同一首歌（低通、音量低 6 dB），没有前奏，还多 3 秒别的结尾
        let song = test_song(40.0, 1);
        let intro = test_song(7.37, 2);
        let outro = test_song(3.0, 3);
        let a: Vec<f32> = intro.iter().chain(&song).copied().collect();
        let b: Vec<f32> = color(&song, 0.6, 0.5).into_iter().chain(outro).collect();
        let al = align(&[&a, &b]).unwrap();
        let k = intro.len();
        // a 从前奏之后开始；b 开头不裁；长度 = 歌的长度（b 多出来的结尾裁掉）
        assert!((al.starts[0] as i64 - k as i64).abs() <= 1, "a 从 {} 开始，应为 {k}", al.starts[0]);
        assert_eq!(al.starts[1], 0);
        assert!((al.len as i64 - song.len() as i64).abs() <= 1);
        assert!(al.pairs[0].exact && al.pairs[0].reliable, "{:?}", al.pairs[0]);
        // 裁开的地方淡入
        let ya = al.apply(0, &a);
        assert_eq!(ya.len(), al.len);
        assert!(ya[0].abs() < 1e-3 && (ya[FADE + 10] - a[al.starts[0] + FADE + 10]).abs() < 1e-9);
    }

    #[test]
    fn different_arrangement_aligns_within_two_ms() {
        // 同一段"旋律"，两种"编曲"：B 的打击声不同、音色更暗，不能逐样本，但起音和音高一致；B 的前奏更长（5.5 秒）
        let melody = test_song(45.0, 7);
        let drums_a = test_song(45.0, 8);
        let a: Vec<f32> = melody.iter().zip(&drums_a).map(|(m, d)| m + 0.3 * d).collect();
        let pre = (5.5 * FS) as usize;
        let mut b = vec![0f32; pre];
        b.extend(color(&melody, 0.3, 0.8));
        let noise = test_song(b.len() as f32 / FS + 1.0, 99);
        for (v, n) in b.iter_mut().zip(&noise) {
            *v += 0.15 * n;
        }
        let al = align(&[&a, &b]).unwrap();
        let err_ms = (al.starts[1] as f64 - pre as f64) / FS as f64 * 1000.0;
        assert_eq!(al.starts[0], 0);
        assert!(err_ms.abs() < 2.0, "偏差 {err_ms:.2} ms：{:?}", al.pairs[0]);
        assert!(al.pairs[0].reliable && al.pairs[0].drift_ms < 10.0, "{:?}", al.pairs[0]);
    }

    #[test]
    fn three_tracks_and_unrelated_track() {
        let song = test_song(30.0, 11);
        let a: Vec<f32> = test_song(2.0, 12).into_iter().chain(song.iter().copied()).collect();
        let b = song.clone();
        let c: Vec<f32> = test_song(9.0, 13).into_iter().chain(song.iter().copied()).collect();
        let al = align(&[&a, &b, &c]).unwrap();
        // 最晚开始唱的是 c（9 秒前奏）：b 开头不裁，a 裁 2 秒，c 裁 9 秒
        assert_eq!(al.starts, vec![(2.0 * FS) as usize, 0, (9.0 * FS) as usize]);
        assert!(al.pairs.iter().all(|p| p.reliable));
        // 毫不相干的一段：标为不可靠
        let other = test_song(30.0, 77);
        let bad = align(&[&a, &other]).unwrap();
        assert!(!bad.pairs[0].reliable, "{:?}", bad.pairs[0]);
        assert!(bad.describe(&["a".into(), "x".into()], false).contains("可能不是同一首歌"));
        // 太短：报错
        assert!(align(&[&a, &a[..FS as usize]]).is_err());
    }
}
