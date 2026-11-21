//! 多音频合成的自动平衡：让几个音频（各在头边的一个位置，比如一左一右）**每一段时间**听起来都差不多一样响。
//!
//! 合成本来就按整段的积分响度把各音频对齐到一样响（[`crate::dsp::mix::balance_gains`]），但那只是整段平均。
//! 实际听到的一边响一边轻，主要来自空间化：两个声音各自忽远忽近、上下换位，常常一个退开了另一个正贴着耳朵；
//! 其次才是内容本身（一段在副歌里大声、另一段在主歌里）。所以这里平衡的是每个声音**空间化之后**的响度：
//!
//! - 声音 k 此刻听起来多响 = 它输入的短时响度（以当前时刻为中心的 3 秒，K 加权，加上整段对齐的增益）
//!   + 它此刻的"位置增益"（引擎输出与输入的 K 加权能量之比，0.3 秒平滑；只跟位置有关，不随字句起伏）。
//!   输入的短时响度由整段事先算好（[`BalanceData`]），试听和文件转换用的是同一份；主播模式的左右双人没法事先算，
//!   边听边算（[`LiveLevels`]：只往回看，两个声音都是麦克风，共用一份）。
//! - 在出声的几个声音（不在自己的长停顿里）取平均作为目标，各自要补的分贝 = 目标 − 自己（最多 ±15 dB），乘上强度
//!   （0–100%：100% = 一样响，50% = 差距减半），0.25 秒平滑地加到它的输出上。停顿里的声音沿用之前的增益，不会把底噪拉上来。
//!
//! 平衡的是声音之间的差，几个声音一起退开、一起变轻时不动；但一个声音单独退开时会被补回一部分（强度越大补得越多），
//! 所以忽远忽近的响度变化会变小，方向、远耳、音色这些远近线索照常。

use super::dynamics::KWeight;
use super::{db_to_lin, FS};

/// 输入响度每 10 ms 一个点。
pub const HOP: usize = 480;
/// 短时响度的窗长（点数，3 秒）、判断是否在出声的窗长（0.4 秒）、最多补多少 dB、低于自己整段响度多少 dB 算停顿。
const WIN: usize = 300;
const MOM: usize = 40;
const MAX_DB: f32 = 15.0;
const PAUSE_DB: f32 = 15.0;
/// 运行时：每 64 个样本更新一次；位置增益的平滑、补偿的平滑（秒）。
pub const STEP: usize = 64;
const TRANSFER_TAU: f32 = 0.3;
const GAIN_TAU: f32 = 0.25;

/// 平衡器要的输入响度：事先由整段算好的 [`BalanceData`]，或主播模式边听边算的 [`LiveLevels`]。
pub trait Levels {
    /// 第 k 个声音的输入在第 `sample` 个样本处的（K 加权均方, 短时响度, 是否在出声）；没有数据时是无声。
    fn at(&self, k: usize, sample: i64) -> (f32, f32, bool);
}

/// 一段音频每 10 ms 的 K 加权均方（最后不满 10 ms 的也算一个点）。
pub fn levels(x: &[f32]) -> Vec<f32> {
    let mut kw = KWeight::new();
    x.chunks(HOP)
        .map(|c| {
            let s: f64 = c.iter().map(|&v| (kw.process(v) as f64).powi(2)).sum();
            (s / HOP as f64) as f32
        })
        .collect()
}

/// 事先由整段输入算好的：每段每 10 ms 的 K 加权均方、短时响度（含整段对齐的增益）、是否在出声。
#[derive(Debug, Clone, PartialEq)]
pub struct BalanceData {
    lv: Vec<Vec<f32>>,
    st: Vec<Vec<f32>>,
    active: Vec<Vec<bool>>,
}

impl BalanceData {
    /// `levels`：各段的 [`levels`]；`static_db`：整段对齐给各段的增益（dB）。
    pub fn new(levels: &[&[f32]], static_db: &[f32]) -> Self {
        let (mut st, mut active) = (Vec::new(), Vec::new());
        for (l, &g) in levels.iter().zip(static_db) {
            // 是否在出声：以当前时刻为中心 0.4 秒的响度不低于自己整段响度 15 dB 以下
            let own = own_loudness(l);
            let act: Vec<bool> = window_mean(l, MOM, None).into_iter().map(|ms| lufs(ms) > own - PAUSE_DB && lufs(ms) > -60.0).collect();
            // 短时响度：3 秒里只按出声的部分平均（停顿不把它拉低），再加上整段对齐的增益
            st.push(window_mean(l, WIN, Some(&act)).into_iter().map(|ms| lufs(ms) + g).collect());
            active.push(act);
        }
        BalanceData { lv: levels.iter().map(|l| l.to_vec()).collect(), st, active }
    }
}

impl Levels for BalanceData {
    /// 第 k 段第 `sample` 个样本处的值；超出这一段时是无声。
    fn at(&self, k: usize, sample: i64) -> (f32, f32, bool) {
        if sample < 0 {
            return (0.0, -120.0, false);
        }
        let j = sample as usize / HOP;
        match (self.lv.get(k).and_then(|l| l.get(j)), self.st.get(k).and_then(|s| s.get(j))) {
            (Some(&l), Some(&s)) => (l, s, self.active[k][j]),
            _ => (0.0, -120.0, false),
        }
    }
}

/// 以每个点为中心、`w` 个点的平均均方；给了 `only` 时只平均其中为真的点（一个都没有时是 0）。
fn window_mean(l: &[f32], w: usize, only: Option<&[bool]>) -> Vec<f64> {
    let n = l.len();
    let (mut cum, mut cnt) = (vec![0f64; n + 1], vec![0usize; n + 1]);
    for t in 0..n {
        let use_t = only.map_or(true, |o| o[t]);
        cum[t + 1] = cum[t] + if use_t { l[t] as f64 } else { 0.0 };
        cnt[t + 1] = cnt[t] + use_t as usize;
    }
    (0..n)
        .map(|t| {
            let a = t.saturating_sub(w / 2);
            let b = (t + w / 2).min(n);
            let k = if only.is_some() { cnt[b] - cnt[a] } else { w };
            if k == 0 { 0.0 } else { (cum[b] - cum[a]) / k as f64 }
        })
        .collect()
}

fn lufs(ms: f64) -> f32 {
    (-0.691 + 10.0 * (ms + 1e-12).log10()) as f32
}

/// 整段响度（按 400 ms 块、绝对门限 −70、相对门限 −10，近似 BS.1770 积分响度）。
fn own_loudness(l: &[f32]) -> f32 {
    let blocks: Vec<f64> = l.chunks(40).map(|c| c.iter().map(|&v| v as f64).sum::<f64>() / c.len() as f64).collect();
    let lufs = |ms: f64| -0.691 + 10.0 * (ms + 1e-12).log10();
    let gated = |thr: f64| {
        let v: Vec<f64> = blocks.iter().copied().filter(|&b| lufs(b) > thr).collect();
        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
    };
    let rel = lufs(gated(-70.0)) - 10.0;
    lufs(gated(rel)) as f32
}

/// 边听边算时记多少个点（5.12 秒）：要装得下短时响度的 3 秒窗，再加左右双人最长的延迟（300 ms）和引擎时延。
const LIVE_RING: usize = 512;
const _: () = assert!(LIVE_RING >= WIN + 2 * MOM);
/// 边听边算的"说话响度"往上、往下跟过去的时间常数（秒）。
const OWN_UP: f32 = 2.0;
const OWN_DOWN: f32 = 20.0;

/// 主播模式边听边算的输入响度（只能往回看）。左右双人的两个声音都是麦克风，所以只记一份时间线上的麦克风，
/// 各声音按自己的延迟往回取（[`Levels::at`] 不看 k）。和整段事先算好的 [`BalanceData`] 的区别：
///
/// - 短时响度取**过去** 3 秒里出声的部分（不是以当前为中心）；是否在出声看过去 0.4 秒。
/// - "自己的整段响度"换成慢慢跟着走的说话响度：400 ms 一块，高于 −70 LUFS、且不低于它 10 dB 的块才算
///   （和整段响度的两道门限一样），往上 2 秒、往下 20 秒跟过去。还没有说过话时一律算停顿。
/// - 正在攒的这 10 ms 还没有值，用最近完成的那一点代替。
///
/// 两个声音的短时响度相同，平衡时互相抵消，起作用的是位置增益（和文件的左右双人一样）。不分配内存，可以在音频线程里用。
pub struct LiveLevels {
    kw: KWeight,
    acc: f64,
    n: usize,
    /// 已经完成了多少个点；各点按 `序号 % LIVE_RING` 放。
    hops: usize,
    lv: Vec<f32>,
    st: Vec<f32>,
    act: Vec<bool>,
    blk: f64,
    blk_n: usize,
    own: Option<f32>,
}

impl Default for LiveLevels {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveLevels {
    pub fn new() -> Self {
        LiveLevels { kw: KWeight::new(), acc: 0.0, n: 0, hops: 0, lv: vec![0.0; LIVE_RING], st: vec![-120.0; LIVE_RING], act: vec![false; LIVE_RING], blk: 0.0, blk_n: 0, own: None }
    }

    /// 送进时间线上接下来的输入样本（从第 0 个样本起连续送）。
    pub fn feed(&mut self, x: &[f32]) {
        for &v in x {
            let y = self.kw.process(v) as f64;
            self.acc += y * y;
            self.n += 1;
            if self.n == HOP {
                self.push((self.acc / HOP as f64) as f32);
                self.acc = 0.0;
                self.n = 0;
            }
        }
    }

    /// 完成一个点（10 ms 的 K 加权均方 `ms`）：更新说话响度，算这一点是否在出声、短时响度。
    fn push(&mut self, ms: f32) {
        let h = self.hops;
        let i = h % LIVE_RING;
        self.lv[i] = ms;
        self.blk += ms as f64;
        self.blk_n += 1;
        if self.blk_n == MOM {
            let l = lufs(self.blk / MOM as f64);
            match self.own {
                _ if l <= -70.0 => {}
                None => self.own = Some(l),
                Some(o) if l > o - 10.0 => {
                    let tau = if l > o { OWN_UP } else { OWN_DOWN };
                    let a = 1.0 - (-((MOM * HOP) as f32) / FS / tau).exp();
                    self.own = Some(o + a * (l - o));
                }
                Some(_) => {}
            }
            self.blk = 0.0;
            self.blk_n = 0;
        }
        // 往回第 j 个点（0 = 这一点）放在哪
        let have = (h + 1).min(LIVE_RING);
        let back = move |j: usize| (h + LIVE_RING - j) % LIVE_RING;
        let nm = have.min(MOM);
        let mom = lufs((0..nm).map(|j| self.lv[back(j)] as f64).sum::<f64>() / nm as f64);
        self.act[i] = self.own.is_some_and(|o| mom > o - PAUSE_DB) && mom > -60.0;
        let (mut s, mut c) = (0f64, 0usize);
        for b in (0..have.min(WIN)).map(back).filter(|&b| self.act[b]) {
            s += self.lv[b] as f64;
            c += 1;
        }
        self.st[i] = lufs(if c == 0 { 0.0 } else { s / c as f64 });
        self.hops = h + 1;
    }
}

impl Levels for LiveLevels {
    fn at(&self, _k: usize, sample: i64) -> (f32, f32, bool) {
        if sample < 0 || self.hops == 0 {
            return (0.0, -120.0, false);
        }
        let j = (sample as usize / HOP).min(self.hops - 1);
        if self.hops - j > LIVE_RING {
            return (0.0, -120.0, false);
        }
        let i = j % LIVE_RING;
        (self.lv[i], self.st[i], self.act[i])
    }
}

/// 一个声音的运行状态。
#[derive(Clone)]
struct VoiceState {
    kl: KWeight,
    kr: KWeight,
    e_out: f64,
    e_in: f64,
    transfer: f32,
    gain_db: f32,
    g: f32,
}

/// 平衡前后的统计：几个声音都在出声时，最响和最轻相差多少（0.1 dB 一格，到 30 dB）。
const BINS: usize = 300;

/// 运行时的自动平衡（在合成器里，每个声音的引擎输出之后、相加之前）。不分配内存，可以在音频线程里用。
pub struct Balancer {
    vs: Vec<VoiceState>,
    est: Vec<f32>,
    act: Vec<bool>,
    hist: [[u32; BINS]; 2],
}

impl Balancer {
    pub fn new(n: usize) -> Self {
        let v = VoiceState { kl: KWeight::new(), kr: KWeight::new(), e_out: 0.0, e_in: 0.0, transfer: 0.0, gain_db: 0.0, g: 1.0 };
        Balancer { vs: vec![v; n], est: vec![0.0; n], act: vec![false; n], hist: [[0; BINS]; 2] }
    }

    /// 处理一块：`outs[k]` 是声音 k 的引擎输出（左, 右），前 `m` 个样本就地乘上平衡增益。`in_pos`：这一块对应的输入
    /// 在时间线上的位置（= 输出位置 − 引擎时延）；`delays[k]`：声音 k 比时间线晚多少个样本（左右双人的第 2 个声音），
    /// 它的输入响度按 `in_pos − delays[k]` 取；`strength`：0–1。按 64 个样本一步更新，块长是 64 的整数倍时结果与块长无关。
    pub fn process(&mut self, data: &impl Levels, strength: f32, in_pos: i64, delays: &[usize], outs: &mut [[Vec<f32>; 2]], m: usize) {
        let n = outs.len().min(self.vs.len());
        let m = outs.iter().map(|o| o[0].len().min(o[1].len())).fold(m, usize::min);
        let mut s = 0;
        while s < m {
            let len = (m - s).min(STEP);
            let a_t = (-(len as f32) / (TRANSFER_TAU * FS)).exp() as f64;
            let a_g = (-(len as f32) / (GAIN_TAU * FS)).exp();
            let p = in_pos + (s + len / 2) as i64;
            for k in 0..n {
                let v = &mut self.vs[k];
                let [l, r] = &outs[k];
                let mut e = 0f64;
                for i in s..s + len {
                    let (a, b) = (v.kl.process(l[i]) as f64, v.kr.process(r[i]) as f64);
                    e += a * a + b * b;
                }
                let (lv, st, act) = data.at(k, p - delays.get(k).copied().unwrap_or(0) as i64);
                v.e_out = a_t * v.e_out + (1.0 - a_t) * e / len as f64;
                v.e_in = a_t * v.e_in + (1.0 - a_t) * lv as f64;
                if act && v.e_in > 1e-9 && v.e_out > 1e-12 {
                    v.transfer = (10.0 * (v.e_out / v.e_in).log10()) as f32;
                }
                self.est[k] = st + v.transfer;
                self.act[k] = act;
            }
            let on = self.act[..n].iter().filter(|&&a| a).count();
            if on >= 2 {
                let target = (0..n).filter(|&k| self.act[k]).map(|k| self.est[k]).sum::<f32>() / on as f32;
                let (mut b0, mut b1, mut a0, mut a1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                for k in (0..n).filter(|&k| self.act[k]) {
                    let want = (target - self.est[k]).clamp(-MAX_DB, MAX_DB) * strength;
                    let v = &mut self.vs[k];
                    v.gain_db = a_g * v.gain_db + (1.0 - a_g) * want;
                    let (b, a) = (self.est[k], self.est[k] + v.gain_db);
                    (b0, b1, a0, a1) = (b0.min(b), b1.max(b), a0.min(a), a1.max(a));
                }
                for (h, d) in self.hist.iter_mut().zip([b1 - b0, a1 - a0]) {
                    h[((d * 10.0) as usize).min(BINS - 1)] += 1;
                }
            }
            // 增益在这一步里从上一步的值线性过渡过去，没有台阶
            for k in 0..n {
                let v = &mut self.vs[k];
                let g1 = db_to_lin(v.gain_db);
                let g0 = v.g;
                let [l, r] = &mut outs[k];
                for (j, i) in (s..s + len).enumerate() {
                    let g = g0 + (g1 - g0) * (j + 1) as f32 / len as f32;
                    l[i] *= g;
                    r[i] *= g;
                }
                v.g = g1;
            }
            s += len;
        }
    }

    /// 各声音的平衡增益都已回到 0 dB 附近（±0.05 dB）：强度调到 0 之后，到这时就可以不再处理了。
    pub fn settled(&self) -> bool {
        self.vs.iter().all(|v| v.gain_db.abs() < 0.05 && (v.g - 1.0).abs() < 0.006)
    }

    /// 几个声音都在出声时，最响和最轻相差多少：平衡前、后的（中位数, 90% 分位）dB；还没有数据时为 None。
    pub fn stats(&self) -> Option<((f32, f32), (f32, f32))> {
        let q = |h: &[u32; BINS], p: f64| {
            let tot: u64 = h.iter().map(|&c| c as u64).sum();
            let want = (tot as f64 * p).ceil().max(1.0) as u64;
            let mut acc = 0u64;
            for (i, &c) in h.iter().enumerate() {
                acc += c as u64;
                if acc >= want {
                    return (i as f32 + 0.5) / 10.0;
                }
            }
            BINS as f32 / 10.0
        };
        (self.hist[0].iter().any(|&c| c > 0)).then(|| ((q(&self.hist[0], 0.5), q(&self.hist[0], 0.9)), (q(&self.hist[1], 0.5), q(&self.hist[1], 0.9))))
    }

    /// 声音 k 此刻的平衡增益（dB），测试用。
    #[cfg(test)]
    pub fn gain_db(&self, k: usize) -> f32 {
        self.vs.get(k).map_or(0.0, |v| v.gain_db)
    }
}

/// 给人看的说明。`stats`：[`Balancer::stats`]。
pub fn describe(strength: f32, stats: Option<((f32, f32), (f32, f32))>, english: bool) -> String {
    let pct = (strength * 100.0).round();
    match stats {
        Some(((b0, b1), (a0, a1))) if english => format!("Auto-balance {pct:.0}%: loudness gap between the voices (median / 90%) {b0:.1} / {b1:.1} dB → {a0:.1} / {a1:.1} dB"),
        Some(((b0, b1), (a0, a1))) => format!("自动平衡 {pct:.0}%：几个声音之间的响度差（中位 / 90%）{b0:.1} / {b1:.1} dB → {a0:.1} / {a1:.1} dB"),
        None if english => format!("Auto-balance {pct:.0}%: the voices never sounded at the same time, nothing to balance"),
        None => format!("自动平衡 {pct:.0}%：几个声音没有同时出声的时候，不用平衡"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::testsig::test_voice;

    fn shaped(x: &[f32], segs: &[(f32, f32)]) -> Vec<f32> {
        let mut out = x.to_vec();
        let mut t0 = 0usize;
        for &(sec, db) in segs {
            let t1 = (t0 + (sec * FS) as usize).min(out.len());
            for v in &mut out[t0..t1] {
                *v *= db_to_lin(db);
            }
            t0 = t1;
        }
        out
    }

    /// 两个"声音"：输出 = 输入 × 位置增益（模拟忽远忽近），每次 `block` 个样本送进平衡器；返回平衡器和各秒的响度差。
    fn run(a: &[f32], b: &[f32], pa: &[(f32, f32)], pb: &[(f32, f32)], strength: f32, block: usize) -> (Balancer, Vec<f32>) {
        let data = BalanceData::new(&[&levels(a), &levels(b)], &[0.0, 0.0]);
        let (mut al, mut bl) = (shaped(a, pa), shaped(b, pb));
        let (mut ar, mut br) = (al.clone(), bl.clone());
        let mut bal = Balancer::new(2);
        let n = a.len().min(b.len());
        for o in (0..n).step_by(block) {
            let e = (o + block).min(n);
            feed(&mut bal, &data, strength, o, [&mut al[o..e], &mut ar[o..e], &mut bl[o..e], &mut br[o..e]]);
        }
        // 每秒的响度差（跳过开头 2 秒的起步）
        let sec = FS as usize;
        let e = |x: &[f32], s: usize| 10.0 * (x[s * sec..(s + 1) * sec].iter().map(|v| v * v).sum::<f32>() + 1e-12).log10();
        let diff = (2..n / sec).map(|s| (e(&al, s) - e(&bl, s)).abs()).collect();
        (bal, diff)
    }

    /// 把四段切片抄进平衡器要的缓冲、处理完再抄回来。
    fn feed(bal: &mut Balancer, data: &BalanceData, strength: f32, o: usize, x: [&mut [f32]; 4]) {
        let m = x[0].len();
        let mut bufs = vec![[x[0].to_vec(), x[1].to_vec()], [x[2].to_vec(), x[3].to_vec()]];
        bal.process(data, strength, o as i64, &[], &mut bufs, m);
        let [a, b, c, d] = x;
        a.copy_from_slice(&bufs[0][0]);
        b.copy_from_slice(&bufs[0][1]);
        c.copy_from_slice(&bufs[1][0]);
        d.copy_from_slice(&bufs[1][1]);
    }

    fn median(d: &[f32]) -> f32 {
        let mut x = d.to_vec();
        x.sort_by(f32::total_cmp);
        x[x.len() / 2]
    }

    #[test]
    fn evens_out_position_and_content_differences() {
        let v = test_voice(24.0);
        // A 的内容前半段正常、后半段轻 10 dB；B 的内容不变，但"位置"在 8–16 秒退开（轻 9 dB）
        let a = shaped(&v, &[(12.0, 0.0), (12.0, -10.0)]);
        let pa = [(24.0, 0.0)];
        let pb = [(8.0, 0.0), (8.0, -9.0), (8.0, 0.0)];
        let (_, off) = run(&a, &v, &pa, &pb, 0.0, 4096);
        let (bal, on) = run(&a, &v, &pa, &pb, 1.0, 4096);
        assert!(median(&off) > 5.0, "不平衡时中位差 {:.1} dB", median(&off));
        assert!(median(&on) < 1.5, "平衡后中位差 {:.1} dB（{on:?}）", median(&on));
        // 统计（每 64 个样本一次，前 8 秒两边本来一样响）：90% 分位从约 10 dB 降到 1.5 dB 以内
        let ((_, b9), (_, a9)) = bal.stats().unwrap();
        assert!(b9 > 8.0 && a9 < 1.5, "统计 90% 分位 {b9} → {a9}");
        // 50%：差距约减半
        let (_, half) = run(&a, &v, &pa, &pb, 0.5, 4096);
        assert!((median(&half) - median(&off) / 2.0).abs() < 2.0, "50% 时 {:.1}，不平衡 {:.1}", median(&half), median(&off));
        // 块长不影响结果（文件转换一次 4096、试听一次 64）
        let (_, small) = run(&a, &v, &pa, &pb, 1.0, 64);
        assert!(on.iter().zip(&small).all(|(x, y)| (x - y).abs() < 1e-3), "{on:?}\n{small:?}");
    }

    #[test]
    fn pauses_are_not_lifted() {
        // B 在 6–14 秒是长停顿（无声）：这期间它的增益不往上拉
        let v = test_voice(20.0);
        let b = shaped(&v, &[(6.0, 0.0), (8.0, -90.0), (6.0, 0.0)]);
        let data = BalanceData::new(&[&levels(&v), &levels(&b)], &[0.0, 0.0]);
        let mut bal = Balancer::new(2);
        let (mut al, mut ar, mut bl, mut br) = (v.clone(), v.clone(), b.clone(), b.clone());
        let mut max_gain = 0f32;
        let n = v.len();
        for o in (0..n).step_by(STEP) {
            let e = (o + STEP).min(n);
            feed(&mut bal, &data, 1.0, o, [&mut al[o..e], &mut ar[o..e], &mut bl[o..e], &mut br[o..e]]);
            if o > 7 * FS as usize && o < 13 * FS as usize {
                max_gain = max_gain.max(bal.gain_db(1));
            }
        }
        assert!(max_gain < 1.0, "停顿里增益到了 {max_gain:.1} dB");
    }

    /// 主播模式的左右双人：两个声音是同一个输入 `x`，2 号晚 `d` 个样本，输出 = 输入 × 位置增益，每 64 个样本处理一次。
    /// `live`：输入响度边听边算（[`LiveLevels`]），否则由整段事先算好。返回平衡器（统计）和 `hold` 这段时间里 2 号增益变了多少。
    fn run_duo(x: &[f32], d: usize, pa: &[(f32, f32)], pb: &[(f32, f32)], live: bool, hold: (f32, f32)) -> (Balancer, f32) {
        let y: Vec<f32> = (0..x.len()).map(|i| if i >= d { x[i - d] } else { 0.0 }).collect();
        let lv = levels(x);
        let data = BalanceData::new(&[&lv, &lv], &[0.0, 0.0]);
        let mut live_lv = LiveLevels::new();
        let (a, b) = (shaped(x, pa), shaped(&y, pb));
        let mut bal = Balancer::new(2);
        let (h0, h1) = ((hold.0 * FS) as usize, (hold.1 * FS) as usize);
        let (mut g_lo, mut g_hi) = (f32::MAX, f32::MIN);
        for o in (0..x.len()).step_by(STEP) {
            let e = (o + STEP).min(x.len());
            let mut bufs = vec![[a[o..e].to_vec(), a[o..e].to_vec()], [b[o..e].to_vec(), b[o..e].to_vec()]];
            if live {
                live_lv.feed(&x[o..e]);
                bal.process(&live_lv, 1.0, o as i64, &[0, d], &mut bufs, e - o);
            } else {
                bal.process(&data, 1.0, o as i64, &[0, d], &mut bufs, e - o);
            }
            if o >= h0 && o < h1 {
                (g_lo, g_hi) = (g_lo.min(bal.gain_db(1)), g_hi.max(bal.gain_db(1)));
            }
        }
        (bal, g_hi - g_lo)
    }

    #[test]
    fn live_levels_balance_like_the_precomputed_ones() {
        // 同一个声音，2 号晚 80 ms；它的"位置"在 6–14 秒退开 9 dB，其间 10–13 秒两个都停顿（无声）。
        // 边听边算的输入响度（只能往回看）和整段事先算好的平衡效果差不多；停顿里 2 号的增益不动
        let v = test_voice(24.0);
        let x = shaped(&v, &[(10.0, 0.0), (3.0, -90.0), (11.0, 0.0)]);
        let pa = [(24.0, 0.0)];
        let pb = [(6.0, 0.0), (8.0, -9.0), (10.0, 0.0)];
        let (pre, _) = run_duo(&x, 3840, &pa, &pb, false, (0.0, 0.0));
        let (live, moved) = run_duo(&x, 3840, &pa, &pb, true, (10.6, 13.0));
        let ((b5, b9), (p5, p9)) = pre.stats().unwrap();
        let ((_, _), (l5, l9)) = live.stats().unwrap();
        println!("左右双人的自动平衡：响度差 中位 / 90% {b5:.1} / {b9:.1} dB → 整段事先算 {p5:.1} / {p9:.1}，边听边算 {l5:.1} / {l9:.1}；停顿里 2 号增益变了 {moved:.2} dB");
        assert!(b9 > 8.0 && l5 < 1.0 && l9 < p9 + 1.5, "{b5:.1} / {b9:.1} → 事先 {p5:.1} / {p9:.1}，边听边算 {l5:.1} / {l9:.1}");
        assert!(moved < 0.5, "停顿里 2 号增益变了 {moved:.2} dB");
    }

    #[test]
    fn live_levels_look_back_only_and_detect_pauses() {
        // 边听边算：还没送到的时刻用最近完成的点；停顿开始后 0.4 秒内判为停顿，说话恢复后 0.4 秒内判为出声；
        // 超过记得住的范围（5 秒多）就当无声
        let v = test_voice(12.0);
        let x = shaped(&v, &[(5.0, 0.0), (3.0, -90.0), (4.0, 0.0)]);
        let mut lv = LiveLevels::new();
        assert_eq!(lv.at(0, 0), (0.0, -120.0, false));
        lv.feed(&x);
        let sec = |s: f32| (s * FS) as i64;
        assert_eq!(lv.at(0, sec(12.0) + 10_000), lv.at(1, sec(12.0) - 1));
        assert_eq!(lv.at(0, sec(4.5)), (0.0, -120.0, false), "5 秒多以前的不再记得");
        // 测试人声每 2.2 秒有 0.7 秒的小停顿，9.6 秒在一句话中间
        assert!(lv.at(0, sec(9.6)).2);
        let mut lv = LiveLevels::new();
        let mut act = Vec::new();
        for c in x.chunks(HOP) {
            lv.feed(c);
            act.push(lv.at(0, i64::MAX / 2).2);
        }
        let t = |s: f32| (s * 100.0) as usize;
        assert!(act[t(2.0)..t(5.0)].iter().filter(|&&a| a).count() > t(2.1), "说话时大多判为出声");
        assert!(act[t(5.4)..t(8.0)].iter().all(|&a| !a), "停顿 0.4 秒后都判为停顿");
        assert!(act[t(8.4)..t(9.0)].iter().any(|&a| a), "说话恢复后 0.4 秒内判为出声");
    }
}
