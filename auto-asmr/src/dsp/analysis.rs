//! 语音分析（因果，可带"预读"）：说话活动、此刻音量、给"声音大就往后让"用的包络。
//!
//! 实时模式没有预读，只能听到声音之后再反应；文件模式把音频延后约 1 秒，分析走在前面，
//! 于是可以像 Python 版那样在响的字到来之前就开始往后让。

use super::filters::{ema_coef, Biquad, BUTTER_Q};
use super::{CTRL, CTRL_DT, FS};

#[derive(Clone, Copy, Debug, Default)]
pub struct Features {
    /// 说话活动 0–1（与音频对齐）。
    pub act: f32,
    /// 此刻音量，dB，相对说话时的中位数，停顿里保持（与音频对齐）。
    pub loud_db: f32,
    /// "声音大就往后让"的包络，dB：提前让开、说话时才慢慢回来，已去掉中位。
    pub follow_db: f32,
}

const ATT_DB_S: f32 = 6.0 / 0.8; // 提前让开：每 0.8 s 6 dB
const REL_DB_S: f32 = 6.0 / 2.5; // 凑回来：每 2.5 s 6 dB，只在说话时进行
const PAVG_S: f32 = 0.45;

pub struct Analyzer {
    hp: Biquad,
    lp: Biquad,
    acc: f32,
    // 最近 50 ms 的块能量（矩形窗，与 Python 版的 50 ms 帧一致；指数平滑会在句尾拖出尾巴）
    eb: Vec<f32>,
    eb_i: usize,
    eb_sum: f64,
    e_s: f32,
    nf_db: f32,
    sp_db: f32,
    fixed: Option<(f32, f32)>,
    started: bool,
    hang: f32,
    ref_db: f32,
    ref_n: f32,
    // 0.45 s 功率平均（只算说话的块）
    pav_p: Vec<f32>,
    pav_a: Vec<bool>,
    pav_i: usize,
    pav_sum: f64,
    pav_cnt: usize,
    dev_hold: f32,
    loud_hold: f32,
    loud_lp: Biquad,
    // 预读环（K+1 个控制块）：下标 pos 是最新的（走在音频前面 K 块）
    k: usize,
    ring_raw: Vec<bool>,
    ring_loud: Vec<f32>,
    ring_dev: Vec<f32>,
    pos: usize,
    last_active: i64,
    blk: i64,
    act_s: f32,
    env: f32,
    med: f32,
    env_lp: Biquad,
    env_primed: bool,
    out: Features,
}

impl Analyzer {
    /// `lookahead_s`：分析比音频提前多少秒（实时模式为 0）。
    pub fn new(lookahead_s: f32) -> Self {
        let k = (lookahead_s / CTRL_DT).round() as usize;
        let np = (PAVG_S / CTRL_DT).round() as usize;
        let fc = FS / CTRL as f32;
        Analyzer {
            hp: Biquad::highpass(150.0, BUTTER_Q),
            lp: Biquad::lowpass(5000.0, BUTTER_Q),
            acc: 0.0,
            eb: vec![0.0; (0.05 / CTRL_DT).round() as usize],
            eb_i: 0,
            eb_sum: 0.0,
            e_s: 0.0,
            nf_db: -75.0,
            sp_db: -45.0,
            fixed: None,
            started: false,
            hang: 0.0,
            ref_db: -30.0,
            ref_n: 0.0,
            pav_p: vec![0.0; np],
            pav_a: vec![false; np],
            pav_i: 0,
            pav_sum: 0.0,
            pav_cnt: 0,
            dev_hold: 0.0,
            loud_hold: 0.0,
            loud_lp: Biquad::lowpass_at(2.0, BUTTER_Q, fc),
            k,
            ring_raw: vec![false; k + 1],
            ring_loud: vec![0.0; k + 1],
            ring_dev: vec![0.0; k + 1],
            pos: 0,
            last_active: -1_000_000,
            blk: 0,
            act_s: 0.0,
            env: 0.0,
            med: 1.0,
            env_lp: Biquad::lowpass_at(0.8, BUTTER_Q, fc),
            env_primed: false,
            out: Features::default(),
        }
    }

    /// 预读对应的音频延时（样本）。
    pub fn lookahead_samples(&self) -> usize {
        self.k * CTRL
    }

    /// 文件模式：用整段音频统计出的说话阈值和中位音量（与 Python 版一致），不再自适应。
    pub fn set_fixed(&mut self, thr_db: f32, median_db: f32) {
        self.fixed = Some((thr_db, median_db));
        self.ref_db = median_db;
    }

    #[inline]
    pub fn push(&mut self, x: f32) {
        let v = self.lp.process(self.hp.process(x));
        self.acc += v * v;
    }

    pub fn features(&self) -> Features {
        self.out
    }

    /// 每个控制块结束时调用一次。
    pub fn end_block(&mut self) {
        let e_blk = self.acc / CTRL as f32;
        self.acc = 0.0;
        self.eb_sum += (e_blk - self.eb[self.eb_i]) as f64;
        self.eb[self.eb_i] = e_blk;
        self.eb_i = (self.eb_i + 1) % self.eb.len();
        self.e_s = (self.eb_sum.max(0.0) / self.eb.len() as f64) as f32;
        let e_db = 10.0 * (self.e_s + 1e-12).log10();
        let thr = match self.fixed {
            Some((t, _)) => t,
            None if !self.started && (self.blk as usize) + 1 < self.eb.len() => {
                // 50 ms 的能量窗还没填满，读数偏低：先不判断。否则底噪会被估得太低，
                // 麦克风的底噪在开头十来秒里都被当成"在说话"
                f32::INFINITY
            }
            None => {
                if !self.started {
                    // 还没听到人声时，先假定说话峰值在 −30 dBFS 左右：比它低 30 dB 以上的轻微声音不算说话
                    self.nf_db = e_db.min(-40.0);
                    self.sp_db = (e_db + 10.0).max(-30.0);
                    self.started = true;
                }
                // 底噪：往下跟得快，往上每秒只涨 1.5 dB；说话电平：往上跟得快，往下每秒只掉 0.5 dB
                if e_db < self.nf_db {
                    self.nf_db += 0.1 * (e_db - self.nf_db);
                } else {
                    self.nf_db += 1.5 * CTRL_DT;
                }
                if e_db > self.sp_db {
                    self.sp_db += ema_coef(0.03, CTRL_DT) * (e_db - self.sp_db);
                } else {
                    self.sp_db -= 0.5 * CTRL_DT;
                }
                self.sp_db = self.sp_db.max(self.nf_db + 12.0);
                self.nf_db = self.nf_db.min(self.sp_db - 12.0);
                (self.sp_db - 30.0).max(self.nf_db + 6.0)
            }
        };
        let raw = e_db > thr;

        // 说话时的中位音量（文件模式直接给定）
        if raw && self.fixed.is_none() {
            if self.ref_n == 0.0 {
                self.ref_db = e_db;
            }
            self.ref_n += CTRL_DT;
            let rate = if self.ref_n < 3.0 { 4.0 } else { 0.5 }; // dB/s
            self.ref_db += rate * CTRL_DT * (e_db - self.ref_db).signum();
        }

        // 0.45 s 功率平均
        let n = self.pav_p.len();
        if self.pav_a[self.pav_i] {
            self.pav_sum -= self.pav_p[self.pav_i] as f64;
            self.pav_cnt -= 1;
        }
        self.pav_p[self.pav_i] = self.e_s;
        self.pav_a[self.pav_i] = raw;
        if raw {
            self.pav_sum += self.e_s as f64;
            self.pav_cnt += 1;
        }
        self.pav_i = (self.pav_i + 1) % n;
        if raw {
            self.loud_hold = e_db - self.ref_db;
            if self.pav_cnt > 0 {
                self.dev_hold = 10.0 * ((self.pav_sum.max(0.0) / self.pav_cnt as f64) as f32 + 1e-12).log10() - self.ref_db;
            }
        }
        let loud = self.loud_lp.process(self.loud_hold);

        // 写进预读环；有预读时把不超过 0.3 s 的词间空隙补成"还在说"
        self.pos = (self.pos + 1) % (self.k + 1);
        self.ring_raw[self.pos] = raw;
        self.ring_loud[self.pos] = loud;
        self.ring_dev[self.pos] = self.dev_hold;
        if raw {
            let gap = (self.blk - self.last_active - 1) as usize;
            if gap > 0 && (gap as f32) * CTRL_DT <= 0.3 && gap <= self.k {
                for g in 1..=gap {
                    let i = (self.pos + self.k + 1 - g) % (self.k + 1);
                    self.ring_raw[i] = true;
                }
            }
            self.last_active = self.blk;
        }
        self.blk += 1;

        // 与音频对齐的读位置：K 块之前写的那一格
        let rd = (self.pos + 1) % (self.k + 1);
        let speaking = self.ring_raw[rd];
        if speaking {
            self.hang = if self.k == 0 { 0.08 } else { 0.0 };
        } else {
            self.hang -= CTRL_DT;
        }
        let target = if speaking || self.hang > 0.0 { 1.0 } else { 0.0 };
        let tau = if target > self.act_s { 0.03 } else { 0.12 };
        self.act_s += ema_coef(tau, CTRL_DT) * (target - self.act_s);

        // 提前让开：往后看 K 块，越远的未来折扣越多；没有预读时就是当前值
        let mut a_env = f32::MIN;
        let step = ATT_DB_S * CTRL_DT;
        let mut i = rd;
        for j in 0..=self.k {
            let v = self.ring_dev[i] - step * j as f32;
            if v > a_env {
                a_env = v;
            }
            i = (i + 1) % (self.k + 1);
        }
        let rel = if speaking { REL_DB_S * CTRL_DT } else { 0.0 };
        self.env = a_env.max(self.env - rel);
        if speaking {
            self.med += 0.4 * CTRL_DT * (self.env - self.med).signum();
        }
        let centered = self.env - self.med;
        if !self.env_primed {
            self.env_lp.prime(centered);
            self.env_primed = true;
        }
        self.out = Features {
            act: self.act_s,
            loud_db: self.ring_loud[rd],
            follow_db: self.env_lp.process(centered),
        };
    }
}

/// 文件模式的预扫描：与 Python 版 `speech_activity` 相同的统计——语音频段 50 ms 帧能量，
/// 阈值 = max(97 分位 − 30 dB, 20 分位 + 6 dB)，再取说话帧的中位数。返回 (阈值 dB, 中位 dB)。
pub fn prescan(x: &[f32]) -> Option<(f32, f32)> {
    let hop = (0.05 * FS) as usize;
    let n = x.len() / hop;
    if n < 20 {
        return None;
    }
    let mut hp = Biquad::highpass(150.0, BUTTER_Q);
    let mut lp = Biquad::lowpass(5000.0, BUTTER_Q);
    let mut e = Vec::with_capacity(n);
    for f in 0..n {
        let mut s = 0f64;
        for &v in &x[f * hop..(f + 1) * hop] {
            let y = lp.process(hp.process(v));
            s += (y as f64) * (y as f64);
        }
        e.push(10.0 * ((s / hop as f64) as f32 + 1e-12).log10());
    }
    let mut sorted = e.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let pct = |q: f32| sorted[((q * (n - 1) as f32).round() as usize).min(n - 1)];
    let thr = (pct(0.97) - 30.0).max(pct(0.20) + 6.0);
    let mut act: Vec<f32> = e.iter().cloned().filter(|&v| v > thr).collect();
    if act.len() < 10 {
        return None;
    }
    act.sort_by(|a, b| a.total_cmp(b));
    Some((thr, act[act.len() / 2]))
}
