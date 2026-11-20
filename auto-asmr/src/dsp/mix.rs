//! 多音频合成：几段音频各用一个引擎、各在头边的一个位置，输出相加。两种排布：
//!
//! - **各自固定**（默认一个在左耳、一个在右耳）：每个音源的引擎参数 = 总参数，只是"左右移动"换成固定在它自己的
//!   位置（方位角、离头距离），随机种子各不相同：忽远忽近、上下、自然晃动照常，每个声音在自己的位置上各自轻轻地动，
//!   但不会跑到别的音源那边去。
//! - **一起移动**：几个音源都按总参数里的移动方式（穿梭、随机走位、环绕……）走，用同一个随机种子和同一个说话时钟
//!   （任何一个音源在说话时按真实时间走），所以走位一模一样；位置再整体绕头错开 360°/N。两个音源时正好相对：
//!   一个贴左耳时另一个贴右耳，换边时一个从脸前、一个从脑后绕过去，不会撞在一起。
//!
//! 各音源先按输入响度（BS.1770 积分响度）对齐到同样响，再加上各自的音量增减；相加之后整体收尾——
//! 文件转换是两遍精确归一化（与转换单个文件相同），试听是自动响度 + 限制器（与单个文件的实时引擎相同）。

use super::dynamics::{Leveler, Limiter, LoudnessMeter};
use super::engine::{Engine, EngineConfig, PostMode, Telemetry};
use super::filters::ema_coef;
use super::params::{MotionMode, Params};
use super::{db_to_lin, lin_to_db, CTRL, FS, R_MAX, R_MIN};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// 最多同时合成几个音源。试听时每个音源是一个实时运行的引擎（每个约占单核 1%）。
pub const MAX_VOICES: usize = 4;

/// 一个音源的位置和音量。
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Voice {
    /// 方位角（度，顺时针：0 前 90 右 180 后 270 左）。
    pub az: f32,
    /// 到头中心的距离（米）；None = 跟随"贴耳距离"。
    pub dist: Option<f32>,
    /// 音量增减（dB），在自动对齐响度之后再加。
    pub gain_db: f32,
}

/// 位置预设：（中文名, 英文名, 方位角, 到头中心的距离；None = 贴耳距离）。左右耳的方位角与"贴耳角度"的默认值（100°，略靠后）一致。
pub const PRESETS: [(&str, &str, f32, Option<f32>); 8] = [
    ("左耳", "Left ear", 260.0, None),
    ("右耳", "Right ear", 100.0, None),
    ("正前", "Front", 0.0, Some(0.35)),
    ("正后", "Back", 180.0, Some(0.35)),
    ("左前", "Front left", 315.0, Some(0.3)),
    ("右前", "Front right", 45.0, Some(0.3)),
    ("左后", "Back left", 225.0, Some(0.3)),
    ("右后", "Back right", 135.0, Some(0.3)),
];

/// 新音源依次放到哪里（`PRESETS` 的下标）：先左耳、右耳——两个音源时一左一右，离得最远——再正前、正后。
const DEFAULT_ORDER: [usize; MAX_VOICES] = [0, 1, 2, 3];

/// 两个方位角相差多少度（0–180）。
fn ang_diff(a: f32, b: f32) -> f32 {
    ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
}

impl Voice {
    pub fn preset(i: usize) -> Self {
        let (_, _, az, dist) = PRESETS[i];
        Voice { az, dist, gain_db: 0.0 }
    }

    /// 和哪个预设位置相同（不看音量）。
    pub fn preset_index(&self) -> Option<usize> {
        PRESETS.iter().position(|&(_, _, az, d)| {
            ang_diff(self.az, az) < 0.5
                && match (self.dist, d) {
                    (None, None) => true,
                    (Some(a), Some(b)) => (a - b).abs() < 1e-3,
                    _ => false,
                }
        })
    }

    /// 实际到头中心的距离（米）。`ear_dist`：总参数里的贴耳距离。
    pub fn distance(&self, ear_dist: f32) -> f32 {
        self.dist.unwrap_or(ear_dist).clamp(R_MIN, R_MAX)
    }
}

/// 给一个新音源挑位置：按默认顺序取第一个附近（30° 以内）没有别的音源的预设；都有了就取离已有音源最远的预设。
pub fn default_voice(used: &[Voice]) -> Voice {
    let free = |az: f32| used.iter().all(|u| ang_diff(u.az, az) >= 30.0);
    if let Some(&i) = DEFAULT_ORDER.iter().find(|&&i| free(PRESETS[i].2)) {
        return Voice::preset(i);
    }
    let gap = |i: usize| used.iter().map(|u| ang_diff(u.az, PRESETS[i].2)).fold(180.0f32, f32::min);
    Voice::preset((0..PRESETS.len()).max_by(|&a, &b| gap(a).total_cmp(&gap(b))).unwrap_or(0))
}

/// 一个音源的引擎参数：总参数里"左右移动"换成固定在它自己的位置，随机种子按序号错开，
/// 几个声音的忽远忽近、上下、晃动各走各的节奏。第 0 个音源的种子与总参数相同。
pub fn voice_params(base: &Params, v: &Voice, k: usize) -> Params {
    let mut p = base.clone();
    p.mode = MotionMode::Static;
    p.static_az = v.az.rem_euclid(360.0);
    p.ear_dist = v.distance(base.ear_dist);
    p.seed = base.seed.wrapping_add(7919 * k as u64);
    p
}

/// 交给引擎（音频线程）的一份合成配置。
pub struct MixSetup {
    /// 每个音源的引擎配置。
    pub cfgs: Vec<Arc<EngineConfig>>,
    /// 每个音源的线性增益（响度对齐 + 音量增减）。
    pub gains: Vec<f32>,
    /// 每个音源的方位角偏移（度）：一起移动时是 0、360/N、2·360/N……；各自固定时全是 0。
    pub offsets: Vec<f32>,
    /// 一起移动：几个音源共用一个说话时钟、同一套走位，位置按 `offsets` 错开。
    pub follow: bool,
}

impl MixSetup {
    /// 由总配置、各音源的位置和音量、各音源的响度（对齐用）算出合成配置。`follow`：一起移动，否则各自固定。
    /// `prev` 里参数没变的音源配置原样复用。
    pub fn new(base: &Arc<EngineConfig>, voices: &[Voice], lufs: &[Option<f32>], follow: bool, prev: Option<&MixSetup>) -> Self {
        let n = voices.len();
        // 一起移动：每个音源都直接用总配置（同一个移动方式、同一个随机种子）
        let cfgs = if follow { vec![base.clone(); n] } else { voice_configs(base, voices, prev) };
        MixSetup { cfgs, gains: balance_gains(lufs, voices), offsets: follow_offsets(n, follow), follow }
    }
}

/// 各音源的方位角偏移：一起移动时均匀分布在一圈上（两个时相隔 180°，正好相对）；各自固定时全是 0。
pub fn follow_offsets(n: usize, follow: bool) -> Vec<f32> {
    (0..n).map(|k| if follow { 360.0 * k as f32 / n as f32 } else { 0.0 }).collect()
}

/// 每个音源的引擎配置。滤波器等派生数据直接用 `base` 里算好的（位置不影响它们），所以很轻；
/// `prev` 里参数没变的音源原样复用。
pub fn voice_configs(base: &EngineConfig, voices: &[Voice], prev: Option<&MixSetup>) -> Vec<Arc<EngineConfig>> {
    voices
        .iter()
        .enumerate()
        .map(|(k, v)| {
            let p = voice_params(&base.params, v, k);
            match prev.and_then(|s| s.cfgs.get(k)) {
                Some(c) if c.params == p => c.clone(),
                _ => Arc::new(EngineConfig::new(p, Some(base))),
            }
        })
        .collect()
}

/// 单声道音频的积分响度（BS.1770，LUFS）；几乎无声时返回 None。
pub fn loudness(x: &[f32]) -> Option<f32> {
    let mut m = LoudnessMeter::new();
    for &v in x {
        m.push(v, 0.0);
    }
    let l = m.integrated();
    (l > -69.9).then_some(l)
}

/// 各音源的线性增益：先把响度对齐到各音源的平均响度（最多拉 ±30 dB；无声的音源不动），再加各自的音量增减。
pub fn balance_gains(lufs: &[Option<f32>], voices: &[Voice]) -> Vec<f32> {
    let known: Vec<f32> = lufs.iter().flatten().copied().collect();
    let mean = if known.is_empty() { 0.0 } else { known.iter().sum::<f32>() / known.len() as f32 };
    voices
        .iter()
        .enumerate()
        .map(|(k, v)| {
            let align = lufs.get(k).copied().flatten().map_or(0.0, |l| (mean - l).clamp(-30.0, 30.0));
            db_to_lin(align + v.gain_db)
        })
        .collect()
}

/// 每次最多处理多少样本（更长的输入分段处理，音频线程里不分配内存）。
const SCRATCH: usize = 4096;

/// 几个音源的引擎 + 相加 + 收尾（实时：自动响度和限制器；文件：不收尾，留给第二遍精确归一化）。
pub struct Mixer {
    engines: Vec<Engine>,
    /// 目标增益、当前增益（试听中改音量时约 20 ms 平滑过去，不会咔嗒）。
    target: Vec<f32>,
    gains: Vec<f32>,
    post: PostMode,
    /// 一起移动：几个引擎按控制块同步推进，共用说话时钟。`phase`：当前控制块里已经处理了多少样本（各引擎相同）。
    follow: bool,
    phase: usize,
    /// 收尾用的参数（自动响度、目标响度、输出增益、峰值上限）取第 0 个音源的配置——各音源这几项都相同。
    cfg: Arc<EngineConfig>,
    leveler: Leveler,
    limiter: Limiter,
    buf: Vec<[Vec<f32>; 2]>,
    out_pk: f32,
    /// 给界面显示的合成状态：输入取各音源中最大的，输出、限幅、自动响度是相加之后的；位置是第 0 个音源的。
    pub tele: Telemetry,
}

impl Mixer {
    /// `lookahead_s`：分析预读（实时 0，文件约 1 s）。各音源的引擎都不做响度和限幅，相加之后统一做。
    /// 排布方式（各自固定 / 一起移动）和音源数以这里为准，之后不能改（要改就重新开始）。
    pub fn new(setup: &MixSetup, post: PostMode, lookahead_s: f32) -> Self {
        assert!(!setup.cfgs.is_empty(), "合成至少要有一个音源");
        let n = setup.cfgs.len();
        let gains: Vec<f32> = (0..n).map(|k| setup.gains.get(k).copied().unwrap_or(1.0)).collect();
        let mut engines: Vec<Engine> = setup.cfgs.iter().map(|c| Engine::new(c.clone(), PostMode::Offline, lookahead_s)).collect();
        for (e, &o) in engines.iter_mut().zip(&setup.offsets) {
            e.az_offset = o;
        }
        Mixer {
            engines,
            target: gains.clone(),
            gains,
            post,
            follow: setup.follow && n > 1,
            phase: 0,
            cfg: setup.cfgs[0].clone(),
            leveler: Leveler::new(),
            limiter: Limiter::new(2.0, 80.0),
            buf: (0..n).map(|_| [vec![0.0; SCRATCH], vec![0.0; SCRATCH]]).collect(),
            out_pk: 0.0,
            tele: Telemetry::default(),
        }
    }

    pub fn voices(&self) -> usize {
        self.engines.len()
    }

    pub fn engine(&self, k: usize) -> &Engine {
        &self.engines[k]
    }

    pub fn engine_mut(&mut self, k: usize) -> &mut Engine {
        &mut self.engines[k]
    }

    /// 换配置（在各引擎的控制块边界生效）：参数、位置、音量。不分配内存，可以在音频线程里调用；
    /// 音源数、排布方式和方位偏移以构造时为准（`setup` 里的这几项不看）。
    pub fn set_setup(&mut self, setup: &MixSetup) {
        for (e, c) in self.engines.iter_mut().zip(&setup.cfgs) {
            e.set_config(c.clone());
        }
        for (t, &g) in self.target.iter_mut().zip(&setup.gains) {
            *t = g;
        }
        if let Some(c) = setup.cfgs.first() {
            self.cfg = c.clone();
        }
    }

    /// 算法延时（样本）：各引擎相同，实时模式再加限制器的前瞻。
    pub fn latency_samples(&self) -> usize {
        self.engines[0].latency_samples() + if self.post == PostMode::Realtime { self.limiter.delay() } else { 0 }
    }

    /// 每个音源一段等长的单声道输入（`inputs[k]` 对应第 k 个音源），输出相加后的左右声道。
    pub fn process(&mut self, inputs: &[&[f32]], out_l: &mut [f32], out_r: &mut [f32]) {
        self.run(inputs, out_l, out_r, false);
    }

    /// 同 [`Mixer::process`]，但各音源的引擎各在一个线程里算（文件转换用）。每个引擎仍按顺序处理自己的音频，
    /// 相加按固定顺序，所以结果与 `process` 逐位相同。一起移动时几个引擎要逐个控制块同步，仍按顺序算。
    pub fn process_parallel(&mut self, inputs: &[&[f32]], out_l: &mut [f32], out_r: &mut [f32]) {
        self.run(inputs, out_l, out_r, true);
    }

    fn run(&mut self, inputs: &[&[f32]], out_l: &mut [f32], out_r: &mut [f32], par: bool) {
        let n = out_l.len().min(out_r.len());
        let mut o = 0;
        while o < n {
            let m = (n - o).min(SCRATCH);
            self.block(inputs, o, m, &mut out_l[o..o + m], &mut out_r[o..o + m], par);
            o += m;
        }
    }

    fn block(&mut self, inputs: &[&[f32]], o: usize, m: usize, out_l: &mut [f32], out_r: &mut [f32], par: bool) {
        static SILENCE: [f32; SCRATCH] = [0.0; SCRATCH];
        let input = |k: usize| inputs.get(k).and_then(|x| x.get(o..o + m)).unwrap_or(&SILENCE[..m]);
        if self.follow {
            // 一起移动：几个引擎按控制块同步推进。每块开始前取各音源这一块的说话活动的最大值交给它们，
            // 说话时钟完全一致（任何一个音源在说话就按真实时间走），同一个随机种子下走位也就一模一样
            let mut i = 0;
            while i < m {
                if self.phase == 0 {
                    let act = self.engines.iter().map(|e| e.upcoming_act()).fold(0.0f32, f32::max);
                    for e in self.engines.iter_mut() {
                        e.clock_act = Some(act);
                    }
                }
                let step = (m - i).min(CTRL - self.phase);
                for (k, (eng, [bl, br])) in self.engines.iter_mut().zip(self.buf.iter_mut()).enumerate() {
                    eng.process(&input(k)[i..i + step], &mut bl[i..i + step], &mut br[i..i + step]);
                }
                self.phase = (self.phase + step) % CTRL;
                i += step;
            }
        } else if par && self.engines.len() > 1 {
            std::thread::scope(|s| {
                for (k, (eng, [bl, br])) in self.engines.iter_mut().zip(self.buf.iter_mut()).enumerate() {
                    let x = input(k);
                    s.spawn(move || {
                        crate::dsp::flush_denormals();
                        eng.process(x, &mut bl[..m], &mut br[..m]);
                    });
                }
            });
        } else {
            for (k, (eng, [bl, br])) in self.engines.iter_mut().zip(self.buf.iter_mut()).enumerate() {
                eng.process(input(k), &mut bl[..m], &mut br[..m]);
            }
        }
        out_l.fill(0.0);
        out_r.fill(0.0);
        let kg = ema_coef(0.02, 1.0 / FS);
        for (k, [bl, br]) in self.buf.iter().enumerate() {
            let tgt = self.target[k];
            let mut g = self.gains[k];
            if g == tgt {
                for i in 0..m {
                    out_l[i] += g * bl[i];
                    out_r[i] += g * br[i];
                }
            } else {
                for i in 0..m {
                    g += kg * (tgt - g);
                    out_l[i] += g * bl[i];
                    out_r[i] += g * br[i];
                }
                if (g - tgt).abs() <= 1e-4 * tgt.abs() {
                    g = tgt;
                }
                self.gains[k] = g;
            }
        }
        if self.post == PostMode::Realtime {
            // 与单个音源的实时引擎相同的收尾：有任何一个音源在说话时测响度
            let speaking = self.engines.iter().any(|e| e.tele.act > 0.5);
            let p = &self.cfg.params;
            let gain = db_to_lin(p.out_gain_db);
            let ceiling = db_to_lin(p.ceiling_db - 0.1);
            for i in 0..m {
                let (l, r) = (out_l[i], out_r[i]);
                let lg = if p.auto_level { self.leveler.process(l, r, speaking, p.target_lufs) } else { 1.0 };
                let (a, b) = self.limiter.process(l * lg * gain, r * lg * gain, ceiling);
                out_l[i] = a;
                out_r[i] = b;
            }
        }
        let pk = out_l.iter().chain(out_r.iter()).fold(0f32, |a, &v| a.max(v.abs()));
        self.out_pk = (self.out_pk * 0.97f32.powf(m as f32 / CTRL as f32)).max(pk);
        self.update_tele();
    }

    fn update_tele(&mut self) {
        let mut t = self.engines[0].tele;
        for e in &self.engines[1..] {
            let u = &e.tele;
            t.act = t.act.max(u.act);
            t.in_db = t.in_db.max(u.in_db);
            t.n_moves += u.n_moves;
            t.n_sways += u.n_sways;
            t.n_tops += u.n_tops;
            t.n_updowns += u.n_updowns;
            t.n_strokes += u.n_strokes;
        }
        t.out_db = lin_to_db(self.out_pk);
        let rt = self.post == PostMode::Realtime;
        t.gr_db = if rt { self.limiter.gr_db } else { 0.0 };
        t.level_gain_db = if rt { self.leveler.gain_db } else { 0.0 };
        self.tele = t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::test_voice;

    fn setup(voices: &[Voice], gains: Vec<f32>) -> MixSetup {
        let base = EngineConfig::new(Params::default(), None);
        MixSetup { cfgs: voice_configs(&base, voices, None), gains, offsets: vec![0.0; voices.len()], follow: false }
    }

    fn energy(v: &[f32]) -> f64 {
        v.iter().map(|x| (x * x) as f64).sum::<f64>() + 1e-30
    }

    /// 前一半只有 A 在说、后一半只有 B 在说：前一半声音在左耳，后一半在右耳。
    fn split_voices(seconds: f32) -> (Vec<f32>, Vec<f32>, usize) {
        let x = test_voice(seconds);
        let half = x.len() / 2;
        let a = x.iter().enumerate().map(|(i, &v)| if i < half { v } else { 0.0 }).collect();
        let b = x.iter().enumerate().map(|(i, &v)| if i >= half { v } else { 0.0 }).collect();
        (a, b, half)
    }

    #[test]
    fn two_voices_stay_on_their_own_side() {
        let (a, b, half) = split_voices(12.0);
        let n = a.len();
        let mut mx = Mixer::new(&setup(&[Voice::preset(0), Voice::preset(1)], vec![1.0, 1.0]), PostMode::Offline, 0.0);
        let (mut l, mut r) = (vec![0f32; n], vec![0f32; n]);
        mx.process(&[&a, &b], &mut l, &mut r);
        // 跳过两段交界处（引擎的时延和尾音）各 0.2 s
        let g = (0.2 * FS) as usize;
        let first = 10.0 * (energy(&l[g..half]) / energy(&r[g..half])).log10();
        let second = 10.0 * (energy(&r[half + g..]) / energy(&l[half + g..])).log10();
        assert!(first > 20.0, "A 应该在左耳：左比右高 {first:.1} dB");
        assert!(second > 20.0, "B 应该在右耳：右比左高 {second:.1} dB");
        // 整段都在说话的那个音源一直留在自己那边：位置始终在左半边 / 右半边
        assert!(mx.engine(0).tele.az > 180.0 && mx.engine(1).tele.az < 180.0);
    }

    #[test]
    fn parallel_and_sequential_are_bit_identical() {
        let (a, b, _) = split_voices(6.0);
        let voices = [Voice::preset(0), Voice::preset(1), Voice::preset(2)];
        let c: Vec<f32> = a.iter().zip(&b).map(|(x, y)| 0.5 * (x + y)).collect();
        let n = a.len();
        let run = |par: bool| {
            let mut mx = Mixer::new(&setup(&voices, vec![1.0, 0.7, 0.5]), PostMode::Offline, 1.0);
            let (mut l, mut r) = (vec![0f32; n], vec![0f32; n]);
            for o in (0..n).step_by(4096) {
                let e = (o + 4096).min(n);
                let ins = [&a[o..e], &b[o..e], &c[o..e]];
                if par {
                    mx.process_parallel(&ins, &mut l[o..e], &mut r[o..e]);
                } else {
                    mx.process(&ins, &mut l[o..e], &mut r[o..e]);
                }
            }
            (l, r)
        };
        let (s, p) = (run(false), run(true));
        assert!(s.0.iter().zip(&p.0).chain(s.1.iter().zip(&p.1)).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn loudness_alignment_and_gain_offsets() {
        let x = test_voice(8.0);
        let quiet: Vec<f32> = x.iter().map(|v| v * 0.25).collect(); // 低 12 dB
        let (la, lb) = (loudness(&x).unwrap(), loudness(&quiet).unwrap());
        assert!((la - lb - 12.04).abs() < 0.05, "{la} {lb}");
        let v = [Voice::preset(0), Voice { gain_db: -3.0, ..Voice::preset(1) }];
        let g = balance_gains(&[Some(la), Some(lb)], &v);
        // 对齐之后 B 比 A 低 3 dB（它自己的音量增减），整体在两者平均响度附近
        let (oa, ob) = (la + lin_to_db(g[0]), lb + lin_to_db(g[1]));
        assert!((oa - ob - 3.0).abs() < 0.01, "{oa} {ob}");
        assert!((oa - 0.5 * (la + lb)).abs() < 0.01);
        // 无声的音源不拉高，只加它自己的音量增减
        assert_eq!(loudness(&vec![0.0; 48_000 * 2]), None);
        let g = balance_gains(&[Some(la), None], &[Voice::preset(0), Voice { gain_db: 6.0, ..Voice::preset(1) }]);
        assert!((g[0] - 1.0).abs() < 1e-6 && (lin_to_db(g[1]) - 6.0).abs() < 1e-3);
    }

    #[test]
    fn default_positions_are_far_apart() {
        let mut used = Vec::new();
        for _ in 0..MAX_VOICES {
            let v = default_voice(&used);
            used.push(v);
        }
        // 两个音源：一左一右；四个：左、右、前、后，两两至少相隔 80°
        assert_eq!(used[0].preset_index(), Some(0));
        assert_eq!(used[1].preset_index(), Some(1));
        for i in 0..used.len() {
            for j in 0..i {
                assert!(ang_diff(used[i].az, used[j].az) >= 80.0, "{:?} {:?}", used[i], used[j]);
            }
        }
        // 第一个音源被挪到正前方后，新来的补到空出来的左耳
        let moved = [Voice::preset(2), Voice::preset(1)];
        assert_eq!(default_voice(&moved).preset_index(), Some(0));
        // 四个方向都占了：挑离已有音源最远的预设，不会叠在某个音源上
        let extra = default_voice(&used);
        assert!(used.iter().all(|u| ang_diff(u.az, extra.az) >= 40.0));
    }

    #[test]
    fn voices_keep_their_own_position_and_share_filters() {
        let base = EngineConfig::new(Params::default(), None);
        let voices = [Voice::preset(0), Voice { dist: Some(0.3), ..Voice::preset(5) }];
        let cfgs = voice_configs(&base, &voices, None);
        assert_eq!(cfgs[0].params.mode, MotionMode::Static);
        assert_eq!((cfgs[0].params.static_az, cfgs[0].params.ear_dist), (260.0, base.params.ear_dist));
        assert_eq!((cfgs[1].params.static_az, cfgs[1].params.ear_dist), (45.0, 0.3));
        assert_ne!(cfgs[0].params.seed, cfgs[1].params.seed);
        // 滤波器直接共用总参数里算好的，不重算
        assert!(Arc::ptr_eq(cfgs[1].derived.ear_eq.as_ref().unwrap(), base.derived.ear_eq.as_ref().unwrap()));
        // 参数没变的音源原样复用
        let prev = MixSetup { cfgs: cfgs.clone(), gains: vec![1.0; 2], offsets: vec![0.0; 2], follow: false };
        let again = voice_configs(&base, &voices, Some(&prev));
        assert!(Arc::ptr_eq(&again[0], &cfgs[0]) && Arc::ptr_eq(&again[1], &cfgs[1]));
    }

    #[test]
    fn follow_mode_moves_together_and_stays_opposite() {
        // 一起移动（默认是穿梭）：A 一直在说，B 只在后半段说。两个音源始终在头的两侧相对，而且确实在两耳之间换边
        let a = test_voice(30.0);
        let n = a.len();
        let half = n / 2;
        let b: Vec<f32> = a.iter().enumerate().map(|(i, &v)| if i >= half { 0.7 * v } else { 0.0 }).collect();
        let base = Arc::new(EngineConfig::new(Params::default(), None));
        let st = MixSetup::new(&base, &[Voice::preset(0), Voice::preset(1)], &[None, None], true, None);
        assert_eq!(st.offsets, vec![0.0, 180.0]);
        assert!(Arc::ptr_eq(&st.cfgs[0], &base) && Arc::ptr_eq(&st.cfgs[1], &base));
        let mut mx = Mixer::new(&st, PostMode::Offline, 0.0);
        // 对照：A 单独过一个同样参数的引擎（前半段只有 A 在说，共同的说话时钟就是 A 自己的）
        let mut solo = Engine::new(base.clone(), PostMode::Offline, 0.0);
        let (mut l, mut r) = (vec![0f32; 1000], vec![0f32; 1000]);
        let (mut sl, mut sr) = (vec![0f32; 1000], vec![0f32; 1000]);
        let (mut min_gap, mut sides, mut max_dev) = (180.0f32, [0usize; 2], 0f32);
        // 每次 1000 个样本（不是 64 的整数倍），顺带检查控制块对齐
        for o in (0..n).step_by(1000) {
            let e = (o + 1000).min(n);
            mx.process(&[&a[o..e], &b[o..e]], &mut l[..e - o], &mut r[..e - o]);
            let (p, q) = (mx.engine(0).tele, mx.engine(1).tele);
            min_gap = min_gap.min(ang_diff(p.az, q.az));
            let s = p.az.to_radians().sin();
            if s < -0.9 {
                sides[0] += 1;
            } else if s > 0.9 {
                sides[1] += 1;
            }
            if e <= half {
                solo.process(&a[o..e], &mut sl[..e - o], &mut sr[..e - o]);
                max_dev = max_dev.max(ang_diff(p.az, solo.tele.az)).max((p.r - solo.tele.r).abs() * 100.0);
            }
        }
        assert!(min_gap > 150.0, "两个音源最近时只相隔 {min_gap:.0}°");
        assert!(sides[0] > 100 && sides[1] > 100, "A 应在两耳之间来回：{sides:?}");
        assert!(mx.engine(0).tele.n_moves >= 4);
        assert_eq!(mx.engine(0).tele.n_moves, mx.engine(1).tele.n_moves, "同一套走位");
        // 只有一个人在说话时，它的走位与单独转换完全相同
        assert!(max_dev < 1e-3, "与单独转换的走位相差 {max_dev}");
    }

    #[test]
    fn follow_mode_spreads_more_voices_evenly() {
        let base = Arc::new(EngineConfig::new(Params { mode: MotionMode::Orbit, ..Params::default() }, None));
        let voices: Vec<Voice> = (0..4).map(Voice::preset).collect();
        let st = MixSetup::new(&base, &voices, &[None; 4], true, None);
        assert_eq!(st.offsets, vec![0.0, 90.0, 180.0, 270.0]);
        let x = test_voice(10.0);
        let mut mx = Mixer::new(&st, PostMode::Offline, 0.0);
        let (mut l, mut r) = (vec![0f32; x.len()], vec![0f32; x.len()]);
        mx.process(&[&x, &x, &x, &x], &mut l, &mut r);
        // 环绕：四个音源相邻两个相隔约 90°
        for k in 0..4 {
            let gap = ang_diff(mx.engine(k).tele.az, mx.engine((k + 1) % 4).tele.az);
            assert!((gap - 90.0).abs() < 15.0, "音源 {k} 与下一个相隔 {gap:.0}°");
        }
        // 各自固定时没有偏移
        assert_eq!(MixSetup::new(&base, &voices, &[None; 4], false, None).offsets, vec![0.0; 4]);
    }
}
