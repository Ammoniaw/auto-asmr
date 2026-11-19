//! 引擎：把人声预处理、语音分析、运动、直达声、房间、回放补偿、响度和限制器串起来，逐块流式处理。
//! 实时模式和文件模式用的是同一个引擎；文件模式多了约 1 秒的"预读"，并把响度归一化留到第二遍做。

use super::analysis::{Analyzer, Features};
use super::direct::{DirectRenderer, BASE_DELAY};
use super::dynamics::{Leveler, Limiter};
use super::filters::{Biquad, Fir, OnePoleLp, BUTTER_Q};
use super::motion::{clock_step, Motion, Pose};
use super::params::{Params, Playback};
use super::room::{conv_energy, Room};
use super::tables::{tables, Hrir, Tables};
use super::voice::{ear_eq_fir, soft_voice_fir, DeEsser, Denoiser, Gate, DENOISE_DELAY, EAR_EQ_TAPS};
use super::{db_to_lin, lin_to_db, C, CTRL, CTRL_DT, EAR_R, FS, NTAP, R_MAX, R_MIN};
use std::sync::Arc;

/// 直达声比 Python 版多出来的固定时延（样本）：分数延时插值器要"往后看"的那一半。
const ROOM_DELAY: usize = BASE_DELAY as usize - 4;

/// 真实房间：贴耳时远耳听到的房间声比近耳直达声低多少 dB（KU100 实测 −24…−35 dB）。
const REAL_ROOM_DB: f32 = -28.0;

/// 由参数算出来的滤波器等"重"数据，在界面线程里算好再交给音频线程。
pub struct Derived {
    pub ear_eq: Option<Arc<Vec<f32>>>,
    pub playback: Option<Arc<Vec<f32>>>,
    pub soft: Option<Arc<Vec<f32>>>,
    pub room_gain: f32,
    pub pref: f32,
}

/// 参数 + 派生数据的一份不可变快照。
pub struct EngineConfig {
    pub params: Params,
    pub derived: Derived,
}

impl EngineConfig {
    /// 根据参数生成配置；`prev` 里没变的部分直接复用。
    pub fn new(params: Params, prev: Option<&EngineConfig>) -> Self {
        let tab = tables();
        let pp = prev.map(|c| &c.params);
        let pd = prev.map(|c| &c.derived);
        let same = |f: &dyn Fn(&Params) -> bool| pp.map_or(false, f);

        let ear_eq = if same(&|q| q.presence_db == params.presence_db && q.dark_db == params.dark_db && q.ku100_eq == params.ku100_eq) {
            pd.unwrap().ear_eq.clone()
        } else {
            ear_eq_fir(params.presence_db, params.dark_db, params.ku100_eq).map(Arc::new)
        };
        let soft = if same(&|q| q.soft_voice == params.soft_voice) {
            pd.unwrap().soft.clone()
        } else if params.soft_voice > 0.0 {
            Some(Arc::new(soft_voice_fir(params.soft_voice)))
        } else {
            None
        };
        let same_pb = same(&|q| q.playback == params.playback);
        let playback = if same_pb {
            pd.unwrap().playback.clone()
        } else {
            match &params.playback {
                Playback::Diffuse => Some(Arc::new(tab.cdfc.clone())),
                Playback::Native => None,
                Playback::Headphone(name) => tab.headphones.iter().find(|(n, _)| n == name).map(|(_, h)| Arc::new(h.clone())).or_else(|| Some(Arc::new(tab.cdfc.clone()))),
            }
        };
        let room_gain = if same_pb && same(&|q| q.wet_db == params.wet_db) {
            pd.unwrap().room_gain
        } else {
            // 贴耳（右耳、方位 100°）时近耳直达声的能量，与扇区 3 远耳（左耳）房间声的能量之比定标
            let mut h = Hrir::default();
            tab.lookup(260.0, EAR_R, &mut h);
            let pb = playback.as_ref().map(|v| v.as_slice());
            let e_near = conv_energy(&h.h[1], pb, usize::MAX);
            let e_room = conv_energy(tab.room_ir(3, 0), pb, tab.room_len);
            ((e_near / e_room.max(1e-30)) as f32 * 10f32.powf((REAL_ROOM_DB + params.wet_db) / 10.0)).sqrt()
        };
        let pref = tab.lookup_power(270.0, 0.14);
        EngineConfig { params, derived: Derived { ear_eq, playback, soft, room_gain, pref } }
    }
}

/// 给界面显示用的实时状态。
#[derive(Clone, Copy, Debug, Default)]
pub struct Telemetry {
    pub az: f32,
    pub r: f32,
    pub closeness: f32,
    pub act: f32,
    pub in_db: f32,
    pub out_db: f32,
    pub gr_db: f32,
    pub level_gain_db: f32,
    pub n_moves: u32,
    pub n_sways: u32,
}

/// 输出端怎么收尾。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PostMode {
    /// 实时：自动响度 + 限制器都在引擎里做。
    Realtime,
    /// 文件：输出不做响度和限幅，留给第二遍精确归一化。
    Offline,
}

/// 贴耳时近耳低频提升、远耳低频压低：4 阶 Linkwitz–Riley 分频，两耳相同的全通相位，ITD 不受影响。
struct EarLf {
    lp: [Biquad; 2],
    hp: [Biquad; 2],
}

impl EarLf {
    fn new() -> Self {
        EarLf { lp: [Biquad::lowpass(320.0, BUTTER_Q); 2], hp: [Biquad::highpass(320.0, BUTTER_Q); 2] }
    }
    #[inline]
    fn process(&mut self, x: f32, g: f32) -> f32 {
        let lo = self.lp[0].process(x);
        let lo = self.lp[1].process(lo);
        let hi = self.hp[0].process(x);
        let hi = self.hp[1].process(hi);
        hi + g * lo
    }
}

/// 嘴贴近某一侧耳朵的程度（0–1）：距离越近、越偏向正侧方越大；转到脸前 / 脑后时为 0。
#[inline]
pub fn closeness(az: f32, r: f32) -> f32 {
    let lat = az.to_radians().sin().abs();
    ((0.2 - r) / (0.2 - EAR_R)).clamp(0.0, 1.0) * lat * lat
}

pub struct Engine {
    tab: &'static Tables,
    cfg: Arc<EngineConfig>,
    post: PostMode,
    in_hp: Biquad,
    gate: Gate,
    denoiser: Denoiser,
    deess: DeEsser,
    soft: Option<Fir>,
    soft_src: Option<Arc<Vec<f32>>>,
    analyzer: Analyzer,
    la_buf: Vec<f32>,
    la_pos: usize,
    warm_lp: OnePoleLp,
    direct: DirectRenderer,
    room: Room,
    lf: [EarLf; 2],
    pb: Option<[Fir; 2]>,
    pb_src: Option<Arc<Vec<f32>>>,
    out_hp: [Biquad; 2],
    leveler: Leveler,
    limiter: Limiter,
    motion: Motion,
    tm: f64,
    phase: usize,
    warm: (f32, f32),
    lfg: [(f32, f32); 2],
    feats: Features,
    pose: Pose,
    hr: Hrir,
    xa: [f32; CTRL],
    xr: [f32; CTRL],
    room_dl: [f32; ROOM_DELAY],
    room_dl_pos: usize,
    xd: [f32; CTRL],
    dl: [f32; CTRL],
    dr: [f32; CTRL],
    in_pk: f32,
    out_pk: f32,
    primed: bool,
    pub tele: Telemetry,
    /// 统计：已处理的控制块数、其中在说话的块数。
    pub n_blocks: u64,
    pub n_speech: u64,
    /// 打开后记录每个控制块的 [时间 s, 方位角, 距离, 不含忽远忽近的距离, 说话活动]（导出轨迹用）。
    pub trace: Option<Vec<[f32; 5]>>,
    amp_ms: f32,
}

impl Engine {
    /// `lookahead_s`：分析比音频提前多少秒（实时为 0；文件模式约 1 s）。
    pub fn new(cfg: Arc<EngineConfig>, post: PostMode, lookahead_s: f32) -> Self {
        let analyzer = Analyzer::new(lookahead_s);
        let la = analyzer.lookahead_samples();
        let seed = cfg.params.seed;
        let mut e = Engine {
            tab: tables(),
            cfg: cfg.clone(),
            post,
            in_hp: Biquad::highpass(40.0, BUTTER_Q),
            gate: Gate::new(),
            denoiser: Denoiser::new(),
            deess: DeEsser::new(),
            soft: None,
            soft_src: None,
            analyzer,
            la_buf: vec![0.0; la],
            la_pos: 0,
            warm_lp: OnePoleLp::new(220.0),
            direct: DirectRenderer::new(),
            room: Room::new(tables()),
            lf: [EarLf::new(), EarLf::new()],
            pb: None,
            pb_src: None,
            out_hp: [Biquad::highpass(25.0, BUTTER_Q); 2],
            leveler: Leveler::new(),
            limiter: Limiter::new(2.0, 80.0),
            motion: Motion::new(seed),
            tm: 0.0,
            phase: 0,
            warm: (0.0, 0.0),
            lfg: [(1.0, 1.0); 2],
            feats: Features::default(),
            pose: Pose::default(),
            hr: Hrir::default(),
            xa: [0.0; CTRL],
            xr: [0.0; CTRL],
            room_dl: [0.0; ROOM_DELAY],
            room_dl_pos: 0,
            xd: [0.0; CTRL],
            dl: [0.0; CTRL],
            dr: [0.0; CTRL],
            in_pk: 0.0,
            out_pk: 0.0,
            primed: false,
            tele: Telemetry::default(),
            n_blocks: 0,
            n_speech: 0,
            trace: None,
            amp_ms: 1.0,
        };
        e.set_config(cfg);
        e
    }

    /// 换一份配置（在控制块边界生效）。
    pub fn set_config(&mut self, cfg: Arc<EngineConfig>) {
        let d = &cfg.derived;
        let changed = |a: &Option<Arc<Vec<f32>>>, b: &Option<Arc<Vec<f32>>>| match (a, b) {
            (Some(x), Some(y)) => !Arc::ptr_eq(x, y),
            (None, None) => false,
            _ => true,
        };
        if changed(&self.pb_src, &d.playback) {
            self.pb = d.playback.as_ref().map(|h| [Fir::new(h), Fir::new(h)]);
            self.pb_src = d.playback.clone();
        }
        if changed(&self.soft_src, &d.soft) {
            match (&mut self.soft, &d.soft) {
                (Some(f), Some(h)) if f.len() == h.len() => f.set_taps(h),
                (_, Some(h)) => self.soft = Some(Fir::new(h)),
                (_, None) => self.soft = None,
            }
            self.soft_src = d.soft.clone();
        }
        self.cfg = cfg;
    }

    /// 文件模式：用整段音频统计的说话阈值和中位音量。
    pub fn set_fixed_analysis(&mut self, thr_db: f32, median_db: f32) {
        self.analyzer.set_fixed(thr_db, median_db);
    }

    /// 算法本身的延时（样本，48 kHz）：降噪 20 ms（开启时）+ 预读 + 分数延时插值 + 限制器。
    /// 声音从嘴传到耳朵的传播时间（贴耳时约 0.3 ms）是效果的一部分，不算在内。
    pub fn latency_samples(&self) -> usize {
        let p = &self.cfg.params;
        (if p.denoise { DENOISE_DELAY } else { 0 }) + self.la_buf.len() + ROOM_DELAY + if self.post == PostMode::Realtime { self.limiter.delay() } else { 0 }
    }

    /// 控制块开始：推进说话时钟和位置，算出块末尾的滤波器、时延和各种增益。
    fn begin_block(&mut self) {
        let cfg = self.cfg.clone();
        let p = &cfg.params;
        self.feats = self.analyzer.features();
        let f = self.feats;
        self.n_blocks += 1;
        if f.act > 0.5 {
            self.n_speech += 1;
        }
        let (step, amp) = if p.pause_motion < 1.0 {
            // 晃动幅度随音量（0.5–1.6 倍），并按说话时段的均方把整体幅度归一，平均幅度不变
            let a = db_to_lin(f.loud_db).clamp(0.5, 1.6);
            if f.act > 0.5 {
                self.amp_ms += super::filters::ema_coef(8.0, CTRL_DT) * (a * a - self.amp_ms);
            }
            (clock_step(f.act, p.pause_motion), a / self.amp_ms.max(0.25).sqrt())
        } else {
            (CTRL_DT as f64, 1.0)
        };
        // 轨迹时间与音频对齐：有预读时音频晚 K 个控制块才出来，这段时间轨迹不动
        if self.n_blocks > (self.la_buf.len() / CTRL) as u64 {
            self.tm += step;
        }
        let pose = self.motion.step(p, self.tm, amp, f.follow_db);
        self.pose = pose;
        if let Some(tr) = &mut self.trace {
            let t = (self.n_blocks.saturating_sub((self.la_buf.len() / CTRL) as u64)) as f32 * CTRL_DT;
            tr.push([t, pose.az, pose.r, pose.r_base, f.act]);
        }
        let az_sofa = (360.0 - pose.az).rem_euclid(360.0); // 顺时针 -> SOFA 逆时针
        self.tab.lookup(az_sofa, pose.r, &mut self.hr);

        // 响度：沿不含忽远忽近的轨迹补偿；退开时说话人自己会说大声一点（只保留一部分物理衰减）
        let p_base = self.tab.lookup_power(az_sofa, pose.r_base).max(1e-20);
        let mut g = if p.level_follow != 1.0 { (p_base / cfg.derived.pref).powf(-(1.0 - p.level_follow) / 2.0) } else { 1.0 };
        let mut talker = 1.0;
        if (pose.r - pose.r_base).abs() > 1e-6 && p.sway_level < 1.0 {
            let p_now = self.tab.lookup_power(az_sofa, pose.r).max(1e-20);
            talker = (p_now / p_base).powf(-(1.0 - p.sway_level) / 2.0);
            g *= talker;
        }
        // 人头麦贴耳感：远耳额外压低，近耳加临场 / 空气感
        let rc = pose.r.clamp(R_MIN, R_MAX);
        let cl = closeness(pose.az, rc);
        let ipsi = if pose.az.rem_euclid(360.0) < 180.0 { 1 } else { 0 };
        for e in 0..2 {
            let mut ge = g;
            if e != ipsi && p.isolation_db > 0.0 {
                ge *= db_to_lin(-p.isolation_db * cl);
            }
            for v in self.hr.h[e].iter_mut() {
                *v *= ge;
            }
        }
        if cl > 0.0 {
            if let Some(eq) = &cfg.derived.ear_eq {
                let h = &mut self.hr.h[ipsi];
                let mut he = [0f32; NTAP];
                for (n, o) in he.iter_mut().enumerate() {
                    let kmax = n.min(EAR_EQ_TAPS - 1);
                    let mut s = 0.0;
                    for k in 0..=kmax {
                        s += eq[k] * h[n - k];
                    }
                    *o = s;
                }
                for i in 0..16 {
                    let t = i as f32 / 15.0;
                    he[NTAP - 16 + i] *= 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
                }
                for (v, e) in h.iter_mut().zip(he.iter()) {
                    *v += cl * (e - *v);
                }
            }
        }
        let d = [rc / C * FS + self.hr.tau[0] + BASE_DELAY, rc / C * FS + self.hr.tau[1] + BASE_DELAY];
        self.direct.set_target(&self.hr.h, d);

        // 近讲温暖感（两耳共同）、贴耳低频（近耳加、远耳减）
        let prox = ((0.6 - pose.r) / (0.6 - 0.13)).clamp(0.0, 1.0);
        let warm = if p.warmth_db > 0.0 { db_to_lin(p.warmth_db * prox) - 1.0 } else { 0.0 };
        let mut lfg = [1.0f32; 2];
        for (e, v) in lfg.iter_mut().enumerate() {
            *v = if e == ipsi { db_to_lin(p.ear_warmth_db * cl) } else { db_to_lin(-p.far_lf_cut_db * cl) };
        }
        if self.primed {
            self.warm = (self.warm.1, warm);
            for e in 0..2 {
                self.lfg[e] = (self.lfg[e].1, lfg[e]);
            }
        } else {
            self.warm = (warm, warm);
            for e in 0..2 {
                self.lfg[e] = (lfg[e], lfg[e]);
            }
            self.primed = true;
        }
        self.room.set_target(pose.az, talker);

        self.tele = Telemetry {
            az: pose.az,
            r: pose.r,
            closeness: cl,
            act: f.act,
            in_db: lin_to_db(self.in_pk),
            out_db: lin_to_db(self.out_pk),
            gr_db: self.limiter.gr_db,
            level_gain_db: self.leveler.gain_db,
            n_moves: self.motion.n_moves,
            n_sways: self.motion.n_sways,
        };
        self.in_pk *= 0.97;
        self.out_pk *= 0.97;
    }

    /// 处理一段单声道输入（任意长度），输出等长的左右声道。
    pub fn process(&mut self, input: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        let n = input.len();
        let mut i = 0;
        while i < n {
            if self.phase == 0 {
                self.begin_block();
            }
            let m = (n - i).min(CTRL - self.phase);
            self.run(&input[i..i + m], &mut out_l[i..i + m], &mut out_r[i..i + m]);
            self.phase += m;
            if self.phase == CTRL {
                self.analyzer.end_block();
                self.phase = 0;
            }
            i += m;
        }
    }

    fn run(&mut self, input: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        let cfg = self.cfg.clone();
        let p = &cfg.params;
        let m = input.len();
        let ph = self.phase;
        let la = self.la_buf.len();
        for i in 0..m {
            let x = input[i];
            self.in_pk = self.in_pk.max(x.abs());
            let mut v = self.in_hp.process(x);
            v = self.gate.process(v, p.gate_db);
            if p.denoise {
                v = self.denoiser.process(v, p.denoise_mix);
            }
            if p.deess_db > 0.0 {
                v = self.deess.process(v, p.deess_db);
            }
            if let Some(f) = &mut self.soft {
                v = f.process(v);
            }
            self.analyzer.push(v);
            let a = if la == 0 {
                v
            } else {
                let o = self.la_buf[self.la_pos];
                self.la_buf[self.la_pos] = v;
                self.la_pos = (self.la_pos + 1) % la;
                o
            };
            self.xa[i] = a;
            // 房间声的输入和直达声一样多延后几个样本（分数延时插值器的固定时延），两者的相对时间与 Python 版一致
            self.xr[i] = self.room_dl[self.room_dl_pos];
            self.room_dl[self.room_dl_pos] = a;
            self.room_dl_pos = (self.room_dl_pos + 1) % ROOM_DELAY;
            let wp = (ph + i) as f32 / CTRL as f32;
            let gb = self.warm.0 + (self.warm.1 - self.warm.0) * wp;
            self.xd[i] = a + gb * self.warm_lp.process(a);
        }
        self.direct.process(&self.xd[..m], ph, &mut self.dl[..m], &mut self.dr[..m]);
        if p.ear_warmth_db > 0.0 || p.far_lf_cut_db > 0.0 {
            for i in 0..m {
                let wp = (ph + i) as f32 / CTRL as f32;
                let gl = self.lfg[0].0 + (self.lfg[0].1 - self.lfg[0].0) * wp;
                let gr = self.lfg[1].0 + (self.lfg[1].1 - self.lfg[1].0) * wp;
                self.dl[i] = self.lf[0].process(self.dl[i], gl);
                self.dr[i] = self.lf[1].process(self.dr[i], gr);
            }
        }
        if p.room {
            self.room.process(&self.xr[..m], ph, cfg.derived.room_gain, &mut self.dl[..m], &mut self.dr[..m]);
        }
        let speaking = self.feats.act > 0.5;
        let gain = db_to_lin(p.out_gain_db);
        let ceiling = db_to_lin(p.ceiling_db - 0.1);
        for i in 0..m {
            let (mut l, mut r) = (self.dl[i], self.dr[i]);
            if let Some(pb) = &mut self.pb {
                l = pb[0].process(l);
                r = pb[1].process(r);
            }
            l = self.out_hp[0].process(l);
            r = self.out_hp[1].process(r);
            if self.post == PostMode::Realtime {
                let lg = if p.auto_level { self.leveler.process(l, r, speaking, p.target_lufs) } else { 1.0 };
                let (a, b) = self.limiter.process(l * lg * gain, r * lg * gain, ceiling);
                l = a;
                r = b;
            }
            self.out_pk = self.out_pk.max(l.abs()).max(r.abs());
            out_l[i] = l;
            out_r[i] = r;
        }
    }
}
