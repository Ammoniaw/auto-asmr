//! 实时模式：麦克风 → 引擎 → 耳机 / 虚拟声卡。
//!
//! 输入回调把麦克风的声音（重采样到 48 kHz）写进无锁环形缓冲；输出回调取出来、过引擎、再重采样到
//! 输出设备的采样率。两个设备的时钟不会完全一致，用输出端重采样比例的微调（±0.5%）把缓冲量稳在目标值。

use crate::dsp::engine::{Engine, EngineConfig, PostMode, Telemetry};
use crate::dsp::mix::{MixSetup, Mixer, MAX_VOICES};
use crate::dsp::resample::Resampler;
use crate::dsp::{FS, FS_USIZE};
use crate::spsc;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, FromSample, Sample, SampleFormat, SizedSample, StreamConfig, SupportedBufferSize};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

#[derive(Clone, Debug, Default)]
pub struct DeviceList {
    pub inputs: Vec<DeviceInfo>,
    pub outputs: Vec<DeviceInfo>,
}

/// 本机可用的音频后端（Windows：WASAPI；macOS：CoreAudio；Linux：ALSA 等）。
pub fn hosts() -> Vec<(cpal::HostId, String)> {
    cpal::available_hosts().into_iter().map(|h| (h, h.name().to_string())).collect()
}

fn host_of(id: Option<cpal::HostId>) -> cpal::Host {
    id.and_then(|h| cpal::host_from_id(h).ok()).unwrap_or_else(cpal::default_host)
}

/// 列出某个后端下的输入、输出设备（自动查找，给下拉框用）。
pub fn list_devices(host: Option<cpal::HostId>) -> DeviceList {
    let host = host_of(host);
    let def_in = host.default_input_device().and_then(|d| d.id().ok()).map(|i| i.to_string());
    let def_out = host.default_output_device().and_then(|d| d.id().ok()).map(|i| i.to_string());
    let mut list = DeviceList::default();
    let info = |d: &cpal::Device, def: &Option<String>| -> Option<DeviceInfo> {
        let id = d.id().ok()?.to_string();
        let name = d.description().map(|x| x.name().to_string()).unwrap_or_else(|_| id.clone());
        Some(DeviceInfo { is_default: def.as_deref() == Some(id.as_str()), id, name })
    };
    if let Ok(devs) = host.input_devices() {
        list.inputs = devs.filter_map(|d| info(&d, &def_in)).collect();
    }
    if let Ok(devs) = host.output_devices() {
        list.outputs = devs.filter_map(|d| info(&d, &def_out)).collect();
    }
    list
}

pub fn print_devices() {
    for (h, name) in hosts() {
        println!("音频后端：{name}");
        let l = list_devices(Some(h));
        println!("  输入设备：");
        for d in &l.inputs {
            println!("    {}{}    [{}]", d.name, if d.is_default { "（默认）" } else { "" }, d.id);
        }
        println!("  输出设备：");
        for d in &l.outputs {
            println!("    {}{}    [{}]", d.name, if d.is_default { "（默认）" } else { "" }, d.id);
        }
    }
}

#[derive(Clone, Debug)]
pub struct RtOptions {
    pub host: Option<cpal::HostId>,
    /// 设备 id；None = 系统默认设备。
    pub input: Option<String>,
    pub output: Option<String>,
    /// 希望的缓冲帧数（0 = 设备默认）。越小延迟越低，太小会爆音。
    pub buffer_frames: u32,
    /// 输入取哪个声道：0、1…；usize::MAX = 所有声道取平均。
    pub input_channel: usize,
}

impl Default for RtOptions {
    fn default() -> Self {
        RtOptions { host: None, input: None, output: None, buffer_frames: 256, input_channel: 0 }
    }
}

/// 音频线程与界面之间共享的状态（都是原子量，音频线程从不阻塞）。
pub struct Shared {
    cfg: Mutex<Arc<EngineConfig>>,
    cfg_ver: AtomicU64,
    /// true = 监听原声（不经过效果），false = 输出效果。
    pub monitor_dry: AtomicBool,
    tele: [AtomicU32; 15],
    /// 环形缓冲里平均积压的时间（微秒）。
    pub fill_us: AtomicU32,
    /// 最近每次输出回调的处理时间（微秒，平滑 / 最大）。
    pub proc_us: AtomicU32,
    pub proc_max_us: AtomicU32,
    /// 断流次数（设备报告的短暂断流，或输入来不及 / 积压过多）。
    pub xruns: AtomicU32,
    /// 引擎算法延时（样本，48 kHz）。
    pub algo_samples: AtomicU32,
    /// 输入 / 输出回调一次的帧数。
    pub in_frames: AtomicU32,
    pub out_frames: AtomicU32,
    /// 试听文件时：播放位置和总长（48 kHz 样本）；播完后 `finished` 置位。
    pub play_pos: AtomicU64,
    pub play_len: AtomicU64,
    pub finished: AtomicBool,
    /// 待执行的文件跳转；音频回调取走最新请求，不重新打开输出设备。
    seek_sample: AtomicU64,
    error: Mutex<Option<String>>,
    /// 多音频合成试听：当前的合成配置（与 `cfg` 共用版本号 `cfg_ver`）。
    mix: Mutex<Option<Arc<MixSetup>>>,
    /// 多音频合成试听时各音源此刻的位置：每个音源 [方位角, 仰角, 距离, 说话活动]。
    vtele: [AtomicU32; MAX_VOICES * 4],
    n_voices: AtomicU32,
}

impl Shared {
    fn new(cfg: Arc<EngineConfig>) -> Self {
        Shared {
            mix: Mutex::new(None),
            vtele: Default::default(),
            n_voices: AtomicU32::new(0),
            cfg: Mutex::new(cfg),
            cfg_ver: AtomicU64::new(1),
            monitor_dry: AtomicBool::new(false),
            tele: Default::default(),
            fill_us: AtomicU32::new(0),
            proc_us: AtomicU32::new(0),
            proc_max_us: AtomicU32::new(0),
            xruns: AtomicU32::new(0),
            algo_samples: AtomicU32::new(0),
            in_frames: AtomicU32::new(0),
            out_frames: AtomicU32::new(0),
            play_pos: AtomicU64::new(0),
            play_len: AtomicU64::new(0),
            finished: AtomicBool::new(false),
            seek_sample: AtomicU64::new(u64::MAX),
            error: Mutex::new(None),
        }
    }

    fn store_tele(&self, t: &Telemetry) {
        let v = [t.az, t.r, t.closeness, t.act, t.in_db, t.out_db, t.gr_db, t.level_gain_db, t.n_moves as f32, t.n_sways as f32, t.el, t.n_tops as f32, t.n_updowns as f32, t.depth_cm, t.n_strokes as f32];
        for (a, x) in self.tele.iter().zip(v) {
            a.store(x.to_bits(), Ordering::Relaxed);
        }
    }

    pub fn telemetry(&self) -> Telemetry {
        let g = |i: usize| f32::from_bits(self.tele[i].load(Ordering::Relaxed));
        Telemetry {
            az: g(0),
            r: g(1),
            closeness: g(2),
            act: g(3),
            in_db: g(4),
            out_db: g(5),
            gr_db: g(6),
            level_gain_db: g(7),
            n_moves: g(8) as u32,
            n_sways: g(9) as u32,
            el: g(10),
            n_tops: g(11) as u32,
            n_updowns: g(12) as u32,
            depth_cm: g(13),
            n_strokes: g(14) as u32,
        }
    }

    /// 设备出错（拔掉、被占用等）时的说明；读一次即清空。
    pub fn take_error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|mut e| e.take())
    }

    fn stream_error(&self, device: &str, error: cpal::Error) {
        // Xrun 表示短暂音频不连续；CPAL 后端仍在运行，不能因此停掉音频流。
        if error.kind() == cpal::ErrorKind::Xrun {
            self.xruns.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if let Ok(mut pending) = self.error.lock() {
            *pending = Some(format!("{device}设备出错：{error}"));
        }
    }

    pub fn seek_preview(&self, sample: u64) {
        self.seek_sample.store(sample.min(self.play_len.load(Ordering::Relaxed)), Ordering::Release);
    }

    pub fn preview_finished(&self) -> bool {
        self.seek_sample.load(Ordering::Acquire) == u64::MAX && self.finished.load(Ordering::Relaxed)
    }

    fn set_config(&self, cfg: Arc<EngineConfig>) {
        if let Ok(mut g) = self.cfg.lock() {
            *g = cfg;
            self.cfg_ver.fetch_add(1, Ordering::Release);
        }
    }

    fn set_mix(&self, setup: Arc<MixSetup>) {
        if let Ok(mut g) = self.mix.lock() {
            *g = Some(setup);
            self.cfg_ver.fetch_add(1, Ordering::Release);
        }
    }

    fn store_voice_tele(&self, mx: &Mixer) {
        let n = mx.voices().min(MAX_VOICES);
        for k in 0..n {
            let t = mx.engine(k).tele;
            for (j, x) in [t.az, t.el, t.r, t.act].into_iter().enumerate() {
                self.vtele[k * 4 + j].store(x.to_bits(), Ordering::Relaxed);
            }
        }
        self.n_voices.store(n as u32, Ordering::Relaxed);
    }

    /// 多音频合成试听时各音源此刻的 [方位角, 仰角, 距离, 说话活动]；不是合成试听时为空。
    pub fn voice_telemetry(&self) -> Vec<[f32; 4]> {
        let n = (self.n_voices.load(Ordering::Relaxed) as usize).min(MAX_VOICES);
        (0..n).map(|k| std::array::from_fn(|j| f32::from_bits(self.vtele[k * 4 + j].load(Ordering::Relaxed)))).collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct RtInfo {
    pub input_name: String,
    pub output_name: String,
    pub in_rate: u32,
    pub out_rate: u32,
    pub in_channels: u16,
    pub out_channels: u16,
}

pub struct Realtime {
    _input: Option<cpal::Stream>,
    _output: cpal::Stream,
    pub shared: Arc<Shared>,
    pub info: RtInfo,
}

fn err_text(what: &str, e: cpal::Error) -> String {
    format!("{what}：{e}")
}

fn pick_config(dev: &cpal::Device, input: bool, want_frames: u32) -> Result<(StreamConfig, SampleFormat), String> {
    let def = if input { dev.default_input_config() } else { dev.default_output_config() }.map_err(|e| err_text("读取设备格式失败", e))?;
    let mut cfg = def.config();
    cfg.buffer_size = match (want_frames, def.buffer_size()) {
        (0, _) => BufferSize::Default,
        (n, SupportedBufferSize::Range { min, max }) => BufferSize::Fixed(n.clamp(*min, *max)),
        (_, _) => BufferSize::Default,
    };
    Ok((cfg, def.sample_format()))
}

fn build_in<T>(dev: &cpal::Device, cfg: StreamConfig, sel: usize, mut sink: impl FnMut(&[f32]) + Send + 'static, shared: Arc<Shared>) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let ch = cfg.channels.max(1) as usize;
    let mut tmp: Vec<f32> = Vec::with_capacity(16384);
    let sh = shared.clone();
    dev.build_input_stream(
        cfg,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            tmp.clear();
            for f in data.chunks_exact(ch) {
                let v = if sel < ch {
                    f32::from_sample(f[sel])
                } else {
                    f.iter().map(|&s| f32::from_sample(s)).sum::<f32>() / ch as f32
                };
                tmp.push(v);
            }
            sink(&tmp);
        },
        move |e| sh.stream_error("输入", e),
        None,
    )
}

fn build_out<T>(dev: &cpal::Device, cfg: StreamConfig, mut fill: impl FnMut(&mut [f32], &mut [f32]) + Send + 'static, shared: Arc<Shared>) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let ch = cfg.channels.max(1) as usize;
    let (mut l, mut r): (Vec<f32>, Vec<f32>) = (vec![0.0; 16384], vec![0.0; 16384]);
    let sh = shared.clone();
    dev.build_output_stream(
        cfg,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            let frames = data.len() / ch;
            if l.len() < frames {
                l.resize(frames, 0.0);
                r.resize(frames, 0.0);
            }
            shared.out_frames.store(frames as u32, Ordering::Relaxed);
            fill(&mut l[..frames], &mut r[..frames]);
            for (i, f) in data.chunks_exact_mut(ch).enumerate() {
                if ch == 1 {
                    f[0] = T::from_sample(0.5 * (l[i] + r[i]));
                } else {
                    f[0] = T::from_sample(l[i]);
                    f[1] = T::from_sample(r[i]);
                    for o in f[2..].iter_mut() {
                        *o = T::EQUILIBRIUM;
                    }
                }
            }
        },
        move |e| sh.stream_error("输出", e),
        None,
    )
}

macro_rules! by_format {
    ($fmt:expr, $f:ident, $($arg:expr),*) => {
        match $fmt {
            SampleFormat::F32 => $f::<f32>($($arg),*),
            SampleFormat::I16 => $f::<i16>($($arg),*),
            SampleFormat::I32 => $f::<i32>($($arg),*),
            SampleFormat::U16 => $f::<u16>($($arg),*),
            SampleFormat::I8 => $f::<i8>($($arg),*),
            SampleFormat::U8 => $f::<u8>($($arg),*),
            SampleFormat::F64 => $f::<f64>($($arg),*),
            other => return Err(format!("不支持的采样格式：{other}")),
        }
    };
}

/// 引擎的输入来源。
enum Source {
    /// 麦克风：输入回调写进环形缓冲。
    Ring(spsc::Consumer),
    /// 试听文件：整段已解码成 48 kHz 单声道。
    Memory { data: Arc<Vec<f32>>, pos: usize },
    /// 多音频合成的试听：几段音频（各自已解码成 48 kHz 单声道）同时播放，总长按最长的那段。
    Multi { data: Vec<Arc<Vec<f32>>>, pos: usize, len: usize },
}

/// 输出回调里的处理器：一个引擎（麦克风 / 试听一个文件），或多音频合成（每个音源一个引擎，相加后统一收尾）。
enum Proc {
    One(Engine),
    Mix(Mixer),
}

impl Proc {
    fn latency_samples(&self) -> usize {
        match self {
            Proc::One(e) => e.latency_samples(),
            Proc::Mix(m) => m.latency_samples(),
        }
    }

    fn tele(&self) -> &Telemetry {
        match self {
            Proc::One(e) => &e.tele,
            Proc::Mix(m) => &m.tele,
        }
    }
}

/// 输出回调里做的全部工作（不依赖声卡驱动，可以单独测试）。
struct Pump {
    src: Source,
    proc: Proc,
    /// 多音频合成时每个音源的一块输入。
    inbs: Vec<Vec<f32>>,
    shared: Arc<Shared>,
    cfg_ver: u64,
    rs: [Resampler; 2],
    out_rate: f64,
    q: [Vec<f32>; 2], // 已经是输出采样率的样本，等着被取走
    qpos: usize,
    primed: bool,
    fill_avg: f32,
    in_max: f32,
    proc_avg: f32,
    inb: Vec<f32>,
    el: Vec<f32>,
    er: Vec<f32>,
    last_output: [f32; 2],
    seek_fade: usize,
}

const N: usize = 64;

impl Pump {
    #[cfg(test)]
    fn new(src: Source, eng: Engine, shared: Arc<Shared>, out_rate: f64) -> Self {
        Self::with_proc(src, Proc::One(eng), shared, out_rate)
    }

    fn with_proc(src: Source, proc: Proc, shared: Arc<Shared>, out_rate: f64) -> Self {
        shared.algo_samples.store(proc.latency_samples() as u32, Ordering::Relaxed);
        match &src {
            Source::Memory { data, pos } => {
                shared.play_len.store(data.len() as u64, Ordering::Relaxed);
                shared.play_pos.store((*pos).min(data.len()) as u64, Ordering::Relaxed);
            }
            Source::Multi { pos, len, .. } => {
                shared.play_len.store(*len as u64, Ordering::Relaxed);
                shared.play_pos.store((*pos).min(*len) as u64, Ordering::Relaxed);
            }
            Source::Ring(_) => {}
        }
        let n_in = if let Source::Multi { data, .. } = &src { data.len() } else { 0 };
        Pump {
            src,
            proc,
            inbs: (0..n_in).map(|_| vec![0.0; N]).collect(),
            shared,
            cfg_ver: 1,
            rs: [Resampler::new(FS as f64, out_rate), Resampler::new(FS as f64, out_rate)],
            out_rate,
            q: [Vec::with_capacity(65536), Vec::with_capacity(65536)],
            qpos: 0,
            primed: false,
            fill_avg: 0.0,
            in_max: 0.0,
            proc_avg: 0.0,
            inb: vec![0.0; N],
            el: vec![0.0; N],
            er: vec![0.0; N],
            last_output: [0.0; 2],
            seek_fade: 0,
        }
    }

    /// 取下一块 64 个输入样本；麦克风还没送到时返回 false。
    fn next_block(&mut self) -> bool {
        match &mut self.src {
            Source::Ring(rx) => rx.len() >= N && rx.pop_slice(&mut self.inb[..N]) == N,
            Source::Memory { data, pos } => {
                let n = data.len();
                for (i, v) in self.inb[..N].iter_mut().enumerate() {
                    *v = if *pos + i < n { data[*pos + i] } else { 0.0 };
                }
                *pos += N;
                self.shared.play_pos.store((*pos).min(n) as u64, Ordering::Relaxed);
                // 播完再多放 1 秒（房间尾音），然后停
                if *pos > n + FS_USIZE {
                    self.shared.finished.store(true, Ordering::Relaxed);
                }
                true
            }
            Source::Multi { data, pos, len } => {
                for (d, b) in data.iter().zip(self.inbs.iter_mut()) {
                    let n = d.len();
                    for (i, v) in b[..N].iter_mut().enumerate() {
                        *v = if *pos + i < n { d[*pos + i] } else { 0.0 };
                    }
                }
                *pos += N;
                self.shared.play_pos.store((*pos).min(*len) as u64, Ordering::Relaxed);
                if *pos > *len + FS_USIZE {
                    self.shared.finished.store(true, Ordering::Relaxed);
                }
                true
            }
        }
    }

    fn fill(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        crate::dsp::flush_denormals();
        let t0 = std::time::Instant::now();
        // 界面改了参数：换一份配置（拿不到锁就下次再说，绝不等待）
        let ver = self.shared.cfg_ver.load(Ordering::Acquire);
        if ver != self.cfg_ver {
            let applied = match &mut self.proc {
                Proc::One(e) => match self.shared.cfg.try_lock() {
                    Ok(g) => {
                        e.set_config(g.clone());
                        true
                    }
                    Err(_) => false,
                },
                // 合成：各音源的配置和增益（界面线程里已经算好，这里只换指针、抄增益，不分配内存）
                Proc::Mix(m) => match self.shared.mix.try_lock() {
                    Ok(g) => {
                        if let Some(s) = g.as_ref() {
                            m.set_setup(s);
                        }
                        true
                    }
                    Err(_) => false,
                },
            };
            if applied {
                self.cfg_ver = ver;
                self.shared.algo_samples.store(self.proc.latency_samples() as u32, Ordering::Relaxed);
            }
        }
        let seek = self.shared.seek_sample.load(Ordering::Acquire);
        if seek != u64::MAX {
            let len = match &self.src {
                Source::Memory { data, .. } => Some(data.len()),
                Source::Multi { len, .. } => Some(*len),
                Source::Ring(_) => None,
            };
            if let (Some(len), Source::Memory { pos, .. } | Source::Multi { pos, .. }) = (len, &mut self.src) {
                *pos = seek.min(len as u64) as usize;
                self.shared.play_pos.store(*pos as u64, Ordering::Relaxed);
                self.shared.finished.store(false, Ordering::Relaxed);
                for q in &mut self.q {
                    q.clear();
                }
                self.qpos = 0;
                for rs in &mut self.rs {
                    rs.reset();
                }
                self.seek_fade = (self.out_rate * 0.005).round().max(1.0) as usize;
            }
            // 先清除播完状态，再确认跳转完成；期间的新拖动请求留到下一回调。
            let _ = self.shared.seek_sample.compare_exchange(seek, u64::MAX, Ordering::AcqRel, Ordering::Acquire);
        }
        let frames = out_l.len();
        // 麦克风：目标积压量 = 一次输入回调的量 + 2 ms 余量；攒够了才开始
        let mut target = 0.0;
        if let Source::Ring(rx) = &self.src {
            let in48 = self.shared.in_frames.load(Ordering::Relaxed) as f32;
            self.in_max = (self.in_max * 0.999).max(in48);
            target = self.in_max + 0.002 * FS;
            if !self.primed {
                if (rx.len() as f32) < target + N as f32 {
                    out_l.fill(0.0);
                    out_r.fill(0.0);
                    return;
                }
                self.primed = true;
                self.fill_avg = rx.len() as f32;
            }
        }
        let dry = self.shared.monitor_dry.load(Ordering::Relaxed);
        let mut stalled = false;
        while self.q[0].len() - self.qpos < frames {
            if !self.next_block() {
                stalled = true; // 输入还没到：这次先输出静音，引擎不往前走
                break;
            }
            match &mut self.proc {
                Proc::One(eng) => {
                    if dry {
                        self.el[..N].copy_from_slice(&self.inb[..N]);
                        self.er[..N].copy_from_slice(&self.inb[..N]);
                    } else {
                        eng.process(&self.inb[..N], &mut self.el[..N], &mut self.er[..N]);
                    }
                }
                Proc::Mix(mx) => {
                    if dry {
                        for i in 0..N {
                            let s: f32 = self.inbs.iter().map(|b| b[i]).sum();
                            self.el[i] = s;
                            self.er[i] = s;
                        }
                    } else {
                        let ins: [&[f32]; MAX_VOICES] = std::array::from_fn(|k| self.inbs.get(k).map_or(&[][..], |b| &b[..N]));
                        // 自动平衡按这一块在各音频里的位置取增益（拖动播放条后也对得上）
                        if let Source::Multi { pos, .. } = &self.src {
                            mx.seek(pos.saturating_sub(N));
                        }
                        mx.process(&ins[..self.inbs.len().min(MAX_VOICES)], &mut self.el[..N], &mut self.er[..N]);
                    }
                }
            }
            if self.qpos > 0 && self.qpos == self.q[0].len() {
                self.q[0].clear();
                self.q[1].clear();
                self.qpos = 0;
            }
            self.rs[0].process(&self.el[..N], &mut self.q[0]);
            self.rs[1].process(&self.er[..N], &mut self.q[1]);
        }
        let have = (self.q[0].len() - self.qpos).min(self.q[1].len() - self.qpos).min(frames);
        out_l[..have].copy_from_slice(&self.q[0][self.qpos..self.qpos + have]);
        out_r[..have].copy_from_slice(&self.q[1][self.qpos..self.qpos + have]);
        out_l[have..].fill(0.0);
        out_r[have..].fill(0.0);
        // 用 5 ms 淡入衔接跳转，避免源波形突变产生咔嗒声。
        let fade_len = (self.out_rate * 0.005).round().max(1.0) as usize;
        for i in 0..frames.min(self.seek_fade) {
            let mix = 1.0 - self.seek_fade as f32 / fade_len as f32;
            out_l[i] = self.last_output[0] * (1.0 - mix) + out_l[i] * mix;
            out_r[i] = self.last_output[1] * (1.0 - mix) + out_r[i] * mix;
            self.seek_fade -= 1;
        }
        if let (Some(&l), Some(&r)) = (out_l.last(), out_r.last()) {
            // 淡入跨多个回调时保留同一衔接起点。
            if self.seek_fade == 0 {
                self.last_output = [l, r];
            }
        }
        self.qpos += have;
        if self.qpos > 8192 {
            self.q[0].drain(..self.qpos);
            self.q[1].drain(..self.qpos);
            self.qpos = 0;
        }
        if stalled {
            self.shared.xruns.fetch_add(1, Ordering::Relaxed);
        }
        if let Source::Ring(rx) = &mut self.src {
            // 时钟漂移：积压偏多就稍微快一点消耗，偏少就慢一点；积压太多（> 目标 + 150 ms）直接丢掉旧的
            let fill = rx.len() as f32;
            self.fill_avg += 0.05 * (fill - self.fill_avg);
            if fill > target + 0.15 * FS {
                rx.skip((fill - target) as usize);
                self.fill_avg = target;
                self.shared.xruns.fetch_add(1, Ordering::Relaxed);
            }
            let trim = 1.0 + (((self.fill_avg - target) / FS) as f64 * 0.5).clamp(-0.005, 0.005);
            for r in self.rs.iter_mut() {
                r.set_trim(FS as f64, self.out_rate, trim);
            }
            self.shared.fill_us.store((self.fill_avg / FS * 1e6) as u32, Ordering::Relaxed);
        }
        self.shared.store_tele(self.proc.tele());
        if let Proc::Mix(mx) = &self.proc {
            self.shared.store_voice_tele(mx);
        }
        let us = t0.elapsed().as_secs_f32() * 1e6;
        self.proc_avg += 0.05 * (us - self.proc_avg);
        self.shared.proc_us.store(self.proc_avg as u32, Ordering::Relaxed);
        self.shared.proc_max_us.fetch_max(us as u32, Ordering::Relaxed);
    }
}

/// 麦克风一侧：把设备采样率的输入重采样到 48 kHz，写进环形缓冲。
fn mic_sink(in_rate: u32, mut tx: spsc::Producer, shared: Arc<Shared>) -> impl FnMut(&[f32]) + Send + 'static {
    let mut rs = if in_rate as usize != FS_USIZE { Some(Resampler::new(in_rate as f64, FS as f64)) } else { None };
    let mut conv: Vec<f32> = Vec::with_capacity(32768);
    move |x: &[f32]| {
        let n = match &mut rs {
            Some(r) => {
                conv.clear();
                r.process(x, &mut conv);
                tx.push_slice(&conv);
                conv.len()
            }
            None => {
                tx.push_slice(x);
                x.len()
            }
        };
        shared.in_frames.store(n as u32, Ordering::Relaxed);
    }
}

/// 实时引擎的输入从哪里来。
enum Feed {
    Mic,
    /// 试听一个文件：（48 kHz 单声道, 从第几个样本开始）。
    File(Arc<Vec<f32>>, usize),
    /// 多音频合成的试听：（各段 48 kHz 单声道, 合成配置, 从第几个样本开始）。
    Mix(Vec<Arc<Vec<f32>>>, Arc<MixSetup>, usize),
}

impl Realtime {
    /// 麦克风 → 引擎 → 输出设备。
    pub fn start(opts: &RtOptions, cfg: Arc<EngineConfig>) -> Result<Realtime, String> {
        Self::start_inner(opts, cfg, Feed::Mic)
    }

    /// 试听：用实时引擎把一段已解码的音频（48 kHz 单声道）播到输出设备，可以边听边调参数。
    pub fn start_preview(opts: &RtOptions, cfg: Arc<EngineConfig>, audio: Arc<Vec<f32>>, start_sample: usize) -> Result<Realtime, String> {
        Self::start_inner(opts, cfg, Feed::File(audio, start_sample))
    }

    /// 多音频合成的试听：几段已解码的音频（48 kHz 单声道，与 `setup` 里的音源一一对应）各在自己的位置，同时播放。
    /// 播放中可以用 [`Realtime::set_mix`] 换参数、挪位置、改音量，不用重新开始。
    pub fn start_preview_mix(opts: &RtOptions, setup: Arc<MixSetup>, audio: Vec<Arc<Vec<f32>>>, start_sample: usize) -> Result<Realtime, String> {
        if audio.is_empty() || audio.len() != setup.cfgs.len() || audio.len() > MAX_VOICES {
            return Err(format!("合成试听需要 1–{MAX_VOICES} 段音频，且与位置一一对应"));
        }
        let cfg = setup.cfgs[0].clone();
        Self::start_inner(opts, cfg, Feed::Mix(audio, setup, start_sample))
    }

    fn start_inner(opts: &RtOptions, cfg: Arc<EngineConfig>, feed: Feed) -> Result<Realtime, String> {
        let host = host_of(opts.host);
        let find = |id: &Option<String>, input: bool| -> Result<cpal::Device, String> {
            let none = if input { "没有可用的输入设备（麦克风）" } else { "没有可用的输出设备" };
            match id {
                Some(s) => {
                    let did = s.parse().map_err(|_| format!("设备 id 无法识别：{s}"))?;
                    host.device_by_id(&did).ok_or_else(|| "找不到所选设备（可能已拔出），请重载设备列表".to_string())
                }
                None => if input { host.default_input_device() } else { host.default_output_device() }.ok_or_else(|| none.to_string()),
            }
        };
        let name = |d: &cpal::Device| d.description().map(|x| x.name().to_string()).unwrap_or_default();
        let out_dev = find(&opts.output, false)?;
        let (out_cfg, out_fmt) = pick_config(&out_dev, false, opts.buffer_frames)?;
        let sh = Shared::new(cfg.clone());
        if let Feed::Mix(_, setup, _) = &feed {
            if let Ok(mut g) = sh.mix.lock() {
                *g = Some(setup.clone());
            }
        }
        let shared = Arc::new(sh);
        let out_rate = out_cfg.sample_rate as f64;
        let mut info = RtInfo { output_name: name(&out_dev), out_rate: out_cfg.sample_rate, out_channels: out_cfg.channels, ..Default::default() };

        let (input, src, proc) = match feed {
            Feed::File(audio, pos) => {
                info.input_name = "文件".into();
                info.in_rate = FS_USIZE as u32;
                info.in_channels = 1;
                // 试听不开主播模式的"话少时也一直在动"：动作严格跟着说话走，与文件转换一致
                let eng = Engine::new(cfg, PostMode::Realtime, 0.0);
                (None, Source::Memory { data: audio, pos }, Proc::One(eng))
            }
            Feed::Mix(audio, setup, pos) => {
                info.input_name = format!("{} 个文件合成", audio.len());
                info.in_rate = FS_USIZE as u32;
                info.in_channels = audio.len() as u16;
                let len = audio.iter().map(|a| a.len()).max().unwrap_or(0);
                (None, Source::Multi { data: audio, pos, len }, Proc::Mix(Mixer::new(&setup, PostMode::Realtime, 0.0)))
            }
            Feed::Mic => {
                let in_dev = find(&opts.input, true)?;
                let (in_cfg, in_fmt) = pick_config(&in_dev, true, opts.buffer_frames).map_err(|e| format!("{e}（可能没有麦克风，或没有授予麦克风权限）"))?;
                info.input_name = name(&in_dev);
                info.in_rate = in_cfg.sample_rate;
                info.in_channels = in_cfg.channels;
                let (tx, rx) = spsc::channel(FS_USIZE); // 最多积压 1 秒
                let sink = mic_sink(in_cfg.sample_rate, tx, shared.clone());
                let st = by_format!(in_fmt, build_in, &in_dev, in_cfg, opts.input_channel, sink, shared.clone()).map_err(|e| err_text("打开输入设备失败（检查麦克风权限，或换一个缓冲大小）", e))?;
                let mut eng = Engine::new(cfg, PostMode::Realtime, 0.0);
                eng.set_live(true);
                (Some(st), Source::Ring(rx), Proc::One(eng))
            }
        };
        let mut pump = Pump::with_proc(src, proc, shared.clone(), out_rate);
        let fill = move |l: &mut [f32], r: &mut [f32]| pump.fill(l, r);
        let output = by_format!(out_fmt, build_out, &out_dev, out_cfg, fill, shared.clone()).map_err(|e| err_text("打开输出设备失败（换一个缓冲大小试试）", e))?;
        if let Some(i) = &input {
            i.play().map_err(|e| err_text("启动输入失败", e))?;
        }
        output.play().map_err(|e| err_text("启动输出失败", e))?;
        Ok(Realtime { _input: input, _output: output, shared, info })
    }

    /// 界面改了参数后调用。
    pub fn set_config(&self, cfg: Arc<EngineConfig>) {
        self.shared.set_config(cfg);
    }

    /// 合成试听中改了参数、位置或音量后调用（音源数要与开始时相同）。
    pub fn set_mix(&self, setup: Arc<MixSetup>) {
        self.shared.set_mix(setup);
    }

    /// 估计的总延迟（毫秒）：输入缓冲 + 环形缓冲积压 + 算法 + 重采样 + 输出缓冲。不含声卡驱动自身的延迟。
    pub fn latency_ms(&self) -> f32 {
        let s = &self.shared;
        let inb = s.in_frames.load(Ordering::Relaxed) as f32 / FS * 1000.0;
        let ring = s.fill_us.load(Ordering::Relaxed) as f32 / 1000.0;
        let algo = s.algo_samples.load(Ordering::Relaxed) as f32 / FS * 1000.0;
        let outb = s.out_frames.load(Ordering::Relaxed) as f32 / self.info.out_rate.max(1) as f32 * 1000.0;
        inb + ring + algo + 0.7 + outb
    }

    /// 只算算法本身的延时（毫秒）。
    pub fn algo_ms(&self) -> f32 {
        self.shared.algo_samples.load(Ordering::Relaxed) as f32 / FS * 1000.0
    }
}

/// 命令行自检：用默认设备跑几秒，打印延迟和处理时间。
pub fn selftest(seconds: f32, opts: &RtOptions, cfg: Arc<EngineConfig>, file: Option<std::path::PathBuf>) -> Result<(), String> {
    let rt = match file {
        Some(f) => {
            let mut o = crate::offline::ConvertOptions::new(f);
            o.duration = Some(seconds as f64 + 2.0);
            let (mono, _) = crate::offline::load_mono(&o, &AtomicBool::new(false))?;
            Realtime::start_preview(opts, cfg, Arc::new(mono), 0)?
        }
        None => Realtime::start(opts, cfg)?,
    };
    println!("输入：{}（{} Hz，{} 声道）\n输出：{}（{} Hz，{} 声道）", rt.info.input_name, rt.info.in_rate, rt.info.in_channels, rt.info.output_name, rt.info.out_rate, rt.info.out_channels);
    let steps = (seconds / 0.5).ceil() as usize;
    for _ in 0..steps {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let t = rt.shared.telemetry();
        println!(
            "  延迟约 {:.1} ms（算法 {:.1} ms，缓冲积压 {:.1} ms，输入 {} 帧 / 输出 {} 帧）；每次回调处理 {} µs（最大 {} µs）；断流 {} 次；输入 {:.0} dB，输出 {:.0} dB，方位 {:.0}°，距离 {:.2} m",
            rt.latency_ms(),
            rt.algo_ms(),
            rt.shared.fill_us.load(Ordering::Relaxed) as f32 / 1000.0,
            rt.shared.in_frames.load(Ordering::Relaxed),
            rt.shared.out_frames.load(Ordering::Relaxed),
            rt.shared.proc_us.load(Ordering::Relaxed),
            rt.shared.proc_max_us.load(Ordering::Relaxed),
            rt.shared.xruns.load(Ordering::Relaxed),
            t.in_db,
            t.out_db,
            t.az,
            t.r
        );
        if let Some(e) = rt.shared.take_error() {
            return Err(e);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::params::Params;

    #[test]
    fn stream_xruns_are_counted_without_stopping_audio() {
        let shared = Shared::new(Arc::new(EngineConfig::new(Params::default(), None)));
        for device in ["输入", "输出"] {
            shared.stream_error(device, cpal::ErrorKind::Xrun.into());
        }
        assert_eq!(shared.xruns.load(Ordering::Relaxed), 2);
        assert!(shared.take_error().is_none());
    }

    #[test]
    fn stream_xruns_do_not_hide_a_fatal_device_error() {
        let shared = Shared::new(Arc::new(EngineConfig::new(Params::default(), None)));
        for device in ["输入", "输出"] {
            shared.stream_error(device, cpal::ErrorKind::DeviceNotAvailable.into());
            shared.stream_error(device, cpal::ErrorKind::Xrun.into());
            let error = shared.take_error().expect("disconnected devices must still stop audio");
            assert!(error.starts_with(&format!("{device}设备出错：")));
            assert!(error.contains(&cpal::ErrorKind::DeviceNotAvailable.to_string()));
            assert!(shared.take_error().is_none());
        }
        assert_eq!(shared.xruns.load(Ordering::Relaxed), 2);
    }

    fn preview_pump(audio: Vec<f32>, out_rate: f64, dry: bool) -> (Pump, Arc<Shared>) {
        let params = Params { auto_level: false, ..Params::default() };
        let cfg = Arc::new(EngineConfig::new(params, None));
        let shared = Arc::new(Shared::new(cfg.clone()));
        shared.monitor_dry.store(dry, Ordering::Relaxed);
        let pump = Pump::new(Source::Memory { data: Arc::new(audio), pos: 0 }, Engine::new(cfg, PostMode::Realtime, 0.0), shared.clone(), out_rate);
        (pump, shared)
    }

    #[test]
    fn preview_seeks_both_directions_and_restarts_after_eof() {
        for rate in [48_000.0, 44_100.0] {
            let mut audio = vec![0.2; FS_USIZE];
            audio.extend(vec![-0.4; FS_USIZE]);
            let (mut pump, shared) = preview_pump(audio, rate, true);
            // 非整块长度使队列残留旧音频，跳转时必须清掉。
            pump.fill(&mut [0.0; 137], &mut [0.0; 137]);
            shared.seek_preview((FS_USIZE / 2) as u64);
            shared.seek_preview(FS_USIZE as u64); // 连续拖动只执行最新位置。
            let (mut l, mut r) = ([0.0; 1024], [0.0; 1024]);
            pump.fill(&mut l, &mut r);
            assert!(l[512..].iter().chain(&r[512..]).all(|x| (*x + 0.4).abs() < 1e-4));
            assert!((FS_USIZE as u64..FS_USIZE as u64 + 2048).contains(&shared.play_pos.load(Ordering::Relaxed)));

            shared.seek_preview(100);
            pump.fill(&mut l, &mut r);
            assert!(l[512..].iter().all(|x| (*x - 0.2).abs() < 1e-4));
            assert!(shared.play_pos.load(Ordering::Relaxed) < 2048);

            shared.seek_preview(u64::MAX); // 超出长度的跳转截到结尾。
            for _ in 0..60 {
                pump.fill(&mut l, &mut r);
            }
            assert!(shared.preview_finished());
            shared.seek_preview(0);
            assert!(!shared.preview_finished());
            pump.fill(&mut l, &mut r);
            assert!(!shared.preview_finished());
            assert!(l[512..].iter().all(|x| (*x - 0.2).abs() < 1e-4));
        }
    }

    #[test]
    fn effect_parameters_change_during_preview_without_restarting() {
        let audio: Vec<f32> = (0..FS_USIZE * 3).map(|n| 0.01 * (n as f32 * std::f32::consts::TAU * 440.0 / FS).sin()).collect();
        let (mut pump, shared) = preview_pump(audio, 48_000.0, false);
        let (mut l, mut r) = ([0.0; 1024], [0.0; 1024]);
        for _ in 0..30 {
            pump.fill(&mut l, &mut r);
        }
        let before = l.iter().chain(&r).map(|x| x * x).sum::<f32>();
        let position = shared.play_pos.load(Ordering::Relaxed);
        let cfg = shared.cfg.lock().unwrap().clone();
        let mut params = cfg.params.clone();
        params.out_gain_db -= 24.0;
        shared.set_config(Arc::new(EngineConfig::new(params, Some(&cfg))));
        for _ in 0..4 {
            pump.fill(&mut l, &mut r);
        }
        let after = l.iter().chain(&r).map(|x| x * x).sum::<f32>();
        assert!(after < before * 0.02, "gain did not change: {before} -> {after}");
        assert!(shared.play_pos.load(Ordering::Relaxed) > position);
        assert!(!shared.preview_finished());
    }

    #[test]
    fn mix_preview_keeps_sources_apart_follows_changes_and_seeks() {
        use crate::dsp::mix::{voice_configs, Voice};
        let energy = |v: &[f32]| v.iter().map(|x| (x * x) as f64).sum::<f64>() + 1e-30;
        // 左耳的音源比右耳的短、而且是静音；右耳的在说话：输出几乎都在右耳，总长按长的那段
        let x = crate::cli::test_voice(8.0);
        let len = x.len();
        let base = EngineConfig::new(Params { auto_level: false, ..Params::default() }, None);
        let setup = |v: [Voice; 2]| Arc::new(MixSetup { cfgs: voice_configs(&base, &v, None), gains: vec![1.0, 1.0], offsets: vec![0.0; 2], follow: false, balance: None, balance_strength: 0.0 });
        let first = setup([Voice::preset(0), Voice::preset(1)]);
        let shared = Arc::new(Shared::new(first.cfgs[0].clone()));
        *shared.mix.lock().unwrap() = Some(first.clone());
        let src = Source::Multi { data: vec![Arc::new(vec![0.0; len / 2]), Arc::new(x)], pos: 0, len };
        let mut pump = Pump::with_proc(src, Proc::Mix(Mixer::new(&first, PostMode::Realtime, 0.0)), shared.clone(), 48_000.0);
        assert_eq!(shared.play_len.load(Ordering::Relaxed), len as u64);
        let (mut l, mut r) = (vec![0f32; 1024], vec![0f32; 1024]);
        let mut run = |pump: &mut Pump, blocks: usize| {
            let (mut el, mut er) = (0f64, 0f64);
            for _ in 0..blocks {
                pump.fill(&mut l, &mut r);
                el += energy(&l);
                er += energy(&r);
            }
            10.0 * (er / el).log10()
        };
        let right_over_left = run(&mut pump, 140); // 约 3 秒
        assert!(right_over_left > 20.0, "右耳应比左耳响得多：{right_over_left:.1} dB");
        let vt = shared.voice_telemetry();
        assert_eq!(vt.len(), 2);
        assert!(vt[0][0] > 180.0 && vt[1][0] < 180.0, "{vt:?}");
        // 播放中把两个音源对调：说话的那个挪到左耳，不用重新开始；跳回开头对两个音源同时生效
        shared.set_mix(setup([Voice::preset(1), Voice::preset(0)]));
        shared.seek_preview(0);
        run(&mut pump, 1);
        assert!(shared.play_pos.load(Ordering::Relaxed) < 2048);
        run(&mut pump, 90); // 等它挪过去
        let after = run(&mut pump, 90);
        assert!(after < -10.0, "挪到左耳后应左耳更响：右比左 {after:.1} dB");
        assert!(shared.voice_telemetry()[1][0] > 180.0);
        // 播到最长那段的结尾（再加 1 秒尾音）就结束
        shared.seek_preview(u64::MAX);
        run(&mut pump, 60);
        assert!(shared.preview_finished());
    }

    #[test]
    fn mix_preview_follow_mode_keeps_voices_opposite() {
        use crate::dsp::mix::Voice;
        // 一起移动的试听：两个音源始终在头的两侧相对；中途改移动方式（穿梭 → 环绕）也一样
        let x = crate::cli::test_voice(16.0);
        let len = x.len();
        let base = Arc::new(EngineConfig::new(Params { auto_level: false, ..Params::default() }, None));
        let voices = [Voice::preset(0), Voice::preset(1)];
        let st = Arc::new(MixSetup::new(&base, &voices, &[None, None], true, None));
        let shared = Arc::new(Shared::new(base.clone()));
        *shared.mix.lock().unwrap() = Some(st.clone());
        let audio = Arc::new(x);
        let src = Source::Multi { data: vec![audio.clone(), audio], pos: 0, len };
        let mut pump = Pump::with_proc(src, Proc::Mix(Mixer::new(&st, PostMode::Realtime, 0.0)), shared.clone(), 44_100.0);
        let (mut l, mut r) = (vec![0f32; 512], vec![0f32; 512]);
        let mut min_gap = 180.0f32;
        let gap = |v: &[[f32; 4]]| ((v[0][0] - v[1][0] + 180.0).rem_euclid(360.0) - 180.0).abs();
        for i in 0..1200 {
            if i == 600 {
                let p = Params { mode: crate::dsp::params::MotionMode::Orbit, auto_level: false, ..Params::default() };
                let orbit = Arc::new(EngineConfig::new(p, Some(&base)));
                shared.set_mix(Arc::new(MixSetup::new(&orbit, &voices, &[None, None], true, Some(&st))));
            }
            pump.fill(&mut l, &mut r);
            let vt = shared.voice_telemetry();
            if i > 5 {
                min_gap = min_gap.min(gap(&vt));
            }
        }
        assert!(min_gap > 150.0, "最近时只相隔 {min_gap:.0}°");
    }

    /// 用虚拟时钟模拟"输入回调每次 `n_in` 帧、输出回调每次 `n_out` 帧、输出设备时钟快 `drift`"，
    /// 返回 (断流次数, 稳定后的平均积压毫秒, 监听原声时测到的端到端延迟样本数)。
    fn simulate(n_in: usize, n_out: usize, drift: f64, seconds: f64, out_rate: f64) -> (u32, f32, i64) {
        let cfg = Arc::new(EngineConfig::new(Params::default(), None));
        let shared = Arc::new(Shared::new(cfg.clone()));
        shared.monitor_dry.store(true, Ordering::Relaxed);
        let (tx, rx) = spsc::channel(FS_USIZE);
        let mut sink = mic_sink(FS_USIZE as u32, tx, shared.clone());
        let mut pump = Pump::new(Source::Ring(rx), Engine::new(cfg, PostMode::Realtime, 0.0), shared.clone(), out_rate);
        let (t_in, t_out) = (n_in as f64 / FS as f64, n_out as f64 / (out_rate * (1.0 + drift)));
        let (mut next_in, mut next_out) = (0.0f64, 0.0005f64);
        let (mut n_fed, mut n_got) = (0usize, 0usize);
        let mut inb = vec![0f32; n_in];
        let (mut ol, mut or) = (vec![0f32; n_out], vec![0f32; n_out]);
        let click_in = (FS as f64 * seconds * 0.8) as usize; // 输入里放一个脉冲，测延迟
        let mut click_out: i64 = -1;
        let mut xr_after_start = 0;
        let mut fills = Vec::new();
        while next_in < seconds {
            if next_in <= next_out {
                for (i, v) in inb.iter_mut().enumerate() {
                    *v = if n_fed + i == click_in { 1.0 } else { 0.0 };
                }
                sink(&inb);
                n_fed += n_in;
                next_in += t_in;
            } else {
                pump.fill(&mut ol, &mut or);
                if click_out < 0 {
                    if let Some(i) = ol.iter().position(|v| v.abs() > 0.5) {
                        click_out = (n_got + i) as i64;
                    }
                }
                n_got += n_out;
                next_out += t_out;
                if next_out > 1.0 {
                    fills.push(shared.fill_us.load(Ordering::Relaxed) as f32 / 1000.0);
                    if next_out > 2.0 && xr_after_start == 0 {
                        xr_after_start = shared.xruns.load(Ordering::Relaxed);
                    }
                }
            }
        }
        let xr = shared.xruns.load(Ordering::Relaxed) - xr_after_start;
        let avg = fills.iter().rev().take(200).sum::<f32>() / 200.0;
        // 输出脉冲的位置换算回 48 kHz 的输入时间轴
        let delay = if click_out >= 0 { (click_out as f64 * FS as f64 / (out_rate * (1.0 + drift))) as i64 - click_in as i64 + (0.0005 * FS as f64) as i64 } else { -1 };
        (xr, avg, delay)
    }

    #[test]
    fn steady_when_clocks_match() {
        for (n_in, n_out) in [(128, 128), (256, 256), (480, 512), (512, 128), (441, 480)] {
            let (xr, fill_ms, delay) = simulate(n_in, n_out, 0.0, 12.0, 48_000.0);
            assert_eq!(xr, 0, "in {n_in} out {n_out}: xruns {xr}");
            println!("输入 {n_in} 帧 / 输出 {n_out} 帧：缓冲积压 {fill_ms:.1} ms，监听原声的端到端延迟 {:.1} ms", delay as f32 / 48.0);
            let target = n_in as f32 / 48.0 + 2.0;
            assert!(fill_ms > 0.0 && fill_ms < target + 4.0 + n_out as f32 / 48.0, "in {n_in} out {n_out}: fill {fill_ms} ms");
            assert!(delay > 0 && (delay as f32) < (fill_ms + 2.0 * n_out as f32 / 48.0 + n_in as f32 / 48.0 + 3.0) * 48.0, "in {n_in} out {n_out}: delay {delay}");
        }
    }

    #[test]
    fn drift_is_absorbed() {
        // 输出设备的时钟快 / 慢 0.1%（比真实声卡之间的偏差大得多），跑 60 秒不应断流、积压不应越来越多
        for drift in [0.001, -0.001] {
            let (xr, fill_ms, _) = simulate(256, 256, drift, 60.0, 48_000.0);
            assert_eq!(xr, 0, "drift {drift}: xruns {xr}");
            assert!(fill_ms > 1.0 && fill_ms < 20.0, "drift {drift}: fill {fill_ms} ms");
        }
    }

    #[test]
    fn other_output_rates_work() {
        let (xr, fill_ms, delay) = simulate(256, 256, 0.0, 12.0, 44_100.0);
        assert_eq!(xr, 0);
        assert!(fill_ms < 15.0 && delay > 0, "fill {fill_ms} delay {delay}");
    }
}
