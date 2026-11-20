//! 文件模式：解码任意常见格式 → 取一个声道 → 48 kHz → 引擎（带约 1 秒预读）→ 精确响度归一化和真峰值限制 → WAV。
//! 多音频合成（[`convert_mix`]）：几个文件各过一个引擎、各在一个位置，相加之后同样两遍处理。

use crate::dsp::analysis::prescan;
use crate::dsp::dynamics::{Limiter, LoudnessMeter, TruePeak};
use crate::dsp::engine::{Engine, EngineConfig, PostMode};
use crate::dsp::mix::{self, MixSetup, Mixer, Voice, MAX_VOICES};
use crate::dsp::params::Params;
use crate::dsp::resample::Resampler;
use crate::dsp::{db_to_lin, FS, FS_USIZE};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// 立体声输入取哪个声道。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Auto,
    Left,
    Right,
    Mix,
}

#[derive(Clone, Debug)]
pub struct ConvertOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub start: f64,
    pub duration: Option<f64>,
    pub channel: Channel,
    pub bits: u16,
    pub lufs: f32,
    pub peak_db: f32,
    pub max_gr_db: f32,
    /// 把实际走位写成 CSV（时间, 方位角, 距离, 不含忽远忽近的距离, 说话活动, 仰角, 入耳深度）。
    pub save_trajectory: Option<PathBuf>,
}

impl ConvertOptions {
    pub fn new(input: PathBuf) -> Self {
        let output = default_output(&input);
        ConvertOptions { input, output, start: 0.0, duration: None, channel: Channel::Auto, bits: 24, lufs: -16.0, peak_db: -1.0, max_gr_db: 8.0, save_trajectory: None }
    }
}

/// 默认输出文件名：输入名_asmr.wav。
pub fn default_output(input: &Path) -> PathBuf {
    let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "output".into());
    input.with_file_name(format!("{stem}_asmr.wav"))
}

/// 多音频合成的默认输出文件名：放在第一个音频旁边，名字是"第一个音频名_mix_asmr.wav"
/// （以 _asmr 结尾，添加文件夹时会被当作本程序的输出跳过）。
pub fn default_mix_output(first: &Path) -> PathBuf {
    let stem = first.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "output".into());
    first.with_file_name(format!("{stem}_mix_asmr.wav"))
}

/// 多音频合成的一个音源：哪个文件、放在哪里、音量增减。
#[derive(Clone, Debug)]
pub struct MixInput {
    pub path: PathBuf,
    pub voice: Voice,
}

#[derive(Clone, Debug, Default)]
pub struct ConvertReport {
    pub seconds: f64,
    pub lufs: f32,
    pub true_peak_db: f32,
    pub max_gr_db: f32,
    pub channel_note: String,
    pub n_moves: u32,
    pub n_sways: u32,
    /// 从头顶绕的次数、上下换位的次数。
    pub n_tops: u32,
    pub n_updowns: u32,
    /// 伸进耳道的次数（入耳深度大于 0 时）。
    pub n_strokes: u32,
    pub speech_pct: f32,
    pub elapsed_s: f64,
}

/// 逐包解码，把每个包交给 `sink(通道数, 采样率, 交织样本)`；返回 false 时提前停止。
fn decode(path: &Path, mut sink: impl FnMut(usize, u32, &[f32]) -> bool) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("打不开 {}：{e}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("不认识的音频格式：{e}"))?;
    let mut format = probed.format;
    let track = format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL).ok_or("文件里没有音轨")?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default()).map_err(|e| format!("不支持的编码：{e}"))?;
    let mut sbuf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(format!("读取失败：{e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                let spec = *buf.spec();
                let need = buf.capacity() as u64;
                if sbuf.as_ref().map_or(true, |s| (s.capacity() as u64) < need * spec.channels.count() as u64) {
                    sbuf = Some(SampleBuffer::<f32>::new(need, spec));
                }
                let sb = sbuf.as_mut().unwrap();
                sb.copy_interleaved_ref(buf);
                if !sink(spec.channels.count(), spec.rate, sb.samples()) {
                    break;
                }
            }
            Err(SymError::DecodeError(_)) => continue,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("解码失败：{e}")),
        }
    }
    Ok(())
}

/// 读入音频并取成 48 kHz 单声道。返回 (样本, 关于声道选择的说明)。
pub fn load_mono(opts: &ConvertOptions, cancel: &AtomicBool) -> Result<(Vec<f32>, String), String> {
    // 第一遍：只统计左右声道的能量和相关性，决定取哪个声道
    let mut note = String::new();
    let mut pick = opts.channel;
    let (mut ll, mut rr, mut lr, mut nch) = (0f64, 0f64, 0f64, 0usize);
    if opts.channel == Channel::Auto {
        // 只统计要处理的那一段（与 Python 版一致：截取之后再比较两个声道）
        let mut seen = 0u64;
        decode(&opts.input, |ch, sr, s| {
            nch = ch;
            if ch < 2 {
                return false;
            }
            let i0 = (opts.start * sr as f64) as u64;
            let i1 = opts.duration.map(|d| i0 + (d * sr as f64) as u64);
            for f in s.chunks_exact(ch) {
                if seen >= i0 && i1.map_or(true, |e| seen < e) {
                    ll += (f[0] as f64) * (f[0] as f64);
                    rr += (f[1] as f64) * (f[1] as f64);
                    lr += (f[0] as f64) * (f[1] as f64);
                }
                seen += 1;
            }
            !cancel.load(Ordering::Relaxed) && i1.map_or(true, |e| seen < e)
        })?;
        if nch >= 2 {
            let corr = lr / ((ll * rr).sqrt() + 1e-30);
            let left = ll >= rr;
            pick = if left { Channel::Left } else { Channel::Right };
            note = format!("立体声输入，相关 {:.2}，取{}声道", corr, if left { "左" } else { "右" });
            let (big, small) = if left { (ll, rr) } else { (rr, ll) };
            if small > 0.25 * big && corr < 0.5 {
                note.push_str("（两个声道内容差别较大，只取一个会丢掉另一侧的内容，可改选「混合」）");
            }
        }
    }
    let mut mono: Vec<f32> = Vec::new();
    let mut rs: Option<Resampler> = None;
    let mut rate = 0u32;
    let mut seen = 0u64;
    let mut tmp: Vec<f32> = Vec::new();
    decode(&opts.input, |ch, sr, s| {
        if rate == 0 {
            rate = sr;
            if sr as usize != FS_USIZE {
                rs = Some(Resampler::new(sr as f64, FS as f64));
            }
        }
        let i0 = (opts.start * sr as f64) as u64;
        let i1 = opts.duration.map(|d| i0 + (d * sr as f64) as u64 + 64);
        tmp.clear();
        for f in s.chunks_exact(ch.max(1)) {
            let keep = seen >= i0 && i1.map_or(true, |e| seen < e);
            seen += 1;
            if !keep {
                continue;
            }
            let v = if ch < 2 {
                f[0]
            } else {
                match pick {
                    Channel::Right => f[1],
                    Channel::Mix => 0.5 * (f[0] + f[1]),
                    _ => f[0],
                }
            };
            tmp.push(v);
        }
        match &mut rs {
            Some(r) => r.process(&tmp, &mut mono),
            None => mono.extend_from_slice(&tmp),
        }
        !cancel.load(Ordering::Relaxed) && i1.map_or(true, |e| seen < e)
    })?;
    if let Some(d) = opts.duration {
        mono.truncate((d * FS as f64) as usize);
    }
    if mono.len() < FS_USIZE / 10 {
        return Err("输入太短".into());
    }
    Ok((mono, note))
}

struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// 第一遍每次处理多少样本。
const CH: usize = 4096;

/// 第一遍的产物：临时文件里是去掉开头时延之后的交错 f32 立体声，外加测好的积分响度和采样峰值。
struct Rendered {
    tmp: TempFile,
    written: usize,
    peak: f32,
    meter: LoudnessMeter,
}

/// 第一遍：从第 0 个样本起逐块调用 `render(起点, 左, 右)`，一共渲染 `total` 个样本；去掉开头 `skip` 个
/// （预读和降噪带来的时延）后写进临时文件，同时测响度和峰值。
fn render_to_temp(total: usize, skip: usize, cancel: &AtomicBool, progress: &mut impl FnMut(f32, &str), mut render: impl FnMut(usize, &mut [f32], &mut [f32])) -> Result<Rendered, String> {
    // 批量转换时多个线程同时转换：用进程内递增的序号保证临时文件名不重复
    static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp_path = std::env::temp_dir().join(format!("auto_asmr_{}_{}.f32", std::process::id(), TMP_SEQ.fetch_add(1, Ordering::Relaxed)));
    let tmp = TempFile(tmp_path.clone());
    let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp_path).map_err(|e| format!("无法创建临时文件：{e}"))?);
    let mut meter = LoudnessMeter::new();
    let mut peak = 0f32;
    let (mut ol, mut or) = (vec![0f32; CH], vec![0f32; CH]);
    let mut done = 0usize;
    let mut written = 0usize;
    let mut bytes = Vec::with_capacity(CH * 8);
    while done < total {
        if cancel.load(Ordering::Relaxed) {
            return Err("已取消".into());
        }
        let m = CH.min(total - done);
        render(done, &mut ol[..m], &mut or[..m]);
        bytes.clear();
        for i in 0..m {
            if done + i < skip {
                continue;
            }
            let (l, r) = (ol[i], or[i]);
            meter.push(l, r);
            peak = peak.max(l.abs()).max(r.abs());
            bytes.extend_from_slice(&l.to_le_bytes());
            bytes.extend_from_slice(&r.to_le_bytes());
            written += 1;
        }
        w.write_all(&bytes).map_err(|e| format!("写临时文件失败：{e}"))?;
        done += m;
        progress(0.05 + 0.75 * done as f32 / total as f32, "空间化");
    }
    w.flush().map_err(|e| e.to_string())?;
    drop(w);
    Ok(Rendered { tmp, written, peak, meter })
}

/// 第二遍的结果。
struct Finished {
    seconds: f64,
    lufs: f32,
    true_peak_db: f32,
    max_gr_db: f32,
}

/// 去掉尾部多余的静音，第二遍精确响度归一化 + 真峰值限制，写出 WAV。`n_sig`：输入（最长的那个）的样本数。
fn finish(opts: &ConvertOptions, r: Rendered, n_sig: usize, cancel: &AtomicBool, progress: &mut impl FnMut(f32, &str)) -> Result<Finished, String> {
    let Rendered { tmp, written, peak, meter } = r;
    let cancelled = || cancel.load(Ordering::Relaxed);
    // 去掉尾部多余的静音：保留到最后一个高于最大值 −100 dB 的样本
    let mut rd = BufReader::with_capacity(1 << 20, File::open(&tmp.0).map_err(|e| e.to_string())?);
    let mut n_out = n_sig.min(written);
    if written > n_sig {
        rd.seek(SeekFrom::Start((n_sig * 8) as u64)).map_err(|e| e.to_string())?;
        let mut b = vec![0u8; (written - n_sig) * 8];
        rd.read_exact(&mut b).map_err(|e| e.to_string())?;
        for (i, f) in b.chunks_exact(4).enumerate() {
            if f32::from_le_bytes(f.try_into().unwrap()).abs() > 1e-5 * peak {
                n_out = n_sig + i / 2 + 1;
            }
        }
        rd.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    }

    // 第二遍：响度归一化；峰值太高时用前瞻限制器最多压 max_gr dB，再不够就整体降低
    let mut gain = db_to_lin(meter.gain_db_for(opts.lufs));
    let lim = db_to_lin(opts.peak_db);
    if peak * gain > lim {
        let need = 20.0 * (peak * gain / lim).log10();
        if need > opts.max_gr_db {
            gain *= db_to_lin(-(need - opts.max_gr_db));
        }
    }
    let spec = hound::WavSpec { channels: 2, sample_rate: FS_USIZE as u32, bits_per_sample: opts.bits, sample_format: hound::SampleFormat::Int };
    let mut wav = hound::WavWriter::create(&opts.output, spec).map_err(|e| format!("无法写入 {}：{e}", opts.output.display()))?;
    let mut limiter = Limiter::new(6.0, 40.0);
    let ldelay = limiter.delay();
    let mut tp = TruePeak::new();
    let mut out_meter = LoudnessMeter::new();
    let full = if opts.bits == 16 { 32767.0 } else { 8_388_607.0 };
    let mut buf = vec![0u8; CH * 8];
    let mut fed = 0usize;
    let mut emitted = 0usize;
    let ceiling = lim * 0.97;
    let total2 = n_out + ldelay; // 末尾多送一段 0，把限制器里延时的样本推出来
    let pass2 = (|| -> Result<(), String> {
        while fed < total2 {
            if cancelled() {
                return Err("已取消".into());
            }
            let m = CH.min(total2 - fed);
            let avail = m.min(n_out.saturating_sub(fed));
            if avail > 0 {
                rd.read_exact(&mut buf[..avail * 8]).map_err(|e| e.to_string())?;
            }
            for i in 0..m {
                let (l, r) = if i < avail {
                    let o = i * 8;
                    (f32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) * gain, f32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap()) * gain)
                } else {
                    (0.0, 0.0)
                };
                let (a, b) = limiter.process(l, r, ceiling);
                if fed + i >= ldelay {
                    tp.push(a, b);
                    out_meter.push(a, b);
                    wav.write_sample((a.clamp(-1.0, 1.0) * full).round() as i32).map_err(|e| e.to_string())?;
                    wav.write_sample((b.clamp(-1.0, 1.0) * full).round() as i32).map_err(|e| e.to_string())?;
                    emitted += 1;
                }
            }
            fed += m;
            progress(0.8 + 0.2 * emitted as f32 / n_out.max(1) as f32, "响度归一化");
        }
        wav.finalize().map_err(|e| e.to_string())
    })();
    if let Err(e) = pass2 {
        // 取消或出错：不留下写了一半的输出文件
        let _ = std::fs::remove_file(&opts.output);
        return Err(e);
    }
    drop(tmp);
    progress(1.0, "完成");
    Ok(Finished { seconds: n_out as f64 / FS as f64, lufs: out_meter.integrated(), true_peak_db: 20.0 * tp.peak.max(1e-9).log10(), max_gr_db: limiter.max_gr_db })
}

/// 转换一个文件。`progress(0–1, 阶段说明)`；`cancel` 置位时中止。
pub fn convert(opts: &ConvertOptions, params: &Params, cancel: Arc<AtomicBool>, mut progress: impl FnMut(f32, &str)) -> Result<ConvertReport, String> {
    let t_start = std::time::Instant::now();
    crate::dsp::flush_denormals();
    progress(0.0, "读取音频");
    let (mono, note) = load_mono(opts, &cancel)?;
    let n_sig = mono.len();
    if cancel.load(Ordering::Relaxed) {
        return Err("已取消".into());
    }

    // 文件模式：响度和限幅放到第二遍，引擎里不做自动响度
    let cfg = Arc::new(EngineConfig::new(params.clone(), None));
    let lookahead = 1.0;
    let mut eng = Engine::new(cfg, PostMode::Offline, lookahead);
    if let Some((thr, med)) = prescan(&mono) {
        eng.set_fixed_analysis(thr, med);
    }
    if opts.save_trajectory.is_some() {
        eng.trace = Some(Vec::new());
    }
    let skip = eng.latency_samples(); // 预读（和降噪）带来的时延，输出时去掉
    let tail = if params.room { (0.65 * FS) as usize } else { 512 };
    let total = n_sig + skip + tail;
    let mut inp = vec![0f32; CH];
    let rendered = render_to_temp(total, skip, &cancel, &mut progress, |done, ol, or| {
        let m = ol.len();
        for (i, v) in inp[..m].iter_mut().enumerate() {
            *v = if done + i < n_sig { mono[done + i] } else { 0.0 };
        }
        eng.process(&inp[..m], ol, or);
    })?;
    if let (Some(path), Some(tr)) = (&opts.save_trajectory, &eng.trace) {
        let mut f = BufWriter::new(File::create(path).map_err(|e| format!("无法写入轨迹文件：{e}"))?);
        writeln!(f, "t_sec,az_deg,dist_m,base_dist_m,speech,el_deg,inear_depth_m").map_err(|e| e.to_string())?;
        for row in tr.iter().step_by(8) {
            writeln!(f, "{:.4},{:.2},{:.5},{:.5},{:.3},{:.2},{:.5}", row[0], row[1], row[2], row[3], row[4], row[5], row[6]).map_err(|e| e.to_string())?;
        }
    }
    let fin = finish(opts, rendered, n_sig, &cancel, &mut progress)?;
    Ok(ConvertReport {
        seconds: fin.seconds,
        lufs: fin.lufs,
        true_peak_db: fin.true_peak_db,
        max_gr_db: fin.max_gr_db,
        channel_note: note,
        n_moves: eng.tele.n_moves,
        n_sways: eng.tele.n_sways,
        n_tops: eng.tele.n_tops,
        n_updowns: eng.tele.n_updowns,
        n_strokes: eng.tele.n_strokes,
        speech_pct: 100.0 * eng.n_speech as f32 / eng.n_blocks.max(1) as f32,
        elapsed_s: t_start.elapsed().as_secs_f64(),
    })
}

/// 多音频合成：`inputs` 里的 2–[`MAX_VOICES`] 个文件各放一个位置（见 [`crate::dsp::mix`]），同时播放，合成一个
/// 双耳 WAV 写到 `opts.output`。`opts.input` 不用；起点、时长、取哪个声道、位深、目标响度、峰值上限对每个音源都一样。
/// 输出的长度按最长的那个音源；各音源先按响度对齐，再加各自的音量增减，相加之后与单个文件一样两遍处理。
/// `follow`：一起移动（按 `params` 的移动方式一起走、彼此错开 360°/N）；否则各自固定在 `inputs` 给的位置。
pub fn convert_mix(opts: &ConvertOptions, inputs: &[MixInput], params: &Params, follow: bool, cancel: Arc<AtomicBool>, mut progress: impl FnMut(f32, &str)) -> Result<ConvertReport, String> {
    let n = inputs.len();
    if n < 2 {
        return Err("多音频合成至少需要两个音频".into());
    }
    if n > MAX_VOICES {
        return Err(format!("多音频合成最多 {MAX_VOICES} 个音频"));
    }
    let out = crate::batch::normalize(&opts.output);
    if let Some(i) = inputs.iter().find(|i| crate::batch::normalize(&i.path) == out) {
        return Err(format!("输出文件不能是输入文件：{}", i.path.display()));
    }
    let t_start = std::time::Instant::now();
    crate::dsp::flush_denormals();
    let mut monos = Vec::with_capacity(n);
    let mut notes = Vec::new();
    for (k, inp) in inputs.iter().enumerate() {
        progress(0.05 * k as f32 / n as f32, "读取音频");
        let mut o = opts.clone();
        o.input = inp.path.clone();
        let name = inp.path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let (m, note) = load_mono(&o, &cancel).map_err(|e| format!("{name}：{e}"))?;
        if !note.is_empty() {
            notes.push(format!("{}. {name}：{note}", k + 1));
        }
        monos.push(m);
        if cancel.load(Ordering::Relaxed) {
            return Err("已取消".into());
        }
    }
    let voices: Vec<Voice> = inputs.iter().map(|i| i.voice).collect();
    let lufs: Vec<Option<f32>> = monos.iter().map(|m| mix::loudness(m)).collect();
    let base = Arc::new(EngineConfig::new(params.clone(), None));
    let setup = MixSetup::new(&base, &voices, &lufs, follow, None);
    let mut mixer = Mixer::new(&setup, PostMode::Offline, 1.0);
    for (k, m) in monos.iter().enumerate() {
        if let Some((thr, med)) = prescan(m) {
            mixer.engine_mut(k).set_fixed_analysis(thr, med);
        }
    }
    let n_sig = monos.iter().map(|m| m.len()).max().unwrap_or(0);
    let skip = mixer.latency_samples();
    let tail = if params.room { (0.65 * FS) as usize } else { 512 };
    let total = n_sig + skip + tail;
    let mut bufs = vec![vec![0f32; CH]; n];
    let rendered = render_to_temp(total, skip, &cancel, &mut progress, |done, ol, or| {
        let m = ol.len();
        for (b, x) in bufs.iter_mut().zip(&monos) {
            for (i, v) in b[..m].iter_mut().enumerate() {
                *v = x.get(done + i).copied().unwrap_or(0.0);
            }
        }
        let ins: Vec<&[f32]> = bufs.iter().map(|b| &b[..m]).collect();
        mixer.process_parallel(&ins, ol, or);
    })?;
    let fin = finish(opts, rendered, n_sig, &cancel, &mut progress)?;
    let speech = (0..n).map(|k| 100.0 * mixer.engine(k).n_speech as f32 / mixer.engine(k).n_blocks.max(1) as f32).sum::<f32>() / n as f32;
    // 各自固定：各音源的往后让、上下换位次数加起来；一起移动：大家走同一套，报告一份
    let t = if follow { mixer.engine(0).tele } else { mixer.tele };
    Ok(ConvertReport {
        seconds: fin.seconds,
        lufs: fin.lufs,
        true_peak_db: fin.true_peak_db,
        max_gr_db: fin.max_gr_db,
        channel_note: notes.join("\n"),
        n_moves: t.n_moves,
        n_sways: t.n_sways,
        n_tops: t.n_tops,
        n_updowns: t.n_updowns,
        n_strokes: t.n_strokes,
        speech_pct: speech,
        elapsed_s: t_start.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::test_voice;

    fn write_wav(path: &Path, x: &[f32]) {
        let spec = hound::WavSpec { channels: 1, sample_rate: FS_USIZE as u32, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for &v in x {
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();
    }

    fn read_stereo(path: &Path) -> (Vec<f32>, Vec<f32>) {
        let mut r = hound::WavReader::open(path).unwrap();
        let s: Vec<f32> = r.samples::<i32>().map(|v| v.unwrap() as f32 / 8_388_607.0).collect();
        (s.iter().step_by(2).copied().collect(), s.iter().skip(1).step_by(2).copied().collect())
    }

    #[test]
    fn mix_puts_each_file_on_its_own_side() {
        let dir = std::env::temp_dir().join(format!("auto_asmr_mix_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A：前 5 秒说话，之后静音；B：比 A 长、轻 12 dB，前 5 秒静音、之后说话
        let v = test_voice(12.0);
        let a: Vec<f32> = v.iter().take(10 * FS_USIZE).enumerate().map(|(i, &x)| if i < 5 * FS_USIZE { x } else { 0.0 }).collect();
        let b: Vec<f32> = v.iter().enumerate().map(|(i, &x)| if i >= 5 * FS_USIZE { 0.25 * x } else { 0.0 }).collect();
        let (pa, pb, out) = (dir.join("a.wav"), dir.join("b.wav"), dir.join("mix.wav"));
        write_wav(&pa, &a);
        write_wav(&pb, &b);
        let mut opts = ConvertOptions::new(pa.clone());
        opts.output = out.clone();
        let inputs = [MixInput { path: pa.clone(), voice: Voice::preset(0) }, MixInput { path: pb.clone(), voice: Voice::preset(1) }];
        let rep = convert_mix(&opts, &inputs, &Params::default(), false, Arc::new(AtomicBool::new(false)), |_, _| {}).unwrap();
        // 长度按最长的 B；响度精确归一化到目标
        assert!((rep.seconds - 12.0).abs() < 0.05, "{}", rep.seconds);
        assert!((rep.lufs - opts.lufs).abs() < 0.5, "{}", rep.lufs);
        let (l, r) = read_stereo(&out);
        let e = |v: &[f32]| v.iter().map(|x| (x * x) as f64).sum::<f64>() + 1e-30;
        let s = |t: f32| (t * FS) as usize;
        let first = 10.0 * (e(&l[s(0.2)..s(4.8)]) / e(&r[s(0.2)..s(4.8)])).log10();
        let second = 10.0 * (e(&r[s(5.2)..s(11.8)]) / e(&l[s(5.2)..s(11.8)])).log10();
        assert!(first > 20.0 && second > 20.0, "A 在左 {first:.1} dB，B 在右 {second:.1} dB");
        // 两个音源响度已对齐：B 的输入轻 12 dB，输出里两段一样响（相差 2 dB 以内）
        let ld = 10.0 * ((e(&l[s(0.2)..s(4.8)]) + e(&r[s(0.2)..s(4.8)])) / 4.6 / ((e(&l[s(5.2)..s(11.8)]) + e(&r[s(5.2)..s(11.8)])) / 6.6)).log10();
        assert!(ld.abs() < 2.0, "两段响度差 {ld:.1} dB");
        // 一起移动：同样长度和响度；两个音源走同一套，所以报告的换耳次数就是一个音源的
        let mut fo = opts.clone();
        fo.output = dir.join("mix_follow.wav");
        let rf = convert_mix(&fo, &inputs, &Params::default(), true, Arc::new(AtomicBool::new(false)), |_, _| {}).unwrap();
        assert!((rf.seconds - 12.0).abs() < 0.05 && (rf.lufs - opts.lufs).abs() < 0.5, "{} s {} LUFS", rf.seconds, rf.lufs);
        assert!(rf.n_moves >= 1, "一起移动应当换过边：{}", rf.n_moves);
        assert_eq!(rep.n_moves, 0, "各自固定不换边");
        // 只给一个音源、输出覆盖输入：拒绝
        assert!(convert_mix(&opts, &inputs[..1], &Params::default(), false, Arc::new(AtomicBool::new(false)), |_, _| {}).is_err());
        let mut bad = opts.clone();
        bad.output = pb.clone();
        assert!(convert_mix(&bad, &inputs, &Params::default(), false, Arc::new(AtomicBool::new(false)), |_, _| {}).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
