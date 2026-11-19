//! 文件模式：解码任意常见格式 → 取一个声道 → 48 kHz → 引擎（带约 1 秒预读）→ 精确响度归一化和真峰值限制 → WAV。

use crate::dsp::analysis::prescan;
use crate::dsp::dynamics::{Limiter, LoudnessMeter, TruePeak};
use crate::dsp::engine::{Engine, EngineConfig, PostMode};
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
    /// 把实际走位写成 CSV（时间, 方位角, 距离, 不含忽远忽近的距离, 说话活动）。
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

#[derive(Clone, Debug, Default)]
pub struct ConvertReport {
    pub seconds: f64,
    pub lufs: f32,
    pub true_peak_db: f32,
    pub max_gr_db: f32,
    pub channel_note: String,
    pub n_moves: u32,
    pub n_sways: u32,
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

/// 转换一个文件。`progress(0–1, 阶段说明)`；`cancel` 置位时中止。
pub fn convert(opts: &ConvertOptions, params: &Params, cancel: Arc<AtomicBool>, mut progress: impl FnMut(f32, &str)) -> Result<ConvertReport, String> {
    let t_start = std::time::Instant::now();
    crate::dsp::flush_denormals();
    progress(0.0, "读取音频");
    let (mono, note) = load_mono(opts, &cancel)?;
    let n_sig = mono.len();
    let cancelled = || cancel.load(Ordering::Relaxed);
    if cancelled() {
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

    let tmp_path = std::env::temp_dir().join(format!("auto_asmr_{}_{}.f32", std::process::id(), t_start.elapsed().as_nanos()));
    let tmp = TempFile(tmp_path.clone());
    let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp_path).map_err(|e| format!("无法创建临时文件：{e}"))?);
    let mut meter = LoudnessMeter::new();
    let mut peak = 0f32;
    const CH: usize = 4096;
    let (mut inp, mut ol, mut or) = (vec![0f32; CH], vec![0f32; CH], vec![0f32; CH]);
    let mut done = 0usize;
    let mut written = 0usize;
    let mut bytes = Vec::with_capacity(CH * 8);
    while done < total {
        if cancelled() {
            return Err("已取消".into());
        }
        let m = CH.min(total - done);
        for (i, v) in inp[..m].iter_mut().enumerate() {
            *v = if done + i < n_sig { mono[done + i] } else { 0.0 };
        }
        eng.process(&inp[..m], &mut ol[..m], &mut or[..m]);
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
    if let (Some(path), Some(tr)) = (&opts.save_trajectory, &eng.trace) {
        let mut f = BufWriter::new(File::create(path).map_err(|e| format!("无法写入轨迹文件：{e}"))?);
        writeln!(f, "t_sec,az_deg,dist_m,base_dist_m,speech").map_err(|e| e.to_string())?;
        for row in tr.iter().step_by(8) {
            writeln!(f, "{:.4},{:.2},{:.5},{:.5},{:.3}", row[0], row[1], row[2], row[3], row[4]).map_err(|e| e.to_string())?;
        }
    }

    // 去掉尾部多余的静音：保留到最后一个高于最大值 −100 dB 的样本
    let mut rd = BufReader::with_capacity(1 << 20, File::open(&tmp_path).map_err(|e| e.to_string())?);
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
    wav.finalize().map_err(|e| e.to_string())?;
    drop(tmp);
    progress(1.0, "完成");
    Ok(ConvertReport {
        seconds: n_out as f64 / FS as f64,
        lufs: out_meter.integrated(),
        true_peak_db: 20.0 * tp.peak.max(1e-9).log10(),
        max_gr_db: limiter.max_gr_db,
        channel_note: note,
        n_moves: eng.tele.n_moves,
        n_sways: eng.tele.n_sways,
        speech_pct: 100.0 * eng.n_speech as f32 / eng.n_blocks.max(1) as f32,
        elapsed_s: t_start.elapsed().as_secs_f64(),
    })
}
