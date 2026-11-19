//! 批量转换：一组文件放进工作队列，按 CPU 核数开若干工作线程并行处理，每个线程一次完整转换一个文件。
//!
//! 单个文件的引擎是有状态的（走位、随机数、响度测量都依赖之前的样本），切成几段并行会改变结果，
//! 所以并行粒度是"文件"。为了让各线程尽量同时结束，先估计每个文件的时长，长的先处理（LPT 调度）：
//! 最长的文件不会被留到最后单独跑，其它核在它运行期间消化短文件。

use crate::dsp::params::Params;
use crate::offline::{convert, ConvertOptions, ConvertReport};
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// 文件选择框里列出的音频扩展名。
pub const AUDIO_EXTS: &[&str] = &["wav", "m4a", "mp3", "flac", "ogg", "opus", "aac", "mp4", "mkv", "mov", "webm", "aiff", "aif", "caf"];

pub fn is_audio(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).map_or(false, |e| AUDIO_EXTS.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// 本程序自己的输出（"原名_asmr.wav"、"原名_asmr_2.wav"……）。展开文件夹时跳过，免得把结果再处理一遍。
pub fn is_own_output(p: &Path) -> bool {
    let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    stem.ends_with("_asmr") || stem.rsplit_once("_asmr_").map_or(false, |(_, k)| !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit()))
}

/// 整理输入列表：文件原样保留；文件夹展开为其中（不含子文件夹）的音频文件，按名字排序，跳过本程序的输出；去掉重复项。
pub fn expand_inputs(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for p in paths {
        let mut add = |q: PathBuf| {
            if seen.insert(normalize(&q)) {
                out.push(q);
            }
        };
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p).into_iter().flatten().flatten().map(|e| e.path()).filter(|q| q.is_file() && is_audio(q) && !is_own_output(q)).collect();
            v.sort();
            v.into_iter().for_each(&mut add);
        } else {
            add(p.clone());
        }
    }
    out
}

/// 默认并行数：逻辑核数。每个工作线程只用一个核，所以等于核数时 CPU 正好占满。
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// 粗略估计音频时长（秒），只读文件头。拿不到帧数时按文件大小估计（约 128 kbps）。只用来排处理顺序和加权进度。
pub fn estimate_seconds(path: &Path) -> f64 {
    let by_size = || std::fs::metadata(path).map(|m| m.len() as f64 / 16_000.0).unwrap_or(60.0);
    let Ok(file) = File::open(path) else { return by_size() };
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let Ok(probed) = symphonia::default::get_probe().format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default()) else { return by_size() };
    probed
        .format
        .default_track()
        .and_then(|t| Some(t.codec_params.n_frames? as f64 / t.codec_params.sample_rate? as f64))
        .filter(|s| s.is_finite() && *s > 0.0)
        .unwrap_or_else(by_size)
}

/// 给每个输入安排输出文件名：`out_dir` 为 None 时放在源文件旁边，否则都放进 `out_dir`。
/// 名字是"原名_asmr.wav"；重名（同名文件来自不同文件夹，或输出会覆盖某个输入）时依次加 _2、_3……
pub fn plan_outputs(inputs: &[PathBuf], out_dir: Option<&Path>) -> Vec<PathBuf> {
    let inputs_set: HashSet<PathBuf> = inputs.iter().map(|p| normalize(p)).collect();
    let mut used: HashSet<PathBuf> = HashSet::new();
    inputs
        .iter()
        .map(|inp| {
            let stem = inp.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "output".into());
            let dir = out_dir.map(Path::to_path_buf).unwrap_or_else(|| inp.parent().map(Path::to_path_buf).unwrap_or_default());
            let mut k = 1;
            loop {
                let name = if k == 1 { format!("{stem}_asmr.wav") } else { format!("{stem}_asmr_{k}.wav") };
                let cand = dir.join(name);
                let key = normalize(&cand);
                if !used.contains(&key) && !inputs_set.contains(&key) {
                    used.insert(key);
                    break cand;
                }
                k += 1;
            }
        })
        .collect()
}

/// 规范化路径，用来判断两个路径是不是同一个文件（文件还不存在时规范化所在文件夹）。
pub fn normalize(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| {
        // 文件还不存在：把所在文件夹规范化后再接上文件名
        match (p.parent().and_then(|d| std::fs::canonicalize(if d.as_os_str().is_empty() { Path::new(".") } else { d }).ok()), p.file_name()) {
            (Some(d), Some(n)) => d.join(n),
            _ => p.to_path_buf(),
        }
    })
}

/// 每个任务实际要处理的时长估计（秒），已考虑起点和时长限制。
pub fn estimates(items: &[ConvertOptions]) -> Vec<f64> {
    items
        .iter()
        .map(|o| {
            let s = (estimate_seconds(&o.input) - o.start).max(0.0);
            o.duration.map_or(s, |d| d.min(s)).max(0.1)
        })
        .collect()
}

/// 并行转换一批文件。
/// - `est`：`estimates()` 的结果，用来排顺序（长的先做）；
/// - `threads`：工作线程数（会限制在 1 到文件数之间）；
/// - `progress(序号, 0–1, 阶段)`、`done(序号, 结果)` 在工作线程里调用，序号是 `items` 里的下标；
/// - `cancel` 置位后：正在转换的文件中止，还没开始的不再开始（报告"已取消"）。
pub fn run(
    items: &[ConvertOptions],
    est: &[f64],
    params: &Params,
    threads: usize,
    cancel: &Arc<AtomicBool>,
    progress: impl Fn(usize, f32, &str) + Sync,
    done: impl Fn(usize, Result<ConvertReport, String>) + Sync,
) {
    if items.is_empty() {
        return;
    }
    // 长的先做
    let mut order: Vec<(usize, f64)> = (0..items.len()).map(|i| (i, est.get(i).copied().unwrap_or(0.0))).collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    let next = AtomicUsize::new(0);
    let workers = threads.clamp(1, items.len());
    std::thread::scope(|sc| {
        for _ in 0..workers {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(i, _)) = order.get(k) else { break };
                if cancel.load(Ordering::Relaxed) {
                    done(i, Err("已取消".into()));
                    continue;
                }
                let r = convert(&items[i], params, cancel.clone(), |f, st| progress(i, f, st));
                done(i, r);
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outputs_are_unique_and_never_overwrite_inputs() {
        let ins = vec![PathBuf::from("/nonexistent/a/x.m4a"), PathBuf::from("/nonexistent/b/x.m4a"), PathBuf::from("/nonexistent/a/x_asmr.wav"), PathBuf::from("/nonexistent/a/y.wav")];
        let beside = plan_outputs(&ins, None);
        assert_eq!(beside[0], PathBuf::from("/nonexistent/a/x_asmr_2.wav"));
        assert_eq!(beside[1], PathBuf::from("/nonexistent/b/x_asmr.wav"));
        assert_eq!(beside[2], PathBuf::from("/nonexistent/a/x_asmr_asmr.wav"));
        let one = plan_outputs(&ins, Some(Path::new("/nonexistent/out")));
        assert_eq!(one[0], PathBuf::from("/nonexistent/out/x_asmr.wav"));
        assert_eq!(one[1], PathBuf::from("/nonexistent/out/x_asmr_2.wav"));
        assert_eq!(one[3], PathBuf::from("/nonexistent/out/y_asmr.wav"));
    }
}
