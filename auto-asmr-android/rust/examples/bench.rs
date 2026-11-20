//! 引擎在这台机器上有多快：60 秒合成人声，按 Chromium 的回调长度（4360 帧）送进处理器，
//! 分别按 48 kHz 和 44.1 kHz（多两次重采样）计时。
//!
//!   cargo run --release --example bench                  # 在电脑上
//!   （安卓上：编译 --target aarch64-linux-android，adb push 到 /data/local/tmp 运行，见 doc/android-dev.md）

use asmr_android::{asmr_proc_free, asmr_proc_new, asmr_proc_process, asmr_set_style};
use auto_asmr::dsp::testsig::test_voice;

fn main() {
    asmr_android::settings::provide_repo_data();
    let secs = 60.0;
    let voice = test_voice(secs);
    for (name, style) in [("贴耳加强", 0), ("KU100 实录", 1)] {
        asmr_set_style(style);
        for rate in [48_000, 44_100] {
            let mut x: Vec<f32> = voice.iter().flat_map(|&v| [v, v]).collect();
            let p = asmr_proc_new(rate, 2);
            let t = std::time::Instant::now();
            for chunk in x.chunks_mut(4360 * 2) {
                unsafe { asmr_proc_process(p, chunk.as_mut_ptr(), (chunk.len() / 2) as i32) };
            }
            let el = t.elapsed().as_secs_f64();
            unsafe { asmr_proc_free(p) };
            let audio = voice.len() as f64 / rate as f64;
            println!("{name} {rate} Hz：{:.1} µs / 128 样本，占单核 {:.2}%", el / audio * 128.0 / rate as f64 * 1e6, el / audio * 100.0);
        }
    }
}
