//! 合成的测试信号（不用录音素材）：库里的单元测试、桌面版命令行的性能测试都用它。

use super::dynamics::LoudnessMeter;
use super::FS;

/// 合成一段像说话的测试信号：带谐波的浊音段和停顿交替。
pub fn test_voice(seconds: f32) -> Vec<f32> {
    let n = (seconds * FS) as usize;
    let mut x = vec![0f32; n];
    let mut seed = 99u32;
    for (i, v) in x.iter_mut().enumerate() {
        let t = i as f32 / FS;
        let phrase = (t * 0.45).fract() < 0.7; // 约 1.5 s 说话、0.7 s 停顿
        let syl = (0.5 - 0.5 * (std::f32::consts::TAU * 4.0 * t).cos()).powf(0.6);
        let f0 = 190.0 + 30.0 * (std::f32::consts::TAU * 0.3 * t).sin();
        let mut s = 0.0;
        for h in 1..12 {
            s += (std::f32::consts::TAU * f0 * h as f32 * t).sin() / h as f32;
        }
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
        *v = if phrase { 0.25 * syl * (s * 0.5 + 0.15 * noise) } else { 0.0005 * noise };
    }
    x
}

/// 合成一段像棉签 / 刮擦的测试信号：约每 0.6 s 一下的摩擦噪声（300 Hz–5 kHz，起音和收尾平滑），夹着稀疏的咔嗒。
pub fn test_scratch(seconds: f32) -> Vec<f32> {
    let n = (seconds * FS) as usize;
    let mut x = vec![0f32; n];
    let mut seed = 4321u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    };
    let (mut lp, mut hp, mut click) = (0f32, 0f32, 0f32);
    for (i, v) in x.iter_mut().enumerate() {
        let ph = (i as f32 / FS / 0.62).fract();
        let env = if ph < 0.7 { (std::f32::consts::PI * ph / 0.7).sin().powi(2) } else { 0.0 };
        lp += 0.45 * (rnd() - lp);
        hp += 0.04 * (lp - hp);
        if rnd() > 0.9996 {
            click = 1.0;
        }
        click *= 0.9;
        *v = 0.2 * env * ((lp - hp) + 0.8 * click * rnd());
    }
    // 隔着空气录的摩擦声没有低频：300 Hz 以下滤掉（4 阶）
    let mut f = [super::filters::Biquad::highpass(300.0, super::filters::BUTTER_Q); 2];
    for v in x.iter_mut() {
        let u = f[0].process(*v);
        *v = f[1].process(u);
    }
    x
}

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

/// 立体声的 BS.1770 积分响度（LUFS）。
pub fn lufs(l: &[f32], r: &[f32]) -> f32 {
    let mut m = LoudnessMeter::new();
    for (&a, &b) in l.iter().zip(r) {
        m.push(a, b);
    }
    m.integrated()
}
