//! 流式重采样（加窗 sinc，多相表）：设备采样率 ↔ 48 kHz，以及输入输出设备时钟不同步时的微调。

use super::filters::SincTable;

pub struct Resampler {
    table: SincTable,
    hist: Vec<f32>,
    pos: usize,
    n: usize,
    /// 每输出一个样本，输入前进多少个样本。
    step: f64,
    /// 下一个输出样本的位置：比"最新输入样本"早多少个样本。攒够插值器半宽的"未来"样本后才能算。
    behind: f64,
    half: usize,
}

impl Resampler {
    /// `in_rate` -> `out_rate`。
    pub fn new(in_rate: f64, out_rate: f64) -> Self {
        let ratio = out_rate / in_rate;
        let cutoff = (ratio.min(1.0) * 0.96) as f32;
        let cutoff = if ratio >= 1.0 { 1.0 } else { cutoff };
        let taps = (((32.0 / cutoff.max(0.25)) as usize + 1) & !1).clamp(32, 128);
        let half = taps / 2 - 1;
        Resampler { table: SincTable::new(taps, 256, 8.0, cutoff), hist: vec![0.0; 2 * taps * 2], pos: 0, n: taps * 2, step: in_rate / out_rate, behind: -1.0, half }
    }

    /// 微调速度（1.0 = 标称；1.001 = 多消耗 0.1% 的输入），用来补偿两个设备的时钟漂移。
    pub fn set_trim(&mut self, in_rate: f64, out_rate: f64, trim: f64) {
        self.step = in_rate / out_rate * trim;
    }

    /// 送入一段输入，把产生的输出追加到 `out`。
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let taps = self.table.taps;
        for &x in input {
            self.pos = if self.pos == 0 { self.n - 1 } else { self.pos - 1 };
            self.hist[self.pos] = x;
            self.hist[self.pos + self.n] = x;
            self.behind += 1.0;
            // 输出位置落在"有足够未来样本"的范围内就产出
            while self.behind >= self.half as f64 {
                let d = self.behind;
                let di = d.floor();
                let start = self.pos + di as usize - self.half;
                if start + taps > self.pos + self.n {
                    break;
                }
                out.push(self.table.interp(&self.hist[start..start + taps], (d - di) as f32));
                self.behind -= self.step;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_44k1_to_48k_keeps_a_sine() {
        let mut r = Resampler::new(44_100.0, 48_000.0);
        let w = 2.0 * std::f64::consts::PI * 1000.0 / 44_100.0;
        let x: Vec<f32> = (0..44_100).map(|n| (w * n as f64).sin() as f32).collect();
        let mut y = Vec::new();
        r.process(&x, &mut y);
        assert!((y.len() as i64 - 48_000).abs() < 40, "len {}", y.len());
        // 输出仍是 1 kHz：比较中间一段的幅度和过零周期
        let seg = &y[10_000..30_000];
        let pk = seg.iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!((pk - 1.0).abs() < 0.01, "peak {pk}");
        let zc = seg.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        assert!((zc as i64 - 417).abs() <= 1, "zero crossings {zc}");
    }
}
