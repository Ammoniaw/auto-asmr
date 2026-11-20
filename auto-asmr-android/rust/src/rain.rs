//! 背景雨声单独输出：hook 层自己开一条 AAudio 输出流（48 kHz 立体声）播放它，网页暂停、换网页时雨声也不断。
//!
//! 做法和桌面版引擎里的完全一样（`engine::sync_rain` + `Rain::render`）：按「目标响度」定音量，再乘输出增益、
//! 过限制器。网页声音的处理器用的是去掉背景音的配置（见 [`crate::settings`]），所以不会有两份雨声。
//! 和桌面版的一个区别：桌面版雨声和人声共用一个限制器，这里各有各的；雨声比人声低十几 dB，两者相加
//! 超过峰值上限的余量很小。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, TryLockError};

use auto_asmr::dsp::dynamics::Limiter;
use auto_asmr::dsp::engine::{sync_rain, EngineConfig};
use auto_asmr::dsp::rain::Rain;
use auto_asmr::dsp::db_to_lin;

use crate::settings::{full_cfg, CFG_VER, GLOBAL};

const CHUNK: usize = 1024;

/// 要不要雨声（界面线程写，音频回调里读）：关掉时淡出，又打开时接着下。同一时刻只有一个渲染器。
static WANT: AtomicBool = AtomicBool::new(true);

/// 打开 / 关掉雨声（淡入淡出在下一次回调里处理，不和回调抢渲染器）。
pub fn set_wanted(on: bool) {
    WANT.store(on, Ordering::Release);
}

pub struct RainOut {
    rain: Option<Rain>,
    cfg: Arc<EngineConfig>,
    ver: u64,
    /// 关掉（原声 / 关雨声）后淡出；淡完 [`RainOut::render`] 返回 false。
    off: bool,
    limiter: Limiter,
    l: Vec<f32>,
    r: Vec<f32>,
}

impl RainOut {
    pub fn new() -> Self {
        let cfg = full_cfg();
        let mut rain = None;
        sync_rain(&mut rain, &cfg);
        WANT.store(true, Ordering::Release);
        RainOut { rain, ver: CFG_VER.load(Ordering::Acquire), cfg, off: false, limiter: Limiter::new(2.0, 80.0), l: vec![0.0; CHUNK], r: vec![0.0; CHUNK] }
    }

    /// 关掉：淡出。
    fn set_off(&mut self) {
        self.off = true;
        if let Some(r) = &mut self.rain {
            r.set_off();
        }
    }

    /// 又打开了（淡出途中）：按当前配置接着下。
    fn set_on(&mut self) {
        if self.off {
            self.off = false;
            sync_rain(&mut self.rain, &self.cfg);
        }
    }

    fn sync_config(&mut self) {
        let v = CFG_VER.load(Ordering::Acquire);
        if v == self.ver {
            return;
        }
        let g = match GLOBAL.try_lock() {
            Ok(g) => g,
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
            Err(TryLockError::WouldBlock) => return,
        };
        if let Some(g) = g.as_ref() {
            self.cfg = g.cfg.clone();
            self.ver = v;
            if !self.off {
                sync_rain(&mut self.rain, &self.cfg);
            }
        }
    }

    /// 生成一段交错立体声。返回 false = 已经没有声音了（关掉且淡完，或背景音音量为 0），可以停掉输出流。
    pub fn render(&mut self, data: &mut [f32]) -> bool {
        match (WANT.load(Ordering::Acquire), self.off) {
            (false, false) => self.set_off(),
            (true, true) => self.set_on(),
            _ => {}
        }
        self.sync_config();
        let p = &self.cfg.params;
        let (target, gain, ceiling) = (p.target_lufs, db_to_lin(p.out_gain_db), db_to_lin(p.ceiling_db - 0.1));
        let n = data.len() / 2;
        let mut done = 0;
        while done < n {
            let m = (n - done).min(CHUNK);
            let (l, r) = (&mut self.l[..m], &mut self.r[..m]);
            match &mut self.rain {
                Some(rain) => rain.render(target, l, r),
                None => {
                    l.fill(0.0);
                    r.fill(0.0);
                }
            }
            for i in 0..m {
                let (a, b) = self.limiter.process(l[i] * gain, r[i] * gain, ceiling);
                data[(done + i) * 2] = a;
                data[(done + i) * 2 + 1] = b;
            }
            done += m;
        }
        if self.rain.as_ref().map_or(false, |r| r.silent()) {
            self.rain = None;
        }
        self.rain.is_some()
    }
}

impl Default for RainOut {
    fn default() -> Self {
        Self::new()
    }
}
