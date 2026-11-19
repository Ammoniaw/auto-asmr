//! 所有可调参数。实时模式和文件模式共用；界面上每个滑块对应这里的一个字段。

use super::EAR_R;
use serde::{Deserialize, Serialize};

/// 左右移动的方式。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionMode {
    /// 慢速穿梭：贴左耳停一会儿 → 慢慢移到右耳 → 停 → 移回，节奏规律、带一点随机。
    Sweep,
    /// 随机走位：停留多久、去哪一侧、落在耳边 / 侧前 / 侧后都随机。
    Random,
    /// 绕头转圈。
    Orbit,
    /// 固定在一个位置。
    Static,
}

/// 换边时从哪里绕。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Via {
    Front,
    Back,
    Alternate,
    Random,
}

/// 回放端补偿。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Playback {
    /// 通用扩散场补偿（KU100 CDFC）。
    Diffuse,
    /// 不补偿：KU100 原机音色，和真实 KU100 录音一样。
    Native,
    /// 按耳机型号补偿（TH Köln 在同一个 KU100 上实测），值为型号名。
    Headphone(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Params {
    // ---- 贴耳感
    /// 贴耳时嘴到头中心的距离（米）。0.105 = 离耳约 1.7 cm。
    pub ear_dist: f32,
    /// 贴耳时远耳额外压低多少 dB（单耳独占感，0 = 纯物理）。
    pub isolation_db: f32,
    /// 贴耳时近耳的临场感 / 空气感，dB。
    pub presence_db: f32,
    /// 近耳音色用 KU100 实测标定的校正曲线（高频更收）。
    pub ku100_eq: bool,
    /// 贴耳时近耳 5 kHz 附近压低多少 dB（更"暗"的音色）。
    pub dark_db: f32,
    /// 凑近时两耳共同的低频温暖感，dB。
    pub warmth_db: f32,
    /// 贴耳时只给近耳加的低频，dB。
    pub ear_warmth_db: f32,
    /// 贴耳时远耳 320 Hz 以下再压低多少 dB。
    pub far_lf_cut_db: f32,

    // ---- 左右移动
    pub mode: MotionMode,
    /// 每只耳朵边停留多少秒。
    pub hold_s: f32,
    /// 从一只耳朵移到另一只耳朵用多少秒。
    pub move_s: f32,
    /// 时长和落点的随机程度（0 = 完全规律，1 = 很不规律）。
    pub randomness: f32,
    /// 换边时改成"先退远再凑近"的概率。
    pub approach: f32,
    pub via: Via,
    /// 贴耳的方位角（90 = 正侧方，100 = 略靠后）。
    pub ear_angle: f32,
    /// 穿过中间时离头中心的距离（米）。
    pub pass_dist: f32,
    /// 环绕模式：一圈的秒数、半径（米）。
    pub orbit_period: f32,
    pub orbit_radius: f32,
    /// 固定模式的方位角（0 前 90 右 180 后 270 左）。
    pub static_az: f32,
    /// 响度随位置变化的程度：1 = 物理真实，0 = 处处一样响。
    pub level_follow: f32,

    // ---- 忽远忽近
    /// 随机凑近 / 退开的幅度（0 = 关闭）。
    pub sway: f32,
    /// 说话时一次凑近—退开的大致秒数。
    pub sway_period: f32,
    /// 凑近 / 退开的随机程度（0 = 固定节奏和深浅，1 = 时长和深浅都随机）。
    pub sway_random: f32,
    /// 跟着说话音量：声音大时往后让、轻声时更贴近（0 = 关闭）。
    pub sway_follow: f32,
    /// 退开时耳边音量保留物理衰减的比例。
    pub sway_level: f32,
    /// 停顿时动作的速度（相对说话时）；1 = 一直在动。
    pub pause_motion: f32,
    /// 自然晃动幅度倍数（1 约 1.2 cm）。
    pub micro: f32,

    // ---- 房间与回放
    /// KU100 实测真实房间。
    pub room: bool,
    /// 房间声电平偏移，dB。
    pub wet_db: f32,
    pub playback: Playback,

    // ---- 人声处理
    /// 输入降噪（RNNoise）。
    pub denoise: bool,
    /// 降噪强度（0–1）。
    pub denoise_mix: f32,
    /// 响应阈值：输入低于这个电平（dBFS）时当作静音；-90 = 关闭。
    pub gate_db: f32,
    /// 去齿音最多压低多少 dB（0 = 关闭）。
    pub deess_db: f32,
    /// 轻声化（ANSI S3.5 说话力度均衡），0 = 关闭。
    pub soft_voice: f32,

    // ---- 输出
    /// 自动响度（实时模式：慢速把输出拉到目标响度；文件模式总是精确归一化）。
    pub auto_level: bool,
    /// 目标响度 LUFS。
    pub target_lufs: f32,
    /// 额外输出增益 dB。
    pub out_gain_db: f32,
    /// 真峰值上限 dBTP。
    pub ceiling_db: f32,
    /// 随机种子（换一个就换一套走位）。
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            ear_dist: EAR_R,
            isolation_db: 32.0,
            presence_db: 3.0,
            ku100_eq: false,
            dark_db: 0.0,
            warmth_db: 6.0,
            ear_warmth_db: 5.0,
            far_lf_cut_db: 12.0,
            mode: MotionMode::Sweep,
            hold_s: 2.5,
            move_s: 2.5,
            randomness: 0.1,
            approach: 0.25,
            via: Via::Front,
            ear_angle: 100.0,
            pass_dist: 0.2,
            orbit_period: 12.0,
            orbit_radius: 0.35,
            static_az: 270.0,
            level_follow: 0.2,
            sway: 1.0,
            sway_period: 4.0,
            sway_random: 1.0,
            sway_follow: 1.0,
            sway_level: 0.45,
            pause_motion: 0.13,
            micro: 1.0,
            room: false,
            wet_db: -3.0,
            playback: Playback::Diffuse,
            denoise: false,
            denoise_mix: 1.0,
            gate_db: -90.0,
            deess_db: 0.0,
            soft_voice: 0.0,
            auto_level: true,
            target_lufs: -16.0,
            out_gain_db: 0.0,
            ceiling_db: -1.0,
            seed: 1,
        }
    }
}

impl Params {
    /// 贴耳加强（默认）：嘴离耳约 1.7 cm，远耳额外压低 32 dB，不加房间。
    pub fn style_ear(mut self) -> Self {
        self.ear_dist = EAR_R;
        self.isolation_db = 32.0;
        self.far_lf_cut_db = 12.0;
        self.room = false;
        self.playback = Playback::Diffuse;
        self
    }

    /// KU100 实录风格：按两段 KU100 实录标定——远耳听得到、离耳约 6 cm、真实房间、KU100 原机音色。
    pub fn style_ku100(mut self) -> Self {
        self.ear_dist = 0.15;
        self.isolation_db = 0.0;
        self.far_lf_cut_db = 0.0;
        self.room = true;
        self.playback = Playback::Native;
        self
    }

    /// 各运动模式默认的"响度随位置变化"程度（与 Python 版相同）。
    pub fn default_level_follow(mode: MotionMode) -> f32 {
        match mode {
            MotionMode::Sweep => 0.2,
            MotionMode::Random => 0.5,
            MotionMode::Orbit => 0.3,
            MotionMode::Static => 1.0,
        }
    }
}
