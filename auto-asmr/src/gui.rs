//! 图形界面（egui）：实时页 + 文件转换页，右侧是两种模式共用的效果参数。

use crate::dsp::engine::{EngineConfig, Telemetry};
use crate::dsp::params::{MotionMode, Params, Playback, Via, DENOISE_MAX_DB};
use crate::dsp::tables::tables;
use crate::dsp::motion::lift_tilt;
use crate::dsp::{FS, HEAD_A, R_MIN};
use crate::offline::{convert, default_output, load_mono, Channel, ConvertOptions, ConvertReport};
use crate::realtime::{self, DeviceInfo, DeviceList, Realtime, RtOptions};
use eframe::egui;
use egui::{Color32, RichText};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};

static ENGLISH: AtomicBool = AtomicBool::new(false);

/// 参数表第一列（文字）的最小宽度，各分组的滑块因此对齐。
const COL: f32 = 104.0;

const CONFIG_FILE: &str = "auto-asmr.ron";

/// 位置图的半径刻度：到头中心的距离（米）→ 离图中心的比例（0–1）。对数压缩：2 cm 在中心，80 cm 在边上。
fn radar_map(r: f32) -> f32 {
    ((r / 0.02).max(1.0).ln() / (RADAR_MAX_R / 0.02).ln()).clamp(0.0, 1.0)
}

/// 位置图上的一点（相对图中心的偏移；`rmax` = 满刻度对应的像素数）→（方位角 度, 到头中心的距离 米）。
/// 距离限制在头外面（不会拖进头里）到图的边缘之间。
fn radar_pick(d: egui::Vec2, rmax: f32) -> (f32, f32) {
    let az = d.x.atan2(-d.y).to_degrees().rem_euclid(360.0);
    let r = 0.02 * ((RADAR_MAX_R / 0.02).ln() * (d.length() / rmax).clamp(0.0, 1.0)).exp();
    (az, r.clamp(R_MIN, RADAR_MAX_R))
}

/// 位置图边缘对应的距离（米）。
const RADAR_MAX_R: f32 = 0.8;

/// 高度角（度）：以"耳朵"为支点看声源在上方还是下方多少度——与运动里倾斜用的支点相同
/// （声源在正侧方时支点是那只耳朵，在正前 / 正后方时是头中心）。同时返回水平面上的"未抬高"位置
/// （方位角, 到头中心的距离），用来把一个三维位置换回手动模式的三个量。
fn up_angle(az: f32, el: f32, r: f32) -> (f32, f32, f32) {
    let (a, e) = (az.to_radians(), el.to_radians());
    let (x, y, z) = (r * e.cos() * a.sin(), r * e.cos() * a.cos(), r * e.sin());
    // 支点的位置取决于"未抬高"时的方位，先用当前方位估计，迭代几次就收敛
    let mut cx = HEAD_A * x / r.max(1e-6);
    let (mut up, mut px, mut py) = (0.0, x, y);
    for _ in 0..6 {
        let vh = (x - cx).hypot(y);
        let d = vh.hypot(z);
        up = z.atan2(vh).to_degrees();
        px = cx + (x - cx) / vh.max(1e-6) * d;
        py = y / vh.max(1e-6) * d;
        cx = HEAD_A * px / px.hypot(py).max(1e-6);
    }
    (up, px.atan2(py).to_degrees().rem_euclid(360.0), px.hypot(py))
}

/// 参数表的一行：文字（悬停显示说明）+ 滑块。
fn row(ui: &mut egui::Ui, label: &str, tip: &str, w: egui::Slider<'_>) {
    let l = ui.label(label);
    if !tip.is_empty() {
        l.on_hover_text(tip);
    }
    ui.add(w);
    ui.end_row();
}

/// 界面文字：中文 / 英文二选一。
fn t(zh: &'static str, en: &'static str) -> &'static str {
    if ENGLISH.load(Ordering::Relaxed) {
        en
    } else {
        zh
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Tab {
    Realtime,
    File,
}

/// 关闭程序时保存、下次打开时恢复的内容。
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
struct Saved {
    params: Params,
    english: bool,
    tab: Tab,
    host: String,
    input: Option<String>,
    output: Option<String>,
    buffer: u32,
    in_channel: usize,
    dry: bool,
    bits: u16,
    file_lufs: f32,
    file_channel: u8,
}

impl Default for Saved {
    fn default() -> Self {
        Saved { params: Params::default(), english: false, tab: Tab::Realtime, host: String::new(), input: None, output: None, buffer: 256, in_channel: 0, dry: false, bits: 24, file_lufs: -16.0, file_channel: 0 }
    }
}

struct Job {
    progress: Arc<Mutex<(f32, String)>>,
    cancel: Arc<AtomicBool>,
    result: Arc<Mutex<Option<Result<ConvertReport, String>>>>,
    output: PathBuf,
}

type Decoded = Result<(PathBuf, u8, Arc<Vec<f32>>), String>;

pub struct App {
    s: Saved,
    cfg: Arc<EngineConfig>,
    sent: Params,
    hosts: Vec<(cpal::HostId, String)>,
    devices: DeviceList,
    rt: Option<Realtime>,
    rt_preview: bool,
    status: String,
    file_in: Option<PathBuf>,
    file_out: Option<PathBuf>,
    start_s: f32,
    dur_s: f32,
    preview_s: f64,
    preview_dragging: bool,
    preview_pending: bool,
    job: Option<Job>,
    report: String,
    decoded: Option<(PathBuf, u8, Arc<Vec<f32>>)>,
    decoding: Option<Receiver<Decoded>>,
    tele: Telemetry,
    trail: Vec<(f32, f32)>,
    /// 最近一次用的自动模式（关掉"手动摆位"时回到它）、上一帧的模式。
    auto_mode: MotionMode,
    last_mode: MotionMode,
    shot: Option<(PathBuf, u32)>,
    /// 截图自检时模拟的一次拖动：按下点和松开点（窗口坐标）。
    shot_drag: Option<[f32; 4]>,
    shot_drag_frame: u32,
    demo: Option<PathBuf>,
}

/// `screenshot`：(PNG 路径, 是否切到文件页, 演示文件)——自检用：打开界面、可选地试听演示文件几秒、截图后退出。
pub fn run(screenshot: Option<(PathBuf, bool, Option<PathBuf>)>) -> Result<(), String> {
    let config_path = std::env::current_dir().map_err(|e| format!("无法获取当前运行目录：{e}"))?.join(CONFIG_FILE);
    let opts = eframe::NativeOptions {
        persistence_path: Some(config_path),
        viewport: egui::ViewportBuilder::default().with_inner_size([1260.0, 840.0]).with_min_inner_size([1040.0, 640.0]).with_title("auto-asmr · KU100 双耳 ASMR"),
        ..Default::default()
    };
    eframe::run_native("auto-asmr", opts, Box::new(move |cc| Ok(Box::new(App::new(cc, screenshot))))).map_err(|e| e.to_string())
}

impl App {
    fn export_config(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter(t("配置文件", "Configuration"), &["ron"])
            .set_file_name("auto-asmr-export.ron")
            .save_file() else { return };
        // 和自动保存使用相同的 eframe 键值格式，可直接替换运行目录中的配置文件。
        let result = (|| -> Result<(), String> {
            let saved = ron::ser::to_string_pretty(&self.s, Default::default()).map_err(|e| e.to_string())?;
            let values = std::collections::BTreeMap::from([(eframe::APP_KEY, saved)]);
            let text = ron::ser::to_string_pretty(&values, Default::default()).map_err(|e| e.to_string())?;
            std::fs::write(&path, text).map_err(|e| e.to_string())
        })();
        self.status = match result {
            Ok(()) => format!("{}：{}", t("配置已导出", "Configuration exported"), path.display()),
            Err(e) => format!("{}：{e}", t("配置导出失败", "Configuration export failed")),
        };
    }

    fn new(cc: &eframe::CreationContext<'_>, screenshot: Option<(PathBuf, bool, Option<PathBuf>)>) -> Self {
        let mut s: Saved = cc.storage.and_then(|st| eframe::get_value(st, eframe::APP_KEY)).unwrap_or_default();
        let mut demo = None;
        let shot = screenshot.map(|(p, file_tab, demo_file)| {
            // 截图自检：用默认设置，保证每次画面一致；要看别的状态时，用环境变量 AUTO_ASMR_SHOT_PARAMS
            // 给出参数（RON，只写要改的项，如 "(denoise:true,denoise_db:12.0)"）
            s = Saved::default();
            if let Some(p) = std::env::var("AUTO_ASMR_SHOT_PARAMS").ok().and_then(|v| ron::from_str::<Params>(&v).ok()) {
                s.params = p;
            }
            if file_tab {
                s.tab = Tab::File;
            }
            if demo_file.is_some() {
                // 演示时不出声：输出压到 −60 dB
                s.params.out_gain_db = -60.0;
                s.params.auto_level = false;
            }
            demo = demo_file;
            (p, 0)
        });
        let hosts = realtime::hosts();
        let host_id = hosts.iter().find(|h| h.1 == s.host).map(|h| h.0);
        let devices = realtime::list_devices(host_id);
        let cfg = Arc::new(EngineConfig::new(s.params.clone(), None));
        let mut style = (*cc.egui_ctx.global_style()).clone();
        style.spacing.slider_width = 150.0;
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        cc.egui_ctx.set_global_style(style);
        let s_mode = s.params.mode;
        App {
            sent: s.params.clone(),
            s,
            cfg,
            hosts,
            devices,
            rt: None,
            rt_preview: false,
            status: String::new(),
            file_in: None,
            file_out: None,
            start_s: 0.0,
            dur_s: 0.0,
            preview_s: 0.0,
            preview_dragging: false,
            preview_pending: false,
            job: None,
            report: String::new(),
            decoded: None,
            decoding: None,
            tele: Telemetry { az: 270.0, r: 0.105, in_db: -120.0, out_db: -120.0, ..Default::default() },
            trail: Vec::new(),
            auto_mode: if s_mode == MotionMode::Manual { MotionMode::Sweep } else { s_mode },
            last_mode: s_mode,
            shot,
            shot_drag: std::env::var("AUTO_ASMR_SHOT_DRAG").ok().and_then(|v| {
                let f: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                (f.len() == 4).then(|| [f[0], f[1], f[2], f[3]])
            }),
            shot_drag_frame: 0,
            demo,
        }
    }

    fn host_id(&self) -> Option<cpal::HostId> {
        self.hosts.iter().find(|h| h.1 == self.s.host).map(|h| h.0)
    }

    fn rt_options(&self) -> RtOptions {
        RtOptions { host: self.host_id(), input: self.s.input.clone(), output: self.s.output.clone(), buffer_frames: self.s.buffer, input_channel: self.s.in_channel }
    }

    fn reload_devices(&mut self) {
        self.hosts = realtime::hosts();
        self.devices = realtime::list_devices(self.host_id());
    }

    fn stop(&mut self) {
        if self.rt_preview && !self.preview_dragging {
            if let Some(rt) = &self.rt {
                self.preview_s = rt.shared.play_pos.load(Ordering::Relaxed) as f64 / FS as f64;
            }
        }
        self.rt = None;
        self.rt_preview = false;
        self.preview_pending = false;
    }

    fn start_mic(&mut self) {
        self.stop();
        self.sync_params();
        match Realtime::start(&self.rt_options(), self.cfg.clone()) {
            Ok(rt) => {
                rt.shared.monitor_dry.store(self.s.dry, Ordering::Relaxed);
                self.status = format!("{}：{} → {}（{} Hz → {} Hz）", t("运行中", "Running"), rt.info.input_name, rt.info.output_name, rt.info.in_rate, rt.info.out_rate);
                self.rt = Some(rt);
            }
            Err(e) => self.status = e,
        }
    }

    fn file_channel(&self) -> Channel {
        match self.s.file_channel {
            1 => Channel::Left,
            2 => Channel::Right,
            3 => Channel::Mix,
            _ => Channel::Auto,
        }
    }

    fn convert_options(&self) -> Option<ConvertOptions> {
        let input = self.file_in.clone()?;
        let mut o = ConvertOptions::new(input.clone());
        o.output = self.file_out.clone().unwrap_or_else(|| default_output(&input));
        o.start = self.start_s.max(0.0) as f64;
        o.duration = if self.dur_s > 0.0 { Some(self.dur_s as f64) } else { None };
        o.channel = self.file_channel();
        o.bits = self.s.bits;
        o.lufs = self.s.file_lufs;
        o.peak_db = self.s.params.ceiling_db;
        Some(o)
    }

    fn start_convert(&mut self) {
        let Some(opts) = self.convert_options() else { return };
        let job = Job { progress: Arc::new(Mutex::new((0.0, String::new()))), cancel: Arc::new(AtomicBool::new(false)), result: Arc::new(Mutex::new(None)), output: opts.output.clone() };
        let (pr, ca, re, params) = (job.progress.clone(), job.cancel.clone(), job.result.clone(), self.s.params.clone());
        std::thread::spawn(move || {
            let r = convert(&opts, &params, ca, |f, stage| {
                if let Ok(mut g) = pr.lock() {
                    *g = (f, stage.to_string());
                }
            });
            if let Ok(mut g) = re.lock() {
                *g = Some(r);
            }
        });
        self.report.clear();
        self.job = Some(job);
    }

    /// 试听：先在后台解码（已解码过同一个文件就直接用），再用实时引擎播放。
    fn start_preview(&mut self) {
        let Some(input) = self.file_in.clone() else { return };
        let selected_s = self.preview_s;
        self.stop();
        self.preview_s = selected_s;
        self.sync_params();
        let ch = self.s.file_channel;
        if let Some((p, c, audio)) = &self.decoded {
            if *p == input && *c == ch {
                let start = ((self.preview_s.max(0.0) * FS as f64) as usize).min(audio.len());
                match Realtime::start_preview(&self.rt_options(), self.cfg.clone(), audio.clone(), start) {
                    Ok(rt) => {
                        self.status = format!("{}：{}（{} Hz）", t("试听中", "Previewing"), rt.info.output_name, rt.info.out_rate);
                        self.rt = Some(rt);
                        self.rt_preview = true;
                    }
                    Err(e) => self.status = e,
                }
                return;
            }
        }
        self.preview_pending = true;
        if self.decoding.is_none() {
            self.decode_file();
        }
    }

    fn decode_file(&mut self) {
        let Some(input) = self.file_in.clone() else { return };
        let ch = self.s.file_channel;
        let (tx, rx) = channel();
        let mut o = ConvertOptions::new(input.clone());
        o.channel = self.file_channel();
        std::thread::spawn(move || {
            let r = load_mono(&o, &AtomicBool::new(false)).map(|(m, _)| (input, ch, Arc::new(m)));
            let _ = tx.send(r);
        });
        self.decoding = Some(rx);
        self.status = t("正在解码…", "Decoding…").to_string();
    }

    fn set_file(&mut self, p: PathBuf) {
        self.stop();
        self.file_out = Some(default_output(&p));
        self.file_in = Some(p);
        self.decoded = None;
        self.preview_s = self.start_s.max(0.0) as f64;
        self.preview_dragging = false;
        self.report.clear();
        self.decode_file();
    }

    // ------------------------------------------------------------------ 各块界面

    fn device_combo(ui: &mut egui::Ui, id: &str, list: &[DeviceInfo], sel: &mut Option<String>) {
        let def_name = list.iter().find(|d| d.is_default).map(|d| d.name.as_str()).unwrap_or("—");
        let text = match sel {
            None => format!("{}（{}）", t("系统默认", "System default"), def_name),
            Some(id) => list.iter().find(|d| &d.id == id).map(|d| d.name.clone()).unwrap_or_else(|| t("（设备未找到）", "(device not found)").to_string()),
        };
        egui::ComboBox::from_id_salt(id).width(300.0).selected_text(text).show_ui(ui, |ui| {
            ui.selectable_value(sel, None, format!("{}（{}）", t("系统默认", "System default"), def_name));
            for d in list {
                ui.selectable_value(sel, Some(d.id.clone()), &d.name);
            }
        });
    }

    fn ui_devices(&mut self, ui: &mut egui::Ui, with_input: bool) {
        ui.group(|ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(t("音频设备", "Audio devices")).strong());
            egui::Grid::new("dev").num_columns(2).spacing([10.0, 6.0]).show(ui, |ui| {
                ui.label(t("设备类型", "Host"));
                let mut host = self.s.host.clone();
                let shown = if host.is_empty() { self.hosts.first().map(|h| h.1.clone()).unwrap_or_default() } else { host.clone() };
                egui::ComboBox::from_id_salt("host").width(300.0).selected_text(shown).show_ui(ui, |ui| {
                    for (_, name) in &self.hosts {
                        ui.selectable_value(&mut host, name.clone(), name);
                    }
                });
                if host != self.s.host {
                    self.s.host = host;
                    self.devices = realtime::list_devices(self.host_id());
                }
                ui.end_row();
                if with_input {
                    ui.label(t("输入设备", "Input"));
                    Self::device_combo(ui, "in", &self.devices.inputs, &mut self.s.input);
                    ui.end_row();
                }
                ui.label(t("输出设备", "Output"));
                Self::device_combo(ui, "out", &self.devices.outputs, &mut self.s.output);
                ui.end_row();
            });
            ui.horizontal(|ui| {
                if ui.button(t("重载设备列表", "Reload devices")).clicked() {
                    self.reload_devices();
                }
                ui.label(t("缓冲", "Buffer"));
                egui::ComboBox::from_id_salt("buf").width(70.0).selected_text(if self.s.buffer == 0 { t("默认", "Default").to_string() } else { self.s.buffer.to_string() }).show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.s.buffer, 0, t("默认", "Default"));
                    for n in [64u32, 128, 256, 512, 1024] {
                        ui.selectable_value(&mut self.s.buffer, n, n.to_string());
                    }
                });
                ui.label(t("帧", "frames")).on_hover_text(t("越小延迟越低；出现爆音就调大", "Smaller = lower latency; increase if you hear crackles"));
                if with_input {
                    ui.label(t("输入声道", "Input ch"));
                    let txt = if self.s.in_channel == usize::MAX { t("混合", "Mix").to_string() } else { (self.s.in_channel + 1).to_string() };
                    egui::ComboBox::from_id_salt("inch").width(60.0).selected_text(txt).show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.s.in_channel, 0, "1");
                        ui.selectable_value(&mut self.s.in_channel, 1, "2");
                        ui.selectable_value(&mut self.s.in_channel, usize::MAX, t("混合", "Mix"));
                    });
                }
            });
            if with_input && self.devices.inputs.is_empty() {
                ui.colored_label(Color32::from_rgb(220, 150, 60), t("没有找到输入设备（麦克风）", "No input device (microphone) found"));
            }
        });
    }

    /// 头顶俯视图：声源相对头的位置（上 = 正前方），半径按对数压缩，贴耳的细节看得清。
    /// 右侧的竖条是高度：从耳朵（声源在正前 / 正后方时从头中心）看过去，嘴在上方还是下方多少度。
    fn ui_radar(&mut self, ui: &mut egui::Ui, size: f32) {
        const GAUGE_W: f32 = 20.0;
        let manual = self.s.params.mode == MotionMode::Manual;
        let sense = if manual { egui::Sense::click_and_drag() } else { egui::Sense::hover() };
        let (resp, p) = ui.allocate_painter(egui::vec2(size + GAUGE_W, size), sense);
        let rect = egui::Rect::from_min_size(resp.rect.min, egui::vec2(size, size));
        let c = rect.center();
        let rmax = size * 0.47;
        let map = |r: f32| -> f32 { radar_map(r) * rmax };
        // 右侧的高度标尺
        let (gx, g0, g1) = (rect.right() + 0.5 * GAUGE_W - 2.0, rect.top() + 30.0, rect.bottom() - 18.0);
        let gm = 0.5 * (g0 + g1);
        if self.shot.as_ref().map_or(false, |s| s.1 == 2) {
            eprintln!("位置图：中心 ({:.0}, {:.0})，满刻度 {:.0} px；高度标尺 x={:.0}，y={:.0}…{:.0}", c.x, c.y, rmax, gx, g0, g1);
        }
        if manual {
            // 手动摆位：在俯视图里点 / 拖改方位和距离，在右侧竖条上点 / 拖改高度（双击竖条回到与耳朵齐平）
            let origin = ui.input(|i| i.pointer.press_origin());
            let on_gauge = |pt: egui::Pos2| pt.x > rect.right();
            if resp.is_pointer_button_down_on() {
                if let (Some(pt), Some(o)) = (resp.interact_pointer_pos(), origin) {
                    let pr = &mut self.s.params;
                    if on_gauge(o) {
                        pr.manual_tilt = ((gm - pt.y) / (0.5 * (g1 - g0)) * 90.0).clamp(-80.0, 85.0);
                    } else {
                        (pr.manual_az, pr.manual_dist) = radar_pick(pt - c, rmax);
                    }
                }
            }
            if resp.double_clicked() && resp.interact_pointer_pos().map_or(false, on_gauge) {
                self.s.params.manual_tilt = 0.0;
            }
            if resp.hovered() {
                ui.ctx().set_cursor_icon(if resp.is_pointer_button_down_on() { egui::CursorIcon::Grabbing } else { egui::CursorIcon::Grab });
            }
            if self.rt.is_none() {
                // 没在运行：圆点直接显示摆好的位置
                let pr = &self.s.params;
                let (az, el, r) = lift_tilt(pr.manual_az, pr.manual_dist, 0.0, pr.manual_tilt);
                (self.tele.az, self.tele.el, self.tele.r) = (az, el, r);
            }
        }
        let dark = ui.visuals().dark_mode;
        let grid = if dark { Color32::from_gray(70) } else { Color32::from_gray(190) };
        let fg = if dark { Color32::from_gray(200) } else { Color32::from_gray(60) };
        p.rect_filled(resp.rect, 8.0, if dark { Color32::from_gray(24) } else { Color32::from_gray(245) });
        for (rr, label) in [(0.1875, "10 cm"), (0.3875, "30 cm"), (0.6875, "60 cm")] {
            p.circle_stroke(c, map(rr), egui::Stroke::new(1.0, grid));
            let a = 0.25 * std::f32::consts::PI; // 距离刻度写在右下方，不挡左上角的读数
            p.text(c + egui::vec2(a.cos(), a.sin()) * (map(rr) + 2.0), egui::Align2::LEFT_TOP, label, egui::FontId::proportional(10.0), grid);
        }
        // 头、鼻子、耳朵
        let hr = map(HEAD_A);
        p.circle_filled(c, hr, if dark { Color32::from_gray(55) } else { Color32::from_gray(215) });
        p.circle_stroke(c, hr, egui::Stroke::new(1.5, fg));
        p.line_segment([c + egui::vec2(-hr * 0.22, -hr * 0.97), c + egui::vec2(0.0, -hr * 1.28)], egui::Stroke::new(1.5, fg));
        p.line_segment([c + egui::vec2(hr * 0.22, -hr * 0.97), c + egui::vec2(0.0, -hr * 1.28)], egui::Stroke::new(1.5, fg));
        for sx in [-1.0f32, 1.0] {
            p.rect_filled(egui::Rect::from_center_size(c + egui::vec2(sx * hr * 1.06, 0.0), egui::vec2(hr * 0.2, hr * 0.5)), 2.0, fg);
        }
        p.text(c + egui::vec2(-hr * 1.5, hr * 0.9), egui::Align2::RIGHT_TOP, t("左", "L"), egui::FontId::proportional(11.0), grid);
        p.text(c + egui::vec2(hr * 1.5, hr * 0.9), egui::Align2::LEFT_TOP, t("右", "R"), egui::FontId::proportional(11.0), grid);
        // 声源（嘴）和它的轨迹
        let pos = |az: f32, r: f32| {
            let a = az.to_radians();
            c + egui::vec2(a.sin(), -a.cos()) * map(r)
        };
        let active = self.rt.is_some();
        // 俯视图里画的是水平投影：声源升高时向头中心靠拢，从头顶绕过去时会穿过头的上方
        let (el, r3) = (self.tele.el.to_radians(), self.tele.r);
        let r_h = r3 * el.cos();
        if active {
            self.trail.push((self.tele.az, r_h));
            if self.trail.len() > 90 {
                self.trail.remove(0);
            }
        }
        let n = self.trail.len();
        for (i, w) in self.trail.windows(2).enumerate() {
            let a = (i as f32 / n.max(1) as f32 * 150.0) as u8;
            p.line_segment([pos(w[0].0, w[0].1), pos(w[1].0, w[1].1)], egui::Stroke::new(2.0, Color32::from_rgba_unmultiplied(255, 140, 60, a)));
        }
        let speaking = self.tele.act > 0.5;
        let col = if !active {
            Color32::from_gray(130)
        } else if speaking {
            Color32::from_rgb(255, 120, 40)
        } else {
            Color32::from_rgb(170, 120, 90)
        };
        let sp = pos(self.tele.az, r_h);
        p.circle_filled(sp, if speaking && active { 7.0 } else { 5.5 }, col);
        let accent = Color32::from_rgb(70, 150, 240);
        if manual {
            // 手动的目标位置：一个可以拖的圈（声音稍微滞后一点跟过来）
            let pr = &self.s.params;
            p.circle_stroke(pos(pr.manual_az, pr.manual_dist), 10.0, egui::Stroke::new(1.5, accent));
            p.circle_stroke(egui::pos2(gx, gm - (pr.manual_tilt / 90.0).clamp(-1.0, 1.0) * 0.5 * (g1 - g0)), 7.0, egui::Stroke::new(1.5, accent));
            p.text(rect.left_bottom() + egui::vec2(8.0, -5.0), egui::Align2::LEFT_BOTTOM, t("拖动圆点 · 右侧拖高度", "drag the dot · height at right"), egui::FontId::proportional(10.5), accent);
        }
        // 高度角：以"耳朵"为支点（与运动里倾斜的支点相同：正侧方时是那只耳朵，正前 / 正后方时是头中心）
        let (up, _, plan_r) = up_angle(self.tele.az, self.tele.el, r3);
        p.line_segment([egui::pos2(gx, g0), egui::pos2(gx, g1)], egui::Stroke::new(2.0, grid));
        p.line_segment([egui::pos2(gx - 4.0, gm), egui::pos2(gx + 4.0, gm)], egui::Stroke::new(1.5, grid));
        p.text(egui::pos2(gx, g0 - 3.0), egui::Align2::CENTER_BOTTOM, t("上", "up"), egui::FontId::proportional(10.0), grid);
        p.text(egui::pos2(gx, g1 + 2.0), egui::Align2::CENTER_TOP, t("下", "dn"), egui::FontId::proportional(10.0), grid);
        p.circle_filled(egui::pos2(gx, gm - (up / 90.0).clamp(-1.0, 1.0) * 0.5 * (g1 - g0)), if speaking && active { 5.0 } else { 4.0 }, col);
        let hgt = if up.abs() < 1.0 { String::new() } else { format!("   {} {:.0}°", if up > 0.0 { t("上", "up") } else { t("下", "down") }, up.abs()) };
        p.text(rect.left_top() + egui::vec2(8.0, 6.0), egui::Align2::LEFT_TOP, format!("{} {:.1} cm   {:.0}°{}", t("离耳", "from ear"), ((plan_r - HEAD_A) * 100.0).max(0.0), self.tele.az, hgt), egui::FontId::proportional(12.0), fg);
    }

    /// 电平条：`frac` 0–1 是填充比例，`hot` 时变红。
    fn meter(ui: &mut egui::Ui, label: &str, frac: f32, hot: bool, value: String) {
        ui.horizontal(|ui| {
            ui.add_sized([44.0, 16.0], egui::Label::new(label));
            let (rect, _) = ui.allocate_exact_size(egui::vec2(118.0, 10.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 3.0, ui.visuals().extreme_bg_color);
            let col = if hot { Color32::from_rgb(230, 90, 60) } else { Color32::from_rgb(90, 190, 120) };
            ui.painter().rect_filled(egui::Rect::from_min_size(rect.min, egui::vec2(rect.width() * frac.clamp(0.0, 1.0), rect.height())), 3.0, col);
            ui.label(value);
        });
    }

    fn ui_status(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                self.ui_radar(ui, 192.0);
                ui.vertical(|ui| {
                    let tl = self.tele;
                    let dbs = |v: f32| if v < -100.0 { "—".to_string() } else { format!("{v:.0} dB") };
                    Self::meter(ui, t("输入", "In"), (tl.in_db + 60.0) / 60.0, tl.in_db > -3.0, dbs(tl.in_db));
                    Self::meter(ui, t("输出", "Out"), (tl.out_db + 60.0) / 60.0, tl.out_db > -1.5, dbs(tl.out_db));
                    Self::meter(ui, t("限幅", "Limit"), tl.gr_db / 12.0, tl.gr_db > 6.0, format!("{:.1} dB", -tl.gr_db.max(0.0) + 0.0));
                    ui.label(format!("{} {:+.1} dB    {} {}", t("自动响度", "Auto level"), tl.level_gain_db, t("说话", "Speech"), if tl.act > 0.5 { "●" } else { "○" }));
                    ui.add_space(4.0);
                    if let Some(rt) = &self.rt {
                        let sh = &rt.shared;
                        ui.label(format!("{} {:.1} ms", t("算法延迟", "Algorithm latency"), rt.algo_ms()));
                        if !self.rt_preview {
                            ui.label(format!("{} {:.1} ms", t("总延迟约", "Total latency ≈"), rt.latency_ms())).on_hover_text(t("输入缓冲 + 环形缓冲 + 算法 + 输出缓冲，不含声卡驱动自身的延迟", "Input buffer + ring + algorithm + output buffer; excludes the driver's own latency"));
                        }
                        ui.label(format!("{} {} µs（{} {}）", t("处理时间", "Processing"), sh.proc_us.load(Ordering::Relaxed), t("最大", "max"), sh.proc_max_us.load(Ordering::Relaxed)));
                        ui.label(format!("{} {}", t("断流次数", "Dropouts"), sh.xruns.load(Ordering::Relaxed)));
                    } else {
                        ui.label(format!("{} {:.1} ms", t("算法延迟", "Algorithm latency"), if self.s.params.denoise { 22.4 } else { 2.4 }));
                        ui.label(RichText::new(t("未运行", "Stopped")).weak());
                    }
                });
            });
            let mut manual = self.s.params.mode == MotionMode::Manual;
            let tip = t(
                "打开后，穿梭、忽远忽近、自动上下、自然晃动全部停掉，声源停在你摆的位置：在位置图里点或拖改方位和距离，在右侧竖条上拖改高度（双击竖条回到与耳朵齐平）。运行中打开时，声源就停在此刻所在的位置。文件转换时相当于固定在这个位置",
                "When on, all automatic motion stops (sweeps, near / far sway, auto up / down, micro motion) and the source stays where you put it: click or drag on the map for direction and distance, drag the bar at the right for height (double-click it for ear level). Turned on while running, the source stays where it is at that moment. For file conversion this is a fixed position",
            );
            if ui.checkbox(&mut manual, t("手动摆位：关闭自动移动，拖动位置图里的声源", "Manual: stop automatic motion and drag the source on the map")).on_hover_text(tip).changed() {
                let p = &mut self.s.params;
                p.mode = if manual { MotionMode::Manual } else { self.auto_mode };
                p.level_follow = Params::default_level_follow(p.mode);
            }
        });
    }

    fn ui_realtime(&mut self, ui: &mut egui::Ui) {
        self.ui_devices(ui, true);
        self.ui_status(ui);
        ui.horizontal(|ui| {
            let running = self.rt.is_some() && !self.rt_preview;
            if ui.add_enabled(!running, egui::Button::new(RichText::new(t("开始", "Start")).strong()).min_size(egui::vec2(96.0, 30.0))).clicked() {
                self.start_mic();
            }
            if ui.add_enabled(self.rt.is_some(), egui::Button::new(t("停止", "Stop")).min_size(egui::vec2(96.0, 30.0))).clicked() {
                self.stop();
                self.status = t("已停止", "Stopped").to_string();
            }
            let before = self.s.dry;
            ui.radio_value(&mut self.s.dry, false, t("输出效果", "Effect"));
            ui.radio_value(&mut self.s.dry, true, t("监听原声", "Monitor dry"));
            if before != self.s.dry {
                if let Some(rt) = &self.rt {
                    rt.shared.monitor_dry.store(self.s.dry, Ordering::Relaxed);
                }
            }
        });
        ui.label(
            RichText::new(t(
                "直播时把输出设备选成虚拟声卡（如 VB-Cable / BlackHole），再在直播软件里把它当作麦克风。自己监听请戴耳机，外放听不出效果。",
                "For streaming, pick a virtual cable (VB-Cable / BlackHole) as output and use it as the microphone in your streaming app. Use headphones to monitor.",
            ))
            .small()
            .weak(),
        );
    }

    fn ui_file(&mut self, ui: &mut egui::Ui) {
        let channel_before = self.s.file_channel;
        ui.group(|ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(t("音频文件", "Audio file")).strong());
            ui.horizontal(|ui| {
                if ui.button(t("选择文件…", "Choose file…")).clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter(t("音频", "Audio"), &["wav", "m4a", "mp3", "flac", "ogg", "aac", "mp4", "mkv", "aiff", "caf"]).pick_file() {
                        self.set_file(p);
                    }
                }
                match &self.file_in {
                    Some(p) => ui.label(p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()).on_hover_text(p.display().to_string()),
                    None => ui.label(RichText::new(t("也可以把文件拖进窗口", "…or drop a file onto the window")).weak()),
                };
            });
            if let Some(out) = self.file_out.clone() {
                ui.horizontal(|ui| {
                    if ui.button(t("输出到…", "Save as…")).clicked() {
                        if let Some(p) = rfd::FileDialog::new().add_filter("WAV", &["wav"]).set_file_name(out.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()).save_file() {
                            self.file_out = Some(p);
                        }
                    }
                    ui.label(out.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()).on_hover_text(out.display().to_string());
                });
            }
            egui::Grid::new("fileopts").num_columns(4).spacing([10.0, 6.0]).show(ui, |ui| {
                ui.label(t("取哪个声道", "Channel"));
                let names = [t("自动（更响的）", "Auto (louder)"), t("左", "Left"), t("右", "Right"), t("混合", "Mix")];
                egui::ComboBox::from_id_salt("fch").width(120.0).selected_text(names[self.s.file_channel.min(3) as usize]).show_ui(ui, |ui| {
                    for (i, n) in names.iter().enumerate() {
                        ui.selectable_value(&mut self.s.file_channel, i as u8, *n);
                    }
                });
                ui.label(t("位深", "Bit depth"));
                egui::ComboBox::from_id_salt("bits").width(70.0).selected_text(format!("{} bit", self.s.bits)).show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.s.bits, 24, "24 bit");
                    ui.selectable_value(&mut self.s.bits, 16, "16 bit");
                });
                ui.end_row();
                ui.label(t("从第几秒", "Start (s)"));
                if ui.add(egui::DragValue::new(&mut self.start_s).range(0.0..=36000.0).speed(1.0)).changed() {
                    self.preview_s = self.start_s as f64;
                }
                ui.label(t("处理多少秒", "Duration (s)"));
                ui.add(egui::DragValue::new(&mut self.dur_s).range(0.0..=36000.0).speed(1.0)).on_hover_text(t("0 = 到结尾。先试 60 秒听听效果很方便", "0 = to the end. Try 60 s first to audition"));
                ui.end_row();
                ui.label(t("目标响度", "Loudness"));
                ui.add(egui::Slider::new(&mut self.s.file_lufs, -24.0..=-10.0).suffix(" LUFS").fixed_decimals(0));
                ui.end_row();
            });
        });
        if channel_before != self.s.file_channel && self.file_in.is_some() {
            let resume = self.rt_preview || self.preview_pending;
            self.stop();
            self.decoded = None;
            self.preview_pending = resume;
            self.decode_file();
        }
        self.ui_devices(ui, false);
        self.ui_status(ui);
        let has = self.file_in.is_some();
        ui.horizontal(|ui| {
            let previewing = self.rt.is_some() && self.rt_preview;
            if ui.add_enabled(has && !previewing && self.decoding.is_none(), egui::Button::new(t("试听", "Preview")).min_size(egui::vec2(96.0, 30.0))).on_hover_text(t("用实时引擎直接播放，可以边听边调右边的参数", "Plays through the real-time engine; tweak the parameters while listening")).clicked() {
                self.start_preview();
            }
            if ui.add_enabled(previewing, egui::Button::new(t("停止", "Stop")).min_size(egui::vec2(72.0, 30.0))).clicked() {
                self.stop();
                self.status = t("已停止", "Stopped").to_string();
            }
            let busy = self.job.is_some();
            if ui.add_enabled(has && !busy, egui::Button::new(RichText::new(t("转换并保存", "Convert & save")).strong()).min_size(egui::vec2(120.0, 30.0))).clicked() {
                self.start_convert();
            }
            if busy && ui.button(t("取消", "Cancel")).clicked() {
                if let Some(j) = &self.job {
                    j.cancel.store(true, Ordering::Relaxed);
                }
            }
        });
        if let Some((_, _, audio)) = &self.decoded {
            let duration = audio.len() as f64 / FS as f64;
            self.preview_s = self.preview_s.clamp(0.0, duration);
            ui.label(format!("{}  {} / {}", t("试听位置", "Preview position"), playback_time(self.preview_s), playback_time(duration)));
            let response = ui.scope(|ui| {
                ui.spacing_mut().slider_width = ui.available_width().max(1.0);
                ui.add_enabled(duration > 0.0, egui::Slider::new(&mut self.preview_s, 0.0..=duration).show_value(false))
            }).inner.on_hover_text(t("点击或拖动，立即试听该位置的处理效果；右侧参数边调边生效", "Click or drag to hear the processed audio here; effect parameters update while playing"));
            self.preview_dragging = response.dragged();
            if response.changed() || response.clicked() || response.drag_started() {
                if let Some(rt) = self.rt.as_ref().filter(|_| self.rt_preview) {
                    rt.shared.seek_preview((self.preview_s * FS as f64) as u64);
                } else {
                    self.start_preview();
                }
            }
        }
        if let Some(j) = &self.job {
            let (f, stage) = j.progress.lock().map(|g| g.clone()).unwrap_or((0.0, String::new()));
            ui.add(egui::ProgressBar::new(f).text(format!("{stage} {:.0}%", f * 100.0)));
        }
        if !self.report.is_empty() {
            ui.label(&self.report);
        }
    }

    /// 左下角：输入降噪。一个 0–100 的滑条：数值 = 底噪最多压低多少分贝，0 = 关闭，100 = 不设上限。
    fn ui_denoise(ui: &mut egui::Ui, p: &mut Params) {
        ui.add_space(5.0);
        let mut v = if p.denoise { p.denoise_db.clamp(0.0, DENOISE_MAX_DB) } else { 0.0 };
        ui.horizontal(|ui| {
            ui.label(RichText::new(t("降噪", "Denoise")).strong()).on_hover_text(t(
                "输入降噪（RNNoise，纯 CPU）：在空间化之前去掉麦克风 / 录音里的底噪。数值 = 底噪最多压低多少分贝，0 = 关闭，100 = 不设上限；60 以上基本没有区别。人声的音量不变",
                "Input denoise (RNNoise, CPU only): removes background noise before spatialization. The value is how many dB the noise is lowered by at most; 0 = off, 100 = no limit, little changes above 60. The voice keeps its level",
            ));
            ui.spacing_mut().slider_width = 300.0;
            let slider = egui::Slider::new(&mut v, 0.0..=DENOISE_MAX_DB)
                .step_by(1.0)
                .fixed_decimals(0)
                .custom_formatter(|x, _| {
                    if x < 0.5 {
                        t("0 关", "0 off").to_string()
                    } else if x >= DENOISE_MAX_DB as f64 - 0.5 {
                        t("100 最大", "100 max").to_string()
                    } else {
                        format!("{x:.0} dB")
                    }
                })
                .custom_parser(|s| s.trim().split(|c: char| !(c.is_ascii_digit() || c == '.')).find(|w| !w.is_empty()).and_then(|w| w.parse::<f64>().ok()));
            if ui.add(slider).changed() {
                p.denoise = v >= 0.5;
                if p.denoise {
                    p.denoise_db = v.round();
                }
            }
        });
        let note = if !p.denoise {
            t("0 = 关闭。麦克风或录音有底噪时往右拖：数值 = 底噪最多压低多少分贝", "0 = off. Drag right for a noisy microphone or recording: the value is how many dB the noise is lowered by at most").to_string()
        } else if p.denoise_db >= DENOISE_MAX_DB {
            t("不设上限，降到最干净；气声的高频会被削掉一些，很轻的尾音可能被压掉。延迟增加 20 ms", "No limit, as clean as it gets; breathy highs lose a little and very soft tails may be cut. Adds 20 ms of latency").to_string()
        } else {
            format!("{} {:.0} dB{}", t("底噪最多压低", "Noise lowered by at most"), p.denoise_db, t("，人声音量不变。数值越大越干净，气声的高频也会被削掉一点。延迟增加 20 ms", "; the voice keeps its level. Higher = cleaner, but breathy highs lose a little. Adds 20 ms of latency"))
        };
        ui.add(egui::Label::new(RichText::new(note).small().weak()).wrap());
        ui.add_space(3.0);
    }

    /// 右侧第一栏：风格、贴耳感、左右移动。
    fn params_left(ui: &mut egui::Ui, p: &mut Params) {
        ui.horizontal(|ui| {
            ui.label(RichText::new(t("风格", "Style")).strong());
            if ui.button(t("贴耳加强", "Close to ear")).on_hover_text(t("嘴离耳约 1.7 cm，另一只耳朵几乎听不到，不加房间", "Mouth ≈1.7 cm from the ear, far ear almost silent, no room")).clicked() {
                *p = p.clone().style_ear();
            }
            if ui.button(t("KU100 实录", "KU100 natural")).on_hover_text(t("按真实 KU100 录音标定：远耳听得到、离耳约 6 cm、真实房间、KU100 原机音色", "Calibrated to real KU100 recordings: far ear audible, ≈6 cm, real room, native KU100 tone")).clicked() {
                *p = p.clone().style_ku100();
            }
            if ui.button(t("恢复默认", "Reset")).clicked() {
                *p = Params::default();
            }
        });
        egui::CollapsingHeader::new(RichText::new(t("贴耳感", "Closeness")).strong()).default_open(true).show(ui, |ui| {
            egui::Grid::new("g1").num_columns(2).min_col_width(COL).spacing([10.0, 6.0]).show(ui, |ui| {
                let mut cm = (p.ear_dist - HEAD_A) * 100.0;
                row(ui, t("贴耳距离", "Distance from ear"), t("嘴离耳朵多远。1.7 cm = 贴着耳朵", "How far the mouth is from the ear. 1.7 cm = touching"), egui::Slider::new(&mut cm, 0.8..=25.0).logarithmic(true).suffix(" cm").fixed_decimals(1));
                p.ear_dist = HEAD_A + cm / 100.0;
                row(ui, t("远耳压低", "Far-ear cut"), t("贴耳时另一只耳朵额外压低多少（0 = 纯物理，像真实 KU100）", "Extra attenuation of the far ear when close (0 = pure physics)"), egui::Slider::new(&mut p.isolation_db, 0.0..=40.0).suffix(" dB").fixed_decimals(0));
                row(ui, t("临场感", "Presence"), t("贴耳时近耳 3.5 kHz 和空气感", "Near-ear 3.5 kHz presence and air"), egui::Slider::new(&mut p.presence_db, 0.0..=8.0).suffix(" dB").fixed_decimals(1));
                row(ui, t("暗度", "Darkness"), t("贴耳时近耳 5 kHz 附近压低，音色更暗、更柔", "Dip around 5 kHz at the near ear: darker, softer"), egui::Slider::new(&mut p.dark_db, 0.0..=8.0).suffix(" dB").fixed_decimals(1));
                row(ui, t("温暖感", "Warmth"), t("凑近时两耳共同的低频", "Low-frequency warmth in both ears when close"), egui::Slider::new(&mut p.warmth_db, 0.0..=12.0).suffix(" dB").fixed_decimals(1));
                row(ui, t("近耳低频", "Near-ear bass"), t("贴耳时只给近耳加的低频", "Bass added only to the near ear"), egui::Slider::new(&mut p.ear_warmth_db, 0.0..=10.0).suffix(" dB").fixed_decimals(1));
                row(ui, t("远耳低频削减", "Far-ear bass cut"), t("贴耳时远耳 320 Hz 以下再压低", "Extra cut below 320 Hz at the far ear"), egui::Slider::new(&mut p.far_lf_cut_db, 0.0..=18.0).suffix(" dB").fixed_decimals(0));
                row(ui, t("气声增强", "Breath lift"), t("贴耳时把嗓音里轻的送气声 / 气息提上来（最多这么多），响的齿音不提；只用嗓音自己的气声，不加噪声。0 = 关闭", "Lifts the soft breathy part of the voice when close (by at most this much); loud sibilants are left alone. No noise is added. 0 = off"), egui::Slider::new(&mut p.breath_db, 0.0..=12.0).suffix(" dB").fixed_decimals(1));
                row(ui, t("气流扑耳", "Breath puff"), t("送气时气流扑到耳朵上的低频脉冲（耳朵痒的感觉），由嗓音自己的送气声导出。0 = 关闭，1 = 标准", "Low-frequency pulses of breath hitting the ear, derived from the voice's own aspiration. 0 = off, 1 = standard"), egui::Slider::new(&mut p.puff, 0.0..=3.0).fixed_decimals(2));
                row(ui, t("耳甲共鸣", "Concha resonance"), t("贴耳时近耳 4.3 kHz 的耳甲腔共振（KU100 贴耳实录的共有特征）。0 = 关闭", "4.3 kHz concha resonance at the near ear when close. 0 = off"), egui::Slider::new(&mut p.concha_db, 0.0..=8.0).suffix(" dB").fixed_decimals(1));
            });
            ui.checkbox(&mut p.ku100_eq, t("近耳用 KU100 实测校正曲线（高频更收）", "KU100-calibrated near-ear EQ (softer highs)"));
        });
        egui::CollapsingHeader::new(RichText::new(t("左右移动", "Left / right motion")).strong()).default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                let before = p.mode;
                for (m, zh, en) in [(MotionMode::Sweep, "穿梭", "Sweep"), (MotionMode::Random, "随机走位", "Random"), (MotionMode::Orbit, "环绕", "Orbit"), (MotionMode::Static, "固定", "Static"), (MotionMode::Manual, "手动", "Manual")] {
                    ui.selectable_value(&mut p.mode, m, t(zh, en));
                }
                if p.mode != before {
                    p.level_follow = Params::default_level_follow(p.mode);
                }
            });
            egui::Grid::new("g2").num_columns(2).min_col_width(COL).spacing([10.0, 6.0]).show(ui, |ui| {
                match p.mode {
                    MotionMode::Sweep | MotionMode::Random => {
                        let mut freq = 60.0 / (p.hold_s + p.move_s).max(0.2);
                        let f0 = freq;
                        row(ui, t("移动频率", "Moves per minute"), t("说话时每分钟换几次耳朵（同时缩放停留和移动时间）", "Ear switches per minute of speech (scales hold and move time)"), egui::Slider::new(&mut freq, 2.0..=40.0).suffix(t(" 次/分", " /min")).fixed_decimals(0));
                        if (freq - f0).abs() > 1e-3 {
                            let k = f0 / freq;
                            p.hold_s = (p.hold_s * k).clamp(0.2, 30.0);
                            p.move_s = (p.move_s * k).clamp(0.2, 30.0);
                        }
                        row(ui, t("停留时间", "Hold time"), t("每只耳朵边停留多久", "How long it stays at each ear"), egui::Slider::new(&mut p.hold_s, 0.3..=12.0).suffix(" s").fixed_decimals(1));
                        row(ui, t("移动时间", "Move time"), t("从一只耳朵移到另一只耳朵用多久", "How long a crossing takes"), egui::Slider::new(&mut p.move_s, 0.3..=12.0).suffix(" s").fixed_decimals(1));
                        row(ui, t("随机程度", "Randomness"), t("时长和落点的随机程度：0 = 完全规律", "Randomness of timing and landing spots: 0 = strictly regular"), egui::Slider::new(&mut p.randomness, 0.0..=1.0).fixed_decimals(2));
                        if p.mode == MotionMode::Sweep {
                            row(ui, t("从远处凑近", "Approach"), t("换边时先退到约 0.5 m 再慢慢凑近的概率", "Chance that a switch backs off to ≈0.5 m and then looms in"), egui::Slider::new(&mut p.approach, 0.0..=1.0).fixed_decimals(2));
                            row(ui, t("贴耳角度", "Ear angle"), t("90 = 正侧方，100 = 略靠后", "90 = straight to the side, 100 = slightly behind"), egui::Slider::new(&mut p.ear_angle, 60.0..=140.0).suffix("°").fixed_decimals(0));
                            row(ui, t("穿过时的距离", "Pass distance"), t("经过脸前 / 脑后时离头中心多远", "Distance from the head centre while crossing"), egui::Slider::new(&mut p.pass_dist, 0.12..=0.6).suffix(" m").fixed_decimals(2));
                        }
                    }
                    MotionMode::Orbit => {
                        row(ui, t("一圈用时", "Period"), "", egui::Slider::new(&mut p.orbit_period, 3.0..=60.0).suffix(" s").fixed_decimals(0));
                        row(ui, t("半径", "Radius"), "", egui::Slider::new(&mut p.orbit_radius, 0.1..=1.5).logarithmic(true).suffix(" m").fixed_decimals(2));
                    }
                    MotionMode::Static => {
                        row(ui, t("方位角", "Azimuth"), t("0 前，90 右，180 后，270 左", "0 front, 90 right, 180 back, 270 left"), egui::Slider::new(&mut p.static_az, 0.0..=359.0).suffix("°").fixed_decimals(0));
                    }
                    MotionMode::Manual => {
                        row(ui, t("方位角", "Azimuth"), t("0 前，90 右，180 后，270 左。也可以直接在左边的位置图上拖", "0 front, 90 right, 180 back, 270 left. Or drag on the map at the left"), egui::Slider::new(&mut p.manual_az, 0.0..=359.9).suffix("°").fixed_decimals(0));
                        let mut cm = (p.manual_dist - HEAD_A) * 100.0;
                        row(ui, t("离头距离", "Distance from head"), t("声源离头表面多远；在耳朵正侧方时就是离耳朵多远", "Distance from the head surface; at the side this is the distance from the ear"), egui::Slider::new(&mut cm, 0.8..=71.0).logarithmic(true).suffix(" cm").fixed_decimals(1));
                        p.manual_dist = (HEAD_A + cm / 100.0).clamp(R_MIN, RADAR_MAX_R);
                        row(ui, t("高度", "Height"), t("以耳朵为支点向上（正）/ 向下（负）偏多少度。也可以拖位置图右侧的竖条", "Above (+) / below (−) the ear, in degrees. Or drag the bar at the right of the map"), egui::Slider::new(&mut p.manual_tilt, -80.0..=85.0).suffix("°").fixed_decimals(0));
                    }
                }
                row(ui, t("响度随位置变化", "Level follows position"), t("1 = 物理真实（贴耳很响、到脸前会弱很多），0 = 处处一样响", "1 = physical (loud at the ear, much quieter in front), 0 = constant"), egui::Slider::new(&mut p.level_follow, 0.0..=1.0).fixed_decimals(2));
            });
            if p.mode == MotionMode::Manual {
                ui.add(egui::Label::new(RichText::new(t("手动：声源停在你摆的位置，在左边的位置图上拖动。忽远忽近、上下、自然晃动此时不起作用", "Manual: the source stays where you put it; drag it on the map at the left. Near / far sway, up / down and micro motion are inactive")).weak().small()).wrap());
            }
            if matches!(p.mode, MotionMode::Sweep) {
                ui.horizontal(|ui| {
                    ui.label(t("换边时从哪里绕", "Cross via"));
                    for (v, zh, en) in [(Via::Front, "脸前", "Front"), (Via::Back, "脑后", "Back"), (Via::Alternate, "交替", "Alternate"), (Via::Random, "随机", "Random")] {
                        ui.selectable_value(&mut p.via, v, t(zh, en));
                    }
                });
            }
        });
    }

    /// 右侧第二栏：忽远忽近、KU100 临场感、人声处理、输出。
    fn params_right(ui: &mut egui::Ui, p: &mut Params) {
        egui::CollapsingHeader::new(RichText::new(t("忽远忽近", "Near / far sway")).strong()).default_open(true).show(ui, |ui| {
            if p.mode == MotionMode::Manual {
                ui.disable(); // 手动摆位时不起作用
            }
            egui::Grid::new("g3").num_columns(2).min_col_width(COL).spacing([10.0, 6.0]).show(ui, |ui| {
                row(ui, t("幅度", "Amount"), t("随机地凑近、往后让的幅度（0 = 关闭）", "How far it leans in and out (0 = off)"), egui::Slider::new(&mut p.sway, 0.0..=2.0).fixed_decimals(2));
                row(ui, t("节奏", "Period"), t("说话时一次凑近—退开大约几秒", "Seconds of speech per lean-out / lean-in cycle"), egui::Slider::new(&mut p.sway_period, 2.0..=14.0).suffix(" s").fixed_decimals(1));
                row(ui, t("随机程度", "Randomness"), t("0 = 固定节奏和深浅，1 = 时长和深浅都随机", "0 = fixed rhythm and depth, 1 = random timing and depth"), egui::Slider::new(&mut p.sway_random, 0.0..=1.0).fixed_decimals(2));
                row(ui, t("跟随音量", "Follow loudness"), t("声音大时往后让、轻声时更贴近，听着更舒服", "Louder → leans back, softer → comes closer; more comfortable"), egui::Slider::new(&mut p.sway_follow, 0.0..=2.0).fixed_decimals(2));
                row(ui, t("退开时的音量", "Level when back"), t("退开时耳边音量保留物理衰减的比例（真人退开会说大声一点；KU100 实录 0.4–0.56）", "Share of the physical level drop kept when leaning back (talkers raise their voice; KU100 recordings 0.4–0.56)"), egui::Slider::new(&mut p.sway_level, 0.0..=1.0).fixed_decimals(2));
                row(ui, t("停顿时的动作", "Motion in pauses"), t("停顿时动作的速度：0.13 = 几乎不动，下一句从原位置接着说；1 = 一直在动。实时（主播）模式下，停顿超过 1 秒会恢复正常速度，话少时声音也一直在动", "Motion speed during pauses: 0.13 = almost still, 1 = always moving. In real-time mode motion resumes at full speed after 1 s of pause, so it keeps moving when you talk little"), egui::Slider::new(&mut p.pause_motion, 0.0..=1.0).fixed_decimals(2));
                row(ui, t("自然晃动", "Micro motion"), t("人体自然的微小晃动（1 约 1.2 cm）", "Natural small body sway (1 ≈ 1.2 cm)"), egui::Slider::new(&mut p.micro, 0.0..=2.0).fixed_decimals(2));
            });
        });
        egui::CollapsingHeader::new(RichText::new(t("上下", "Up / down")).strong()).default_open(true).show(ui, |ui| {
            if p.mode == MotionMode::Manual {
                ui.disable(); // 手动摆位时不起作用（高度在"左右移动"里调，或拖位置图右侧的竖条）
            }
            egui::Grid::new("g7").num_columns(2).min_col_width(COL).spacing([10.0, 6.0]).show(ui, |ui| {
                row(ui, t("高度", "Height"), t("嘴相对耳朵向上（正）/ 向下（负）偏多少度，离耳朵的距离不变；0 = 与耳朵齐平", "Mouth above (+) / below (−) the ear, in degrees, at the same distance from the ear; 0 = level with the ear"), egui::Slider::new(&mut p.height, -60.0..=75.0).suffix("°").fixed_decimals(0));
                row(ui, t("自动上下", "Auto up / down"), t("嘴在耳朵上方 / 下方之间慢慢换位置的幅度（0 = 关闭）", "How far the mouth drifts above / below the ear on its own (0 = off)"), egui::Slider::new(&mut p.updown, 0.0..=60.0).suffix("°").fixed_decimals(0));
                row(ui, t("节奏", "Period"), t("说话时上下一个来回大约几秒", "Seconds of speech per up / down cycle"), egui::Slider::new(&mut p.updown_period, 3.0..=30.0).suffix(" s").fixed_decimals(1));
                row(ui, t("随机程度", "Randomness"), t("0 = 节奏和高低固定，1 = 时长和高低都随机", "0 = fixed rhythm and height, 1 = random timing and height"), egui::Slider::new(&mut p.updown_random, 0.0..=1.0).fixed_decimals(2));
                row(ui, t("从头顶绕", "Over the head"), t("换耳时改成从头顶绕过去的概率（穿梭 / 随机走位）：0 = 只从脸前 / 脑后绕，1 = 每次都从头顶绕", "Chance that an ear switch goes over the top of the head (sweep / random): 0 = front / back only, 1 = always over the top"), egui::Slider::new(&mut p.over_top, 0.0..=1.0).fixed_decimals(2));
            });
            ui.add(egui::Label::new(RichText::new(t("上下比左右、远近弱，因人而异；气声、轻语更明显", "Height is a weaker cue than left / right or distance and varies between listeners; clearer on breathy voices")).weak().small()).wrap());
        });
        egui::CollapsingHeader::new(RichText::new(t("KU100 临场感", "KU100 presence")).strong()).default_open(true).show(ui, |ui| {
            ui.checkbox(&mut p.room, t("真实房间（KU100 实测，远耳听到的是房间声而不是寂静）", "Real room (measured with a KU100: the far ear hears the room, not silence)"));
            egui::Grid::new("g4").num_columns(2).min_col_width(COL).spacing([10.0, 6.0]).show(ui, |ui| {
                ui.add_enabled_ui(p.room, |ui| ui.label(t("房间声大小", "Room level")));
                ui.add_enabled(p.room, egui::Slider::new(&mut p.wet_db, -12.0..=9.0).suffix(" dB").fixed_decimals(0));
                ui.end_row();
                ui.label(t("回放补偿", "Playback EQ")).on_hover_text(t("扩散场：通用；KU100 原机：和真实 KU100 录音一样（高频略暗）；或按你的耳机型号补偿", "Diffuse: generic; Native: like a real KU100 recording (slightly darker); or pick your headphone model"));
                let cur = match &p.playback {
                    Playback::Diffuse => t("通用扩散场", "Diffuse field").to_string(),
                    Playback::Native => t("KU100 原机音色", "KU100 native").to_string(),
                    Playback::Headphone(n) => n.clone(),
                };
                egui::ComboBox::from_id_salt("pb").width(190.0).selected_text(cur).show_ui(ui, |ui| {
                    ui.selectable_value(&mut p.playback, Playback::Diffuse, t("通用扩散场", "Diffuse field"));
                    ui.selectable_value(&mut p.playback, Playback::Native, t("KU100 原机音色", "KU100 native"));
                    for (n, _) in &tables().headphones {
                        ui.selectable_value(&mut p.playback, Playback::Headphone(n.clone()), n);
                    }
                });
                ui.end_row();
            });
        });
        egui::CollapsingHeader::new(RichText::new(t("人声处理", "Voice")).strong()).default_open(true).show(ui, |ui| {
            egui::Grid::new("g5").num_columns(2).min_col_width(COL).spacing([10.0, 6.0]).show(ui, |ui| {
                row(ui, t("响应阈值", "Gate threshold"), t("输入低于这个电平时当作静音（−90 = 关闭）", "Input below this level is muted (−90 = off)"), egui::Slider::new(&mut p.gate_db, -90.0..=-20.0).suffix(" dB").fixed_decimals(0));
                row(ui, t("去齿音", "De-ess"), t("只在 s / sh / x 出现时压低 4.5 kHz 以上，最多压这么多", "Ducks >4.5 kHz only on sibilants, by at most this much"), egui::Slider::new(&mut p.deess_db, 0.0..=10.0).suffix(" dB").fixed_decimals(1));
                row(ui, t("轻声化", "Soft voice"), t("把嗓音均衡成「压低声音说话」（只做均衡，不改嗓音）", "EQ towards a lowered-voice spectrum (EQ only)"), egui::Slider::new(&mut p.soft_voice, 0.0..=1.5).fixed_decimals(2));
            });
        });
        egui::CollapsingHeader::new(RichText::new(t("输出", "Output")).strong()).default_open(true).show(ui, |ui| {
            ui.checkbox(&mut p.auto_level, t("自动响度（实时 / 试听：慢慢把音量拉到目标）", "Auto level (real-time / preview: slowly rides to the target)"));
            egui::Grid::new("g6").num_columns(2).min_col_width(COL).spacing([10.0, 6.0]).show(ui, |ui| {
                ui.add_enabled_ui(p.auto_level, |ui| ui.label(t("目标响度", "Target")));
                ui.add_enabled(p.auto_level, egui::Slider::new(&mut p.target_lufs, -30.0..=-10.0).suffix(" LUFS").fixed_decimals(0));
                ui.end_row();
                row(ui, t("输出增益", "Output gain"), "", egui::Slider::new(&mut p.out_gain_db, -60.0..=12.0).suffix(" dB").fixed_decimals(1));
                row(ui, t("峰值上限", "Ceiling"), t("限制器的真峰值上限", "True-peak ceiling of the limiter"), egui::Slider::new(&mut p.ceiling_db, -6.0..=-0.1).suffix(" dBTP").fixed_decimals(1));
                ui.label(t("随机种子", "Seed")).on_hover_text(t("换一个数字就换一套走位（下次开始时生效）", "A different number gives a different path (applies on next start)"));
                ui.add(egui::DragValue::new(&mut p.seed).speed(0.1));
                ui.end_row();
            });
        });
    }

    /// 在控件修改参数的当帧提交，音频线程下一次回调直接使用最新快照。
    fn sync_params(&mut self) {
        let mode = self.s.params.mode;
        if mode != self.last_mode {
            if mode == MotionMode::Manual {
                self.auto_mode = self.last_mode;
                if self.rt.is_some()
                    && self.s.params.manual_az == self.sent.manual_az
                    && self.s.params.manual_dist == self.sent.manual_dist
                    && self.s.params.manual_tilt == self.sent.manual_tilt
                {
                    // 运行中切到手动：声源就停在此刻所在的位置，从这里开始拖
                    let (up, az, dist) = up_angle(self.tele.az, self.tele.el, self.tele.r);
                    let p = &mut self.s.params;
                    p.manual_az = az;
                    p.manual_dist = dist.clamp(R_MIN, RADAR_MAX_R);
                    p.manual_tilt = up.clamp(-80.0, 85.0);
                }
            }
            self.last_mode = mode;
        }
        if self.s.params != self.sent {
            let cfg = Arc::new(EngineConfig::new(self.s.params.clone(), Some(&self.cfg)));
            if let Some(rt) = &self.rt {
                rt.set_config(cfg.clone());
            }
            self.cfg = cfg;
            self.sent = self.s.params.clone();
        }
    }

    /// 每帧的后台事务：试听解码完成、转换完成、设备出错。
    fn poll(&mut self, ctx: &egui::Context) {
        ENGLISH.store(self.s.english, Ordering::Relaxed);
        if let Some(rx) = &self.decoding {
            if let Ok(r) = rx.try_recv() {
                self.decoding = None;
                match r {
                    Ok(d) => {
                        self.decoded = Some(d);
                        if self.preview_pending {
                            self.start_preview();
                        } else {
                            self.status = t("音频已就绪，点击或拖动播放条即可试听", "Audio ready; click or drag the timeline to preview").to_string();
                        }
                    }
                    Err(e) => {
                        self.preview_pending = false;
                        self.status = e;
                    }
                }
            }
        }
        let mut finished = None;
        if let Some(j) = &self.job {
            if let Ok(mut g) = j.result.lock() {
                if let Some(r) = g.take() {
                    finished = Some((r, j.output.clone()));
                }
            }
        }
        if let Some((r, out)) = finished {
            self.job = None;
            self.report = match r {
                Ok(rep) => format!(
                    "{}：{}\n{:.1} s，{:.1} LUFS，{} {:.1} dBTP，{} {:.1} dB；{} {:.0}%，{} {}（{} {}），{} {}，{} {}；{} {:.1} s{}",
                    t("已保存", "Saved"),
                    out.display(),
                    rep.seconds,
                    rep.lufs,
                    t("真峰值", "true peak"),
                    rep.true_peak_db,
                    t("限制器最多压", "limiter max"),
                    rep.max_gr_db,
                    t("说话占", "speech"),
                    rep.speech_pct,
                    t("换耳", "switches"),
                    rep.n_moves,
                    t("从头顶绕", "over the head"),
                    rep.n_tops,
                    t("往后让", "lean-backs"),
                    rep.n_sways,
                    t("上下换位", "up / down moves"),
                    rep.n_updowns,
                    t("用时", "took"),
                    rep.elapsed_s,
                    if rep.channel_note.is_empty() { String::new() } else { format!("\n{}", rep.channel_note) }
                ),
                Err(e) => format!("{}：{e}", t("失败", "Failed")),
            };
        }
        let mut stop = false;
        if let Some(rt) = &self.rt {
            self.tele = rt.shared.telemetry();
            if let Some(e) = rt.shared.take_error() {
                self.status = e;
                stop = true;
            }
            if self.rt_preview && !self.preview_dragging {
                self.preview_s = rt.shared.play_pos.load(Ordering::Relaxed) as f64 / FS as f64;
            }
            if self.rt_preview && rt.shared.preview_finished() {
                self.status = t("试听结束", "Preview finished").to_string();
                stop = true;
            }
        } else {
            self.tele.act = 0.0;
            self.tele.in_db = -120.0;
            self.tele.out_db = -120.0;
            self.tele.gr_db = 0.0;
        }
        if stop {
            self.stop();
        }
        // 拖进窗口的文件
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).filter(|p| !p.as_os_str().is_empty()).collect());
        if let Some(p) = dropped.into_iter().next() {
            self.set_file(p);
            self.s.tab = Tab::File;
        }
        if self.rt.is_some() || self.job.is_some() || self.decoding.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        }
    }
}

impl eframe::App for App {
    /// 截图自检时模拟一次鼠标拖动（环境变量 AUTO_ASMR_SHOT_DRAG="x0,y0,x1,y1"，窗口坐标）：
    /// 第 5 帧在 (x0, y0) 按下，之后几帧移到 (x1, y1)，第 13 帧松开；截图在第 20 帧。
    /// 文件试听自检从解码完成后开始计数，确保播放条已经出现。
    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw: &mut egui::RawInput) {
        if let (Some((_, frame)), Some([x0, y0, x1, y1])) = (&self.shot, self.shot_drag) {
            let frame = if self.file_in.is_some() {
                if self.decoded.is_none() {
                    return;
                }
                self.shot_drag_frame += 1;
                self.shot_drag_frame
            } else {
                *frame
            };
            let (a, b) = (egui::pos2(x0, y0), egui::pos2(x1, y1));
            let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
            match frame {
                4 => raw.events.push(egui::Event::PointerMoved(a)),
                5 => raw.events.push(button(a, true)),
                6..=12 => raw.events.push(egui::Event::PointerMoved(a + (b - a) * ((frame - 5) as f32 / 7.0))),
                13 => raw.events.push(button(b, false)),
                _ => {}
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("auto-asmr");
                ui.add_space(12.0);
                let before = self.s.tab;
                ui.selectable_value(&mut self.s.tab, Tab::Realtime, RichText::new(t("  实时（主播）  ", "  Real-time  ")).size(15.0));
                ui.selectable_value(&mut self.s.tab, Tab::File, RichText::new(t("  文件转换  ", "  File  ")).size(15.0));
                if before != self.s.tab {
                    self.stop();
                    self.status.clear();
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.selectable_value(&mut self.s.english, true, "English");
                    ui.selectable_value(&mut self.s.english, false, "中文");
                    egui::widgets::global_theme_preference_buttons(ui);
                    if ui.button(t("导出配置", "Export config")).clicked() {
                        self.export_config();
                    }
                });
            });
            ui.add_space(4.0);
        });
        egui::Panel::bottom("bottom").show(ui, |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.hyperlink_to(RichText::new("随意Official").strong().small(), "https://space.bilibili.com/79356601")
                    .on_hover_text(t("打开研发人的 B 站主页", "Open the developer's Bilibili profile"));
                ui.label(RichText::new("研发人 B站").small());
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.add(egui::Label::new(RichText::new(if self.status.is_empty() { t("纯 CPU 处理 · KU100 人工头实测 HRIR（TH Köln，CC BY 4.0）· 请戴耳机", "CPU only · KU100 dummy-head HRIRs (TH Köln, CC BY 4.0) · use headphones").to_string() } else { self.status.clone() }).small()).truncate());
                });
            });
        });
        egui::Panel::left("left").resizable(false).exact_size(470.0).show(ui, |ui| {
            egui::Panel::bottom("left_bottom").resizable(false).show(ui, |ui| {
                Self::ui_denoise(ui, &mut self.s.params);
            });
            egui::ScrollArea::vertical().id_salt("leftscroll").show(ui, |ui| {
                ui.add_space(6.0);
                match self.s.tab {
                    Tab::Realtime => self.ui_realtime(ui),
                    Tab::File => self.ui_file(ui),
                }
            });
        });
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().id_salt("rightscroll").show(ui, |ui| {
                let p = &mut self.s.params;
                ui.columns(2, |c| {
                    Self::params_left(&mut c[0], p);
                    Self::params_right(&mut c[1], p);
                });
            });
        });

        self.sync_params();

        if self.rt.is_some() || self.job.is_some() || self.decoding.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
        // 截图自检：等界面稳定后截一张图存成 PNG，然后退出
        if let Some(f) = self.demo.take() {
            self.set_file(f);
            self.start_s = 4.0;
            self.preview_s = 4.0;
            self.start_preview();
        }
        let wait = if self.file_in.is_some() { 150 } else { 20 };
        let drag_note = self.shot_drag.map(|_| format!("手动位置：方位角 {:.1}°，到头中心 {:.3} m，高度 {:.1}°", self.s.params.manual_az, self.s.params.manual_dist, self.s.params.manual_tilt));
        if let Some((path, frames)) = &mut self.shot {
            *frames += 1;
            if *frames == wait {
                if let Some(n) = &drag_note {
                    eprintln!("{n}");
                }
                let path = path.clone();
                let c2 = ctx.clone();
                ctx.request_screenshot(move |img| {
                    let _ = save_png(&path, &img);
                    c2.send_viewport_cmd(egui::ViewportCommand::Close);
                });
            }
            ctx.request_repaint();
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if self.shot.is_none() {
            eframe::set_value(storage, eframe::APP_KEY, &self.s);
        }
    }
}

fn playback_time(seconds: f64) -> String {
    let seconds = seconds.max(0.0) as u64;
    format!("{}:{:02}", seconds / 60, seconds % 60)
}

fn save_png(path: &PathBuf, img: &egui::ColorImage) -> Result<(), String> {
    let f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), img.size[0] as u32, img.size[1] as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(|e| e.to_string())?;
    let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
    w.write_image_data(&bytes).map_err(|e| e.to_string())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radar_pick_is_the_inverse_of_the_drawing_scale() {
        let rmax = 90.0;
        for &(az, r) in &[(0.0f32, 0.3f32), (90.0, 0.105), (200.0, 0.5), (315.0, 0.8), (270.0, R_MIN)] {
            // 画的时候：方位角顺时针、上 = 正前方，半径 = radar_map(r) * rmax
            let a = az.to_radians();
            let d = egui::vec2(a.sin(), -a.cos()) * (radar_map(r) * rmax);
            let (az2, r2) = radar_pick(d, rmax);
            assert!(((az2 - az + 180.0).rem_euclid(360.0) - 180.0).abs() < 0.05, "az {az} -> {az2}");
            assert!((r2 / r - 1.0).abs() < 1e-3, "r {r} -> {r2}");
        }
        // 拖进头里：停在头外面；拖出图外：停在图的边缘
        assert_eq!(radar_pick(egui::vec2(1.0, 0.0), rmax).1, R_MIN);
        assert!((radar_pick(egui::vec2(0.0, -500.0), rmax).1 - RADAR_MAX_R).abs() < 1e-5);
    }

    #[test]
    fn up_angle_undoes_lift_tilt() {
        // 手动的三个量（方位角, 距离, 高度）→ 三维位置 → 再换回来
        for &(az, dist, tilt) in &[(90.0f32, 0.105f32, 45.0f32), (260.0, 0.15, -30.0), (0.0, 0.5, 30.0), (135.0, 0.3, 60.0), (300.0, 0.12, 0.0)] {
            let (a3, e3, r3) = lift_tilt(az, dist, 0.0, tilt);
            let (up, az2, dist2) = up_angle(a3, e3, r3);
            assert!((up - tilt).abs() < 0.2, "tilt {tilt} -> {up}");
            assert!(((az2 - az + 180.0).rem_euclid(360.0) - 180.0).abs() < 0.2, "az {az} -> {az2}");
            assert!((dist2 - dist).abs() < 5e-4, "dist {dist} -> {dist2}");
        }
    }
}
