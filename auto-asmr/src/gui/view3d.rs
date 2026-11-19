//! 3D 观察窗：看音源和人头的相对位置（方向、远近、高低）。右键拖动旋转视角，左键拖动音源
//! （没在手动摆位时会自动切到手动，从音源此刻的位置开始拖）。
//!
//! 纯 egui 画笔绘制，不用 GPU 3D：正交投影（头在任何角度下都是一个圆，遮挡判断简单），
//! 先画头后面的东西，再画头，最后画头前面的东西，被头挡住的部分自然被盖住。
//! 窗口关掉时这里的代码根本不运行，不占渲染资源。

use super::{t, up_angle};
use crate::dsp::engine::Telemetry;
use crate::dsp::motion::lift_tilt;
use crate::dsp::params::{MotionMode, Params};
use crate::dsp::{HEAD_A, R_MAX, R_MIN};
use eframe::egui;
use egui::{Color32, Pos2, Shape, Stroke};
use std::f32::consts::TAU;

/// 默认视角：从右后上方看（偏航 −30°，俯仰 20°）。
const DEF_YAW: f32 = -30.0;
const DEF_PITCH: f32 = 20.0;

/// 多音频合成时画的一个音源：位置（头部坐标，米）、颜色、编号、是否在说话。
pub struct Dot {
    pub xyz: [f32; 3],
    pub col: Color32,
    pub label: String,
    pub speaking: bool,
}

/// 画一个音源：到耳朵高度水平面的垂线和影子、从头表面指向它的虚线、实心圆点。返回圆点的屏幕位置、深度和半径。
fn draw_source(ly: &mut Layers, cam: &Cam, src: [f32; 3], col: Color32, speaking: bool, extent: f32) -> (Pos2, f32, f32) {
    let r = HEAD_A;
    let ground = [src[0], src[1], 0.0];
    if src[2].abs() > 0.003 {
        for s in Shape::dashed_line(&[cam.pt(src), cam.pt(ground)], Stroke::new(1.0, col.gamma_multiply(0.7)), 4.0, 3.0) {
            ly.put(cam.depth(ground).max(cam.depth(src)), s);
        }
    }
    let shadow: Vec<Pos2> = (0..14)
        .map(|i| {
            let a = i as f32 / 14.0 * TAU;
            cam.pt([src[0] + 0.012 * a.cos(), src[1] + 0.012 * a.sin(), 0.0])
        })
        .collect();
    ly.put(cam.depth(ground), Shape::convex_polygon(shadow, col.gamma_multiply(0.35), Stroke::NONE));
    let len = (src[0] * src[0] + src[1] * src[1] + src[2] * src[2]).sqrt().max(1e-6);
    let surf = [src[0] / len * r, src[1] / len * r, src[2] / len * r];
    for s in Shape::dashed_line(&[cam.pt(surf), cam.pt(src)], Stroke::new(1.2, col.gamma_multiply(0.8)), 5.0, 3.0) {
        ly.put(cam.depth([(surf[0] + src[0]) * 0.5, (surf[1] + src[1]) * 0.5, (surf[2] + src[2]) * 0.5]), s);
    }
    let (sp, sd) = cam.proj(src);
    let rad = (if speaking { 7.5 } else { 6.0 }) * (1.0 + 0.25 * (sd / extent).clamp(-1.0, 1.0));
    ly.put(sd, Shape::circle_filled(sp, rad, col));
    (sp, sd, rad)
}

pub struct View3d {
    /// 视角：偏航（度，0 = 从脑后看，屏幕右边就是人的右边）、俯仰（度，90 = 从头顶往下看）。
    pub yaw: f32,
    pub pitch: f32,
    /// 画面半宽对应多少米。`zoom` 为 None 时自动跟随音源的远近，滚轮缩放后固定。
    extent: f32,
    zoom: Option<f32>,
    trail: Vec<[f32; 3]>,
}

impl Default for View3d {
    fn default() -> Self {
        View3d { yaw: DEF_YAW, pitch: DEF_PITCH, extent: 0.3, zoom: None, trail: Vec::new() }
    }
}

/// （方位角 度, 仰角 度, 到头中心 米）→ 头部坐标（x 朝右，y 朝前，z 朝上，米）。
pub fn to_xyz(az: f32, el: f32, r: f32) -> [f32; 3] {
    let (a, e) = (az.to_radians(), el.to_radians());
    [r * e.cos() * a.sin(), r * e.cos() * a.cos(), r * e.sin()]
}

/// 正交相机。`depth` > 0 表示在头中心所在、垂直于视线的平面之前（离观察者更近）。
struct Cam {
    cy: f32,
    sy: f32,
    cp: f32,
    sp: f32,
    c: Pos2,
    k: f32,
}

impl Cam {
    fn new(yaw: f32, pitch: f32, c: Pos2, k: f32) -> Self {
        let (y, p) = (yaw.to_radians(), pitch.to_radians());
        Cam { cy: y.cos(), sy: y.sin(), cp: p.cos(), sp: p.sin(), c, k }
    }

    fn proj(&self, p: [f32; 3]) -> (Pos2, f32) {
        let x1 = p[0] * self.cy - p[1] * self.sy;
        let y1 = p[0] * self.sy + p[1] * self.cy;
        let up = p[2] * self.cp + y1 * self.sp;
        let depth = -y1 * self.cp + p[2] * self.sp;
        (self.c + egui::vec2(x1, -up) * self.k, depth)
    }

    fn pt(&self, p: [f32; 3]) -> Pos2 {
        self.proj(p).0
    }

    fn depth(&self, p: [f32; 3]) -> f32 {
        self.proj(p).1
    }

    /// 屏幕上的一段位移（像素）→ 垂直于视线的平面里的三维位移（米）。`proj` 的逆，深度不变。
    fn unproject(&self, d: egui::Vec2) -> [f32; 3] {
        let (rx, uy) = (d.x / self.k, -d.y / self.k);
        // 屏幕向右 = (cy, −sy, 0)；屏幕向上 = (sp·sy, sp·cy, cp)
        [rx * self.cy + uy * self.sp * self.sy, -rx * self.sy + uy * self.sp * self.cy, uy * self.cp]
    }
}

/// 头部坐标 → 手动摆位的三个量（方位角 度, 未抬高时到头中心的距离 米, 以耳朵为支点的高度 度），限制在手动摆位的范围内。
pub fn to_manual(p: [f32; 3]) -> (f32, f32, f32) {
    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-6);
    let az = p[0].atan2(p[1]).to_degrees().rem_euclid(360.0);
    let el = (p[2] / r).clamp(-1.0, 1.0).asin().to_degrees();
    let (up, az2, dist) = up_angle(az, el, r);
    (az2, dist.clamp(R_MIN, R_MAX), up.clamp(-80.0, 85.0))
}

/// 头后面 / 头前面两层。
#[derive(Default)]
struct Layers {
    back: Vec<Shape>,
    front: Vec<Shape>,
}

impl Layers {
    fn put(&mut self, depth: f32, s: Shape) {
        if depth < 0.0 {
            self.back.push(s);
        } else {
            self.front.push(s);
        }
    }

    /// 三维折线，逐段按深度分层；头后面的段颜色淡一些，显出前后。
    fn polyline(&mut self, cam: &Cam, pts: &[[f32; 3]], stroke: Stroke) {
        for w in pts.windows(2) {
            let mid = [(w[0][0] + w[1][0]) * 0.5, (w[0][1] + w[1][1]) * 0.5, (w[0][2] + w[1][2]) * 0.5];
            let d = cam.depth(mid);
            let st = if d < 0.0 { Stroke::new(stroke.width, stroke.color.gamma_multiply(0.55)) } else { stroke };
            self.put(d, Shape::line_segment([cam.pt(w[0]), cam.pt(w[1])], st));
        }
    }
}

fn circle3(n: usize, f: impl Fn(f32) -> [f32; 3]) -> Vec<[f32; 3]> {
    (0..=n).map(|i| f(i as f32 / n as f32 * TAU)).collect()
}

impl View3d {
    pub fn clear_trail(&mut self) {
        self.trail.clear();
    }

    /// 画一帧。`active`：引擎在运行（实时或试听）。左键拖动会改 `params` 里的手动摆位。
    /// `dots` 不为空时是多音频合成：画这几个音源（只看不拖，位置在左边的位置图上拖）。
    pub fn ui(&mut self, ui: &mut egui::Ui, tele: &Telemetry, active: bool, params: &mut Params, dots: &[Dot]) {
        let mix = !dots.is_empty();
        ui.horizontal(|ui| {
            for (zh, en, y, p) in [("背后", "Back", 0.0, 0.0), ("正面", "Front", 180.0, 0.0), ("头顶", "Top", 0.0, 90.0), ("左侧", "Left", 90.0, 0.0), ("右侧", "Right", -90.0, 0.0), ("默认", "Default", DEF_YAW, DEF_PITCH)] {
                if ui.small_button(t(zh, en)).clicked() {
                    (self.yaw, self.pitch) = (y, p);
                }
            }
        });
        let side = ui.available_width().min(ui.available_height().max(220.0)).max(200.0);
        let (resp, painter) = ui.allocate_painter(egui::vec2(ui.available_width().max(side), side), egui::Sense::click_and_drag());
        let rect = resp.rect;
        let painter = painter.with_clip_rect(rect);

        // 右键拖动旋转（抓住离自己近的那一面跟着鼠标走），滚轮缩放，右键双击复位
        let rotating = resp.dragged_by(egui::PointerButton::Secondary);
        if rotating {
            let d = resp.drag_delta();
            self.yaw = (self.yaw + d.x * 0.5 + 180.0).rem_euclid(360.0) - 180.0;
            self.pitch = (self.pitch + d.y * 0.5).clamp(-89.0, 90.0);
        }
        if resp.double_clicked_by(egui::PointerButton::Secondary) {
            (self.yaw, self.pitch, self.zoom) = (DEF_YAW, DEF_PITCH, None);
        }
        let moving = !mix && resp.is_pointer_button_down_on() && ui.input(|i| i.pointer.primary_down());
        if resp.hovered() {
            let s = ui.input(|i| i.smooth_scroll_delta.y);
            if s != 0.0 {
                self.zoom = Some((self.zoom.unwrap_or(self.extent) * (-s * 0.003).exp()).clamp(0.12, 3.0));
            }
            ui.ctx().set_cursor_icon(if rotating {
                egui::CursorIcon::Grabbing
            } else if moving {
                egui::CursorIcon::Move
            } else {
                egui::CursorIcon::Crosshair
            });
        }

        let src = to_xyz(tele.az, tele.el, tele.r);
        if mix {
            self.trail.clear();
        } else if active {
            self.trail.push(src);
            if self.trail.len() > 120 {
                self.trail.remove(0);
            }
        }
        // 自动缩放：让音源（和手动目标）留在画面里，平滑过渡。左键拖动音源时不缩放，否则画面
        // 跟着音源变大变小，鼠标下的位置一直在变，越拖越远
        let manual_xyz = |p: &Params| {
            let (a, e, r) = lift_tilt(p.manual_az, p.manual_dist, 0.0, p.manual_tilt);
            to_xyz(a, e, r)
        };
        if !moving {
            let want = match self.zoom {
                Some(z) => z,
                None => {
                    let norm = |p: &[f32; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    let far = if mix {
                        dots.iter().map(|d| norm(&d.xyz)).fold(0.0f32, f32::max)
                    } else {
                        let tgt = (params.mode == MotionMode::Manual).then(|| manual_xyz(params));
                        self.trail.iter().chain(std::iter::once(&src)).chain(tgt.iter()).map(norm).fold(0.0f32, f32::max)
                    };
                    // 留一段缓冲：音源快出画面或画面空得太多时才缩放，拖完松手画面不会马上跟着变
                    let fit = (far * 1.25).clamp(0.22, 3.2);
                    if fit > self.extent || fit < 0.55 * self.extent { fit } else { self.extent }
                }
            };
            if (self.extent - want).abs() > 1e-4 {
                self.extent += (want - self.extent) * 0.2;
                ui.ctx().request_repaint();
            } else {
                self.extent = want;
            }
        }
        let cam = Cam::new(self.yaw, self.pitch, rect.center() + egui::vec2(0.0, 8.0), 0.44 * rect.width().min(rect.height()) / self.extent);

        // 左键：把音源放到鼠标所在处（在穿过音源、垂直于视线的平面上移动，深度不变）
        if moving {
            if params.mode != MotionMode::Manual {
                // 切到手动，从音源此刻的位置开始
                (params.manual_az, params.manual_dist, params.manual_tilt) = to_manual(src);
                params.mode = MotionMode::Manual;
                params.level_follow = Params::default_level_follow(MotionMode::Manual);
            }
            if let Some(pt) = resp.interact_pointer_pos() {
                let cur = manual_xyz(params);
                let d = cam.unproject(pt - cam.pt(cur));
                (params.manual_az, params.manual_dist, params.manual_tilt) = to_manual([cur[0] + d[0], cur[1] + d[1], cur[2] + d[2]]);
            }
        }
        let tgt = (!mix && params.mode == MotionMode::Manual).then(|| manual_xyz(params));

        let dark = ui.visuals().dark_mode;
        let grid = if dark { Color32::from_gray(75) } else { Color32::from_gray(185) };
        let fg = if dark { Color32::from_gray(200) } else { Color32::from_gray(60) };
        painter.rect_filled(rect, 8.0, if dark { Color32::from_gray(24) } else { Color32::from_gray(245) });
        let mut ly = Layers::default();
        let r = HEAD_A;

        // 耳朵高度的水平面：距离圈和前后左右
        let mut labels: Vec<([f32; 3], String)> = Vec::new();
        for rr in [0.1f32, 0.3, 0.6, 1.0, 2.0] {
            if rr > self.extent * 1.6 {
                break;
            }
            ly.polyline(&cam, &circle3(72, |a| [rr * a.cos(), rr * a.sin(), 0.0]), Stroke::new(1.0, grid));
            let a = 0.75 * std::f32::consts::PI;
            labels.push(([rr * a.cos(), rr * a.sin(), 0.0], format!("{:.0} cm", rr * 100.0)));
        }
        let ax = (self.extent * 1.15).max(0.15);
        ly.polyline(&cam, &[[0.0, r, 0.0], [0.0, ax, 0.0]], Stroke::new(1.0, grid));
        ly.polyline(&cam, &[[r, 0.0, 0.0], [ax, 0.0, 0.0]], Stroke::new(1.0, grid));
        ly.polyline(&cam, &[[-r, 0.0, 0.0], [-ax, 0.0, 0.0]], Stroke::new(1.0, grid));
        for (p, zh, en) in [([0.0, ax * 1.05, 0.0], "前", "front"), ([0.0, -ax * 1.05, 0.0], "后", "back"), ([-ax * 1.05, 0.0, 0.0], "左", "L"), ([ax * 1.05, 0.0, 0.0], "右", "R")] {
            labels.push((p, t(zh, en).to_string()));
        }

        // 耳朵（贴在头两侧的小椭圆）和鼻子（朝前的三角）
        let ear_col = if dark { Color32::from_gray(150) } else { Color32::from_gray(110) };
        for sx in [-1.0f32, 1.0] {
            let poly: Vec<Pos2> = (0..16).map(|i| {
                let a = i as f32 / 16.0 * TAU;
                cam.pt([sx * r * 1.04, 0.22 * r * a.cos() - 0.05 * r, 0.32 * r * a.sin()])
            }).collect();
            ly.put(cam.depth([sx, 0.0, 0.0]), Shape::convex_polygon(poly, ear_col, Stroke::NONE));
        }
        // 鼻子是一个小楔子：正面看是左右的三角，侧面看是上下的三角，任何角度都看得到
        let tip = cam.pt([0.0, 1.32 * r, -0.12 * r]);
        let nd = cam.depth([0.0, 1.0, 0.0]);
        ly.put(nd, Shape::convex_polygon(vec![cam.pt([-0.2 * r, 0.96 * r, -0.08 * r]), tip, cam.pt([0.2 * r, 0.96 * r, -0.08 * r])], fg, Stroke::NONE));
        ly.put(nd, Shape::convex_polygon(vec![cam.pt([0.0, 0.99 * r, 0.12 * r]), tip, cam.pt([0.0, 0.94 * r, -0.32 * r])], fg, Stroke::NONE));

        // 轨迹
        let n = self.trail.len();
        for (i, w) in self.trail.windows(2).enumerate() {
            let a = (i as f32 / n.max(1) as f32 * 170.0) as u8;
            ly.polyline(&cam, w, Stroke::new(2.0, Color32::from_rgba_unmultiplied(255, 140, 60, a)));
        }

        // 音源：到地面（耳朵高度的水平面）的垂线和影子、从头指向音源的虚线。合成时每个音源各画一个
        let speaking = active && tele.act > 0.5;
        let col = if !active { Color32::from_gray(140) } else if speaking { Color32::from_rgb(255, 120, 40) } else { Color32::from_rgb(190, 130, 95) };
        let mut drawn: Vec<(Pos2, f32, f32, Color32)> = Vec::new(); // 屏幕位置、深度、半径、颜色
        if mix {
            for d in dots {
                let c = if active { d.col } else { d.col.gamma_multiply(0.8) };
                let (p, dd, rad) = draw_source(&mut ly, &cam, d.xyz, c, d.speaking, self.extent);
                drawn.push((p, dd, rad, c));
            }
        } else {
            let (sp, sd, rad) = draw_source(&mut ly, &cam, src, col, speaking, self.extent);
            drawn.push((sp, sd, rad, col));
        }
        let len = (src[0] * src[0] + src[1] * src[1] + src[2] * src[2]).sqrt().max(1e-6);
        let accent = Color32::from_rgb(70, 150, 240);
        if let Some(tp) = tgt {
            let (p, d) = cam.proj(tp);
            ly.put(d, Shape::circle_stroke(p, 11.0, Stroke::new(1.5, accent)));
        }

        // 合成：头后面 → 头 → 头前面
        painter.extend(ly.back);
        let hc = cam.c;
        let hr = r * cam.k;
        painter.circle_filled(hc, hr, if dark { Color32::from_gray(58) } else { Color32::from_gray(212) });
        painter.circle_filled(hc + egui::vec2(-0.28, -0.3) * hr, 0.55 * hr, Color32::from_white_alpha(if dark { 10 } else { 40 }));
        painter.circle_stroke(hc, hr, Stroke::new(1.5, fg));
        // 头上的三条大圆（耳朵高度的一圈、前后中线、过两耳的一圈），只画朝向观察者的半边
        let gc = fg.gamma_multiply(0.35);
        for ring in [circle3(64, |a| [r * a.cos(), r * a.sin(), 0.0]), circle3(64, |a| [0.0, r * a.cos(), r * a.sin()]), circle3(64, |a| [r * a.cos(), 0.0, r * a.sin()])] {
            for w in ring.windows(2) {
                if cam.depth(w[0]) >= 0.0 && cam.depth(w[1]) >= 0.0 {
                    painter.line_segment([cam.pt(w[0]), cam.pt(w[1])], Stroke::new(1.0, gc));
                }
            }
        }
        painter.extend(ly.front);
        // 音源在头后面被挡住时，画一个虚的圈，仍然看得到它在哪
        for &(sp, sd, rad, c) in &drawn {
            if sd < 0.0 && (sp - hc).length() < hr {
                painter.circle_stroke(sp, rad, Stroke::new(1.2, c.gamma_multiply(0.8)));
            }
        }
        // 合成：每个音源旁边写上编号（与左边列表、位置图一致）
        for (d, &(sp, _, rad, _)) in dots.iter().zip(&drawn) {
            painter.text(sp + egui::vec2(rad + 2.0, -rad - 2.0), egui::Align2::LEFT_BOTTOM, &d.label, egui::FontId::proportional(11.0), d.col);
        }

        let small = egui::FontId::proportional(10.0);
        for (p3, s) in labels {
            let (p, d) = cam.proj(p3);
            if d < 0.0 && (p - hc).length() < hr + 6.0 {
                continue; // 被头挡住的刻度文字不画
            }
            painter.text(p, egui::Align2::CENTER_CENTER, s, small.clone(), grid);
        }
        if mix {
            painter.text(rect.left_top() + egui::vec2(8.0, 6.0), egui::Align2::LEFT_TOP, format!("{} {} {}", t("多音频合成：", "Mixing "), dots.len(), t("个音频", "files")), egui::FontId::proportional(12.0), fg);
            painter.text(rect.left_bottom() + egui::vec2(8.0, -5.0), egui::Align2::LEFT_BOTTOM, t("右键拖视角 · 滚轮缩放 · 右键双击复位 · 位置在左边的位置图上拖", "right: rotate · scroll: zoom · right double-click: reset · move sources on the map"), egui::FontId::proportional(10.5), grid);
            return;
        }
        // 读数与左边的位置图一致：高度以耳朵为支点（声源在正前 / 正后方时以头中心），离耳 = 未抬高时离头表面的距离
        let (up, _, plan_r) = up_angle(tele.az, tele.el, tele.r);
        let hgt = if up.abs() < 1.0 { t("与耳齐平", "ear level").to_string() } else { format!("{} {:.0}°", if up > 0.0 { t("上", "up") } else { t("下", "down") }, up.abs()) };
        painter.text(
            rect.left_top() + egui::vec2(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            format!("{} {:.0}°   {}   {} {:.1} cm", t("方位", "az"), tele.az, hgt, t("离耳", "from ear"), ((plan_r - r) * 100.0).max(0.0)),
            egui::FontId::proportional(12.0),
            fg,
        );
        painter.text(
            rect.left_top() + egui::vec2(8.0, 23.0),
            egui::Align2::LEFT_TOP,
            format!("{}{:+.0}°，{} {:.1} cm", t("相对头中心：仰角 ", "from head centre: el "), tele.el, t("距离", "dist"), len * 100.0),
            egui::FontId::proportional(10.5),
            grid,
        );
        painter.text(rect.left_bottom() + egui::vec2(8.0, -5.0), egui::Align2::LEFT_BOTTOM, t("左键拖音源 · 右键拖视角 · 滚轮缩放 · 右键双击复位", "left: move source · right: rotate · scroll: zoom · right double-click: reset"), egui::FontId::proportional(10.5), grid);
        if tgt.is_some() {
            painter.text(rect.right_bottom() + egui::vec2(-8.0, -5.0), egui::Align2::RIGHT_BOTTOM, t("蓝圈 = 手动目标", "blue ring = manual target"), egui::FontId::proportional(10.5), accent);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_presets_put_the_right_ear_where_expected() {
        let c = Pos2::new(0.0, 0.0);
        // 从脑后看：右耳在屏幕右边，前方离观察者远
        let back = Cam::new(0.0, 0.0, c, 100.0);
        assert!(back.pt([1.0, 0.0, 0.0]).x > 0.0);
        assert!(back.depth([0.0, 1.0, 0.0]) < 0.0);
        // 从头顶看：前方在屏幕上方，右边在右边，上方离观察者近
        let top = Cam::new(0.0, 90.0, c, 100.0);
        assert!(top.pt([0.0, 1.0, 0.0]).y < 0.0);
        assert!(top.pt([1.0, 0.0, 0.0]).x > 0.0);
        assert!(top.depth([0.0, 0.0, 1.0]) > 0.0);
        // 从右侧看：右耳朝向观察者，前方在屏幕右边
        let right = Cam::new(-90.0, 0.0, c, 100.0);
        assert!(right.depth([1.0, 0.0, 0.0]) > 0.0);
        assert!(right.pt([0.0, 1.0, 0.0]).x > 0.0);
        // 从正面看：右耳在屏幕左边
        let front = Cam::new(180.0, 0.0, c, 100.0);
        assert!(front.pt([1.0, 0.0, 0.0]).x < 0.0);
        // 坐标换算：方位 90° = 正右，仰角 90° = 正上
        let p = to_xyz(90.0, 0.0, 1.0);
        assert!((p[0] - 1.0).abs() < 1e-6 && p[1].abs() < 1e-6);
        assert!((to_xyz(0.0, 90.0, 1.0)[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dragging_moves_the_point_exactly_under_the_cursor() {
        // 任意视角：把一个点在屏幕上挪一段，反投影回三维，再投影，正好落在挪到的位置，且深度不变
        for &(yaw, pitch) in &[(0.0f32, 0.0f32), (-30.0, 20.0), (150.0, 30.0), (0.0, 90.0), (-90.0, 0.0), (70.0, -45.0)] {
            let cam = Cam::new(yaw, pitch, Pos2::new(200.0, 200.0), 300.0);
            let p = [0.05f32, -0.1, 0.07];
            let (sp, sd) = cam.proj(p);
            let d = egui::vec2(37.0, -21.0);
            let q = cam.unproject(d);
            let (sq, qd) = cam.proj([p[0] + q[0], p[1] + q[1], p[2] + q[2]]);
            assert!((sq - (sp + d)).length() < 1e-3, "yaw {yaw} pitch {pitch}");
            assert!((qd - sd).abs() < 1e-6);
        }
    }

    #[test]
    fn manual_parameters_round_trip_through_3d() {
        for &(az, dist, tilt) in &[(90.0f32, 0.105f32, 45.0f32), (260.0, 0.3, -30.0), (10.0, 0.6, 70.0), (300.0, 0.12, 0.0)] {
            let (a, e, r) = lift_tilt(az, dist, 0.0, tilt);
            let (az2, dist2, tilt2) = to_manual(to_xyz(a, e, r));
            assert!(((az2 - az + 180.0).rem_euclid(360.0) - 180.0).abs() < 0.2, "az {az} -> {az2}");
            assert!((dist2 - dist).abs() < 5e-4, "dist {dist} -> {dist2}");
            assert!((tilt2 - tilt).abs() < 0.2, "tilt {tilt} -> {tilt2}");
        }
    }
}
