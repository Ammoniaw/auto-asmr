// 弹窗：开始 / 停止处理当前标签页，常用的几个开关，实时显示声源的位置。

import { loadAsmr, loadStored, saveStored } from './asmr.js';

const $ = (id) => document.getElementById(id);
const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
$('tab').textContent = tab ? tab.title || tab.url || '' : '没有标签页';
$('tab').title = $('tab').textContent;

let active = false;
let busy = false;

// ------------------------------------------------------------------ 开始 / 停止

async function toggle() {
  if (!tab || busy) return;
  busy = true;
  $('toggle').disabled = true;
  try {
    if (active) {
      await chrome.runtime.sendMessage({ to: 'bg', type: 'stop', tabId: tab.id });
    } else {
      // 先让后台取 streamId（同 Chrome 官方示例）；后台取不到时在弹窗里取一次再交给它。
      // 两种都要求用户刚点过插件图标（打开这个弹窗就算）。
      let r = await chrome.runtime.sendMessage({ to: 'bg', type: 'start', tabId: tab.id });
      if (r && r.needStreamId) {
        const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });
        r = await chrome.runtime.sendMessage({ to: 'bg', type: 'start', tabId: tab.id, streamId });
      }
      if (!r || r.ok === false) throw new Error((r && r.error) || '没有回应');
    }
    await poll();
  } catch (e) {
    showError(e);
  } finally {
    busy = false;
    $('toggle').disabled = false;
  }
}

function showError(e) {
  let text = String((e && e.message) || e);
  if (/chrome:\/\/|Chrome pages cannot be captured|cannot be captured/i.test(text)) text = 'Chrome 自己的页面（chrome://、扩展商店等）不能处理';
  else if (/active stream/i.test(text)) text = '这个标签页的声音已经被别的程序采集了';
  $('status').innerHTML = '';
  $('status').append(Object.assign(document.createElement('span'), { className: 'error', textContent: text }));
}

$('toggle').addEventListener('click', toggle);

// ------------------------------------------------------------------ 状态

const side = (az) => {
  const a = ((az % 360) + 360) % 360;
  if (a < 20 || a > 340) return '正前方';
  if (a > 160 && a < 200) return '脑后';
  return a < 180 ? '右侧' : '左侧';
};

async function poll() {
  let st;
  try {
    st = await chrome.runtime.sendMessage({ to: 'bg', type: 'status' });
  } catch {
    return;
  }
  const me = st && st.tabs && tab ? st.tabs[tab.id] : null;
  active = !!me;
  $('toggle').textContent = active ? '停止处理' : '开始处理这个标签页的声音';
  $('toggle').className = active ? '' : 'primary';
  if (!busy) $('toggle').disabled = !tab;
  const others = Object.keys((st && st.tabs) || {}).filter((id) => !tab || +id !== tab.id).length;
  const lines = [];
  if (me) {
    if (me.error) lines.push(['error', me.error]);
    if (me.tele) {
      const t = me.tele;
      const cm = Math.max(0, (t.r - 0.0875) * 100);
      lines.push(['', `${side(t.az)} · 离头 ${cm < 10 ? cm.toFixed(1) : cm.toFixed(0)} cm${t.depthCm > 0.05 ? ` · 入耳 ${t.depthCm.toFixed(1)} cm` : ''}`]);
      lines.push(['muted', `说话 ${(t.act * 100).toFixed(0)}% · 输出 ${t.outDb > -100 ? t.outDb.toFixed(0) : '—'} dB`]);
      lines.push(['muted', `延时：引擎 ${t.engineMs.toFixed(1)} ms + 输出 ${st.outputMs.toFixed(0)} ms`]);
    } else {
      lines.push(['muted', '正在启动…']);
    }
    drawRadar(me.tele);
  } else {
    lines.push(['muted', '没在处理。开始后这个标签页的声音会变成双耳 ASMR 效果再播放。']);
    drawRadar(null);
  }
  if (others) lines.push(['muted', `另外还在处理 ${others} 个标签页`]);
  $('status').innerHTML = '';
  for (const [cls, text] of lines) $('status').append(Object.assign(document.createElement('div'), { className: cls, textContent: text }));
}

// 位置图：头在中间，上 = 前，右 = 右耳；距离按对数刻度（同桌面版：2 cm 在中心，80 cm 在边上）
function drawRadar(t) {
  const c = $('radar');
  const g = c.getContext('2d');
  const W = c.width;
  const css = getComputedStyle(document.documentElement);
  const line = css.getPropertyValue('--line').trim();
  const muted = css.getPropertyValue('--muted').trim();
  const accent = css.getPropertyValue('--accent').trim();
  g.clearRect(0, 0, W, W);
  const R = W / 2 - 6;
  const map = (r) => Math.min(1, Math.max(0, Math.log(Math.max(r / 0.02, 1)) / Math.log(0.8 / 0.02)));
  g.strokeStyle = line;
  g.lineWidth = 2;
  for (const r of [0.1, 0.3, 0.8]) {
    g.beginPath();
    g.arc(W / 2, W / 2, map(r) * R, 0, 2 * Math.PI);
    g.stroke();
  }
  // 头和鼻子
  g.fillStyle = muted;
  g.globalAlpha = 0.35;
  g.beginPath();
  g.arc(W / 2, W / 2, map(0.0875) * R, 0, 2 * Math.PI);
  g.fill();
  g.globalAlpha = 1;
  g.beginPath();
  g.moveTo(W / 2 - 6, W / 2 - map(0.0875) * R + 2);
  g.lineTo(W / 2, W / 2 - map(0.0875) * R - 8);
  g.lineTo(W / 2 + 6, W / 2 - map(0.0875) * R + 2);
  g.fill();
  if (!t) return;
  const a = (t.az * Math.PI) / 180;
  const k = map(t.r) * R;
  g.fillStyle = accent;
  g.beginPath();
  g.arc(W / 2 + Math.sin(a) * k, W / 2 - Math.cos(a) * k, 9 + 5 * Math.min(1, t.act), 0, 2 * Math.PI);
  g.fill();
}

// ------------------------------------------------------------------ 常用开关（和设置页改的是同一份配置）

const asmr = await loadAsmr();
let lastText = await loadStored(asmr);
let state = asmr.ui();

function render() {
  for (const b of document.querySelectorAll('#dry button')) b.classList.toggle('on', String(state.dry) === b.dataset.dry);
  for (const b of document.querySelectorAll('#mode button')) b.classList.toggle('on', state.params.mode === b.dataset.mode);
  if (document.activeElement !== $('rain')) $('rain').value = state.params.bg_level;
  $('rain-on').classList.toggle('on', state.params.bg_level > 0);
}

function change(key, value) {
  try {
    asmr.set(key, value);
  } catch (e) {
    showError(e);
    return;
  }
  state = asmr.ui();
  render();
  const s = saveStored(asmr);
  lastText = s.text;
}

for (const b of document.querySelectorAll('#dry button')) b.addEventListener('click', () => change('dry', b.dataset.dry === 'true'));
for (const b of document.querySelectorAll('[data-style]')) b.addEventListener('click', () => change('style', +b.dataset.style));
for (const b of document.querySelectorAll('#mode button')) b.addEventListener('click', () => change('mode', b.dataset.mode));
$('rain-on').addEventListener('click', () => change('bg_preset', state.params.bg_level > 0 ? 'off' : 'rain_night'));
$('rain').addEventListener('change', () => change('bg_level', +$('rain').value));

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'local' || !changes.config) return;
  const text = changes.config.newValue || '';
  if (text === lastText) return;
  lastText = text;
  try {
    if (text) asmr.load(text);
    else asmr.resetParams();
  } catch {
    return;
  }
  state = asmr.ui();
  render();
});

$('options').addEventListener('click', () => chrome.runtime.openOptionsPage());
chrome.commands.getAll().then((cmds) => {
  const c = cmds.find((x) => x.name === 'toggle');
  if (c && c.shortcut) $('key').textContent = c.shortcut;
  else if (c) $('key').textContent = '（未设置）';
});

render();
await poll();
setInterval(poll, 400);
