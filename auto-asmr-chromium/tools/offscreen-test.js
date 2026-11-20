// offscreen-test.html 的测试步骤：点「开始测试」后按插件里的顺序发消息给 offscreen.js，结果显示在页面上，
// window.testResult 里有详细数据。

const logEl = document.getElementById('log');
const results = [];
const log = (text, cls = '') => {
  const s = document.createElement('span');
  s.className = cls;
  s.textContent = text + '\n';
  logEl.append(s);
};
const check = (name, ok, detail = '') => {
  results.push({ name, ok, detail });
  log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`, ok ? 'pass' : 'fail');
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** 像 chrome.runtime.sendMessage 一样把消息交给 offscreen.js 注册的处理函数，等它回应。 */
function send(msg) {
  return new Promise((resolve) => {
    let answered = false;
    for (const f of window.__listeners) {
      const keep = f(msg, {}, (r) => {
        answered = true;
        resolve(r);
      });
      if (keep === true || answered) return;
    }
    resolve(undefined);
  });
}

async function run() {
  document.getElementById('run').disabled = true;
  check('offscreen.js 注册了消息处理', window.__listeners.length === 1);

  // 和设置页存的一样：桌面版格式的配置文本（这里用直接写的 Saved）
  const t0 = performance.now();
  const r = await send({ to: 'offscreen', type: 'start', tabId: 7, streamId: 'fake', config: '(params: (mode: Sweep, hold_s: 0.8, move_s: 0.8))' });
  check('start 成功', r && r.ok === true, `${(performance.now() - t0).toFixed(0)} ms（含编译 wasm、建 AudioContext）`);
  check('通知后台 started', window.__notes.some((n) => n.type === 'started' && n.tabId === 7));

  // 实时跑 4 秒，期间收集方位
  const az = [];
  let st;
  for (let i = 0; i < 16; i++) {
    await sleep(250);
    st = await send({ to: 'offscreen', type: 'status' });
    const me = st.tabs[7];
    if (me && me.tele) az.push(me.tele.az);
  }
  const me = st.tabs[7];
  check('AudioContext 在运行、48 kHz', st.state === 'running', `state ${st.state}，输出延时 ${st.outputMs.toFixed(1)} ms`);
  check('处理器在报状态、没有出错', !!(me && me.tele) && !me.error, me && me.tele ? `说话 ${(me.tele.act * 100).toFixed(0)}%，输出 ${me.tele.outDb.toFixed(1)} dB，引擎延时 ${me.tele.engineMs.toFixed(1)} ms` : me && me.error);
  const span = az.length ? Math.max(...az) - Math.min(...az) : 0;
  check('声源在左右移动（输入真的进了引擎）', me && me.tele && me.tele.act > 0.05 && span > 60, `方位角范围 ${span.toFixed(0)}°（${az.length} 次）`);

  // 换配置：切到「固定在正前方」，方位角应该停在 0°
  await send({ to: 'offscreen', type: 'config', config: '(params: (mode: Static, static_az: 0.0, micro: 0.0, sway: 0.0, updown: 0.0))' });
  await sleep(2500);
  st = await send({ to: 'offscreen', type: 'status' });
  const a = st.tabs[7] && st.tabs[7].tele ? st.tabs[7].tele.az : NaN;
  const dist = Math.min(Math.abs(a), Math.abs(360 - a));
  check('换配置立即生效（固定在正前方）', dist < 10, `方位角 ${a.toFixed(1)}°`);

  // 停止
  await send({ to: 'offscreen', type: 'stop', tabId: 7 });
  st = await send({ to: 'offscreen', type: 'status' });
  check('stop 之后没有在处理的标签页', Object.keys(st.tabs).length === 0);
  check('通知后台 stopped、idle', window.__notes.some((n) => n.type === 'stopped' && n.tabId === 7) && window.__notes.some((n) => n.type === 'idle'));

  // 再开始一次（同一个 AudioContext、同一个 wasm 实例）
  const r2 = await send({ to: 'offscreen', type: 'start', tabId: 8, streamId: 'fake', config: '' });
  await sleep(1500);
  st = await send({ to: 'offscreen', type: 'status' });
  check('停止后再开始', r2 && r2.ok && st.tabs[8] && st.tabs[8].tele && !st.tabs[8].error);
  await send({ to: 'offscreen', type: 'stop', tabId: 8 });
  if (window.__fakeTab) window.__fakeTab.close();

  const failed = results.filter((x) => !x.ok).length;
  log(failed ? `有 ${failed} 项失败` : `全部通过（${results.length} 项）`, failed ? 'fail' : 'pass');
  window.testResult = { failed, results };
}

document.getElementById('run').addEventListener('click', run);
