// 开发用：在普通网页里模拟插件页面用到的那几个 chrome.* 接口（tools/serve.py 自动插入）。
// 配置存在 localStorage（同一个服务器下的弹窗和设置页共用，改了会互相通知）；状态是假的，
// 采集标签页声音会报错——那只有装成插件才能用。
(() => {
  if (window.chrome && window.chrome.runtime && window.chrome.runtime.id) return;
  const KEY = 'auto-asmr-shim-storage';
  const listeners = [];
  const read = () => {
    try {
      return JSON.parse(localStorage.getItem(KEY) || '{}');
    } catch {
      return {};
    }
  };
  const fire = (changes) => {
    for (const f of listeners) f(changes, 'local');
  };
  window.addEventListener('storage', (e) => {
    if (e.key !== KEY) return;
    const before = JSON.parse(e.oldValue || '{}');
    const after = JSON.parse(e.newValue || '{}');
    const changes = {};
    for (const k of new Set([...Object.keys(before), ...Object.keys(after)])) {
      if (before[k] !== after[k]) changes[k] = { oldValue: before[k], newValue: after[k] };
    }
    if (Object.keys(changes).length) fire(changes);
  });
  const fakeTele = () => {
    const t = Date.now() / 1000;
    // 配置里开着左右双人时，假装有两个声音（2 号在对面）
    const m = /duo_ms:\s*([\d.]+)/.exec(read().config || '');
    const duo = !!m && +m[1] > 0;
    const az = (t * 30) % 360;
    return { az, el: 0, r: 0.105 + 0.05 * (1 + Math.sin(t)), act: 0.5 + 0.5 * Math.sin(t * 3), inDb: -24, outDb: -18, depthCm: 0, engineMs: 2.4, duo, az2: (az + 180) % 360, el2: 0, r2: 0.12, act2: 0.6 };
  };
  let capturing = false;
  window.chrome = {
    storage: {
      local: {
        async get(key) {
          const all = read();
          return key == null ? all : { [key]: all[key] };
        },
        async set(obj) {
          const all = read();
          const changes = {};
          for (const [k, v] of Object.entries(obj)) {
            changes[k] = { oldValue: all[k], newValue: v };
            all[k] = v;
          }
          localStorage.setItem(KEY, JSON.stringify(all));
          setTimeout(() => fire(changes), 0);
        },
      },
      onChanged: { addListener: (f) => listeners.push(f) },
    },
    runtime: {
      id: 'shim',
      async sendMessage(msg) {
        if (msg.type === 'status') return { tabs: capturing ? { 1: { error: '', seconds: 1, tele: fakeTele() } } : {}, outputMs: 12, state: 'running' };
        if (msg.type === 'start') {
          capturing = true;
          return { ok: true };
        }
        if (msg.type === 'stop') {
          capturing = false;
          return { ok: true };
        }
        return { ok: true };
      },
      onMessage: { addListener() {} },
      openOptionsPage() {
        window.open('/extension/options.html');
      },
    },
    tabs: { query: async () => [{ id: 1, title: '（开发预览）一个在放 ASMR 视频的标签页', url: 'https://example.com/' }] },
    tabCapture: { getMediaStreamId: async () => 'fake-stream-id' },
    commands: { getAll: async () => [{ name: 'toggle', shortcut: 'Alt+Shift+A' }] },
  };
})();
