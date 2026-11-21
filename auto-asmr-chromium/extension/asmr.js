// 插件页面（弹窗、设置页）用的 wasm 封装：读写配置文档。配置以桌面版的格式（导出配置的 .ron 文本）
// 存在 chrome.storage.local 的 `config` 里；background.js 看到它变了就发给正在处理的标签页。

const enc = new TextEncoder();
const dec = new TextDecoder();

export async function loadAsmr() {
  let instance;
  try {
    ({ instance } = await WebAssembly.instantiateStreaming(fetch('asmr.wasm'), {}));
  } catch {
    ({ instance } = await WebAssembly.instantiate(await (await fetch('asmr.wasm')).arrayBuffer(), {}));
  }
  const w = instance.exports;
  w.asmr_init();

  const withStr = (s, f) => {
    const b = enc.encode(s);
    const p = w.asmr_alloc(b.length);
    new Uint8Array(w.memory.buffer, p, b.length).set(b);
    try {
      return f(p, b.length);
    } finally {
      w.asmr_free(p, b.length);
    }
  };
  const out = (n) => dec.decode(new Uint8Array(w.memory.buffer, w.asmr_out_ptr(), n));
  const error = () => out(w.asmr_error());
  const str = (n) => {
    if (n < 0) throw new Error(error());
    return out(n);
  };

  return {
    /** 读一份配置文件（桌面版格式）；读不懂时抛出原因，文档不变。 */
    load(text) {
      if (withStr(text, (p, n) => w.asmr_doc_load(p, n)) !== 0) throw new Error(error());
    },
    /** 写成配置文件（与桌面版「导出配置」相同）。 */
    text() {
      return str(w.asmr_doc_text());
    },
    /** { params, dry, derived, doc }，见 wasm/src/lib.rs 的 asmr_doc_ui。 */
    ui() {
      return JSON.parse(str(w.asmr_doc_ui()));
    },
    /** 改一项（参数名或特殊名字，见 asmr_doc_set）；值不对时抛出原因。 */
    set(key, value) {
      const r = withStr(key, (kp, kn) => withStr(JSON.stringify(value), (vp, vn) => w.asmr_doc_set(kp, kn, vp, vn)));
      if (r !== 0) throw new Error(error());
    },
    resetParams() {
      w.asmr_doc_reset_params();
    },
    /** { pinnas, headphones }：下拉框的选项。 */
    lists() {
      return JSON.parse(str(w.asmr_lists()));
    },
  };
}

/** 从 chrome.storage 读出配置放进文档；没有存过就用默认值。返回存着的文本（没有时为空串）。 */
export async function loadStored(asmr) {
  const { config } = await chrome.storage.local.get('config');
  if (typeof config === 'string' && config) {
    try {
      asmr.load(config);
    } catch (e) {
      console.warn('[auto-asmr] 存着的配置读不懂，用默认值：', e);
    }
    return config;
  }
  return '';
}

/** 把文档存回 chrome.storage。返回 { text, done }：text 是存进去的文本（立即可用，
 * 好让调用方在 storage.onChanged 回来之前认出这是自己存的），done 在存完时完成。 */
export function saveStored(asmr) {
  const text = asmr.text();
  return { text, done: chrome.storage.local.set({ config: text }) };
}
