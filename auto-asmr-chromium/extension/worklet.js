// AudioWorklet（音频线程）：每个被处理的标签页一个 AsmrProcessor，立体声进、双耳声出。
//
// - 同一个 AudioContext 里的处理器共用一个 wasm 实例（数据表只解开一次）。wasm 模块由 offscreen.js 在主线程编译好，
//   经 processorOptions 传进来，这里同步实例化。
// - 配置：offscreen.js 把配置文件的 UTF-8 字节和一个递增的编号（gen）发过来；同一个编号只应用一次。
//   生成引擎配置在音频线程上做，改参数的那一下可能卡一小下（见 doc/chromium.md）。
// - 出错不影响网页的声音：wasm 里出错（panic）后，所有处理器改成直通，原因报给 offscreen.js；
//   之后新开始处理的标签页会换一个新的 wasm 实例。

let W = null; // 当前 wasm 实例的导出
let inst = 0; // 实例编号：处理器只用建它时的那个实例
let broken = false;
let appliedGen = -1;

function setup(module) {
  if (W && !broken) return;
  W = new WebAssembly.Instance(module, {}).exports;
  inst += 1;
  broken = false;
  appliedGen = -1;
  W.asmr_init();
  // 数据表和雨声数据在这里一次建好（还没出声），之后改参数、开雨声都不用再建
  W.asmr_warm_up();
}

/** 读出 wasm 里记下的出错原因（UTF-8 字节，交给主线程解码；这里没有 TextDecoder）。 */
function wasmError() {
  try {
    const n = W.asmr_error();
    return new Uint8Array(W.memory.buffer, W.asmr_out_ptr(), n).slice();
  } catch {
    return null;
  }
}

/** 应用一份配置（配置文件的 UTF-8 字节；空 = 用默认参数）。返回 null = 成功，否则是出错原因的字节。 */
function applyConfig(bytes, gen) {
  if (gen <= appliedGen) return null;
  appliedGen = gen;
  let err = null;
  if (bytes && bytes.length) {
    const p = W.asmr_alloc(bytes.length);
    new Uint8Array(W.memory.buffer, p, bytes.length).set(bytes);
    if (W.asmr_doc_load(p, bytes.length) !== 0) err = wasmError();
    W.asmr_free(p, bytes.length);
  }
  W.asmr_engine_apply();
  return err;
}

function passthrough(input, out) {
  for (let c = 0; c < out.length; c++) {
    const src = input[c] || input[0];
    if (src) out[c].set(src);
    else out[c].fill(0);
  }
}

const TELE_EVERY = 12000; // 每 0.25 秒报一次状态

class AsmrProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.alive = true;
    this.p = 0;
    this.inst = -1;
    this.sinceTele = 0;
    this.port.onmessage = (e) => this.onMessage(e.data);
    const { module, config, gen } = options.processorOptions;
    try {
      if (sampleRate !== 48000) throw new Error(`AudioContext 的采样率是 ${sampleRate} Hz，引擎只处理 48 kHz`);
      setup(module);
      const err = applyConfig(config, gen);
      if (err) this.port.postMessage({ type: 'config-error', detail: err });
      this.p = W.asmr_proc_new();
      this.inst = inst;
      this.cap = W.asmr_proc_cap();
      this.port.postMessage({ type: 'ready' });
    } catch (e) {
      this.fail(e);
    }
  }

  fail(e) {
    broken = true;
    this.port.postMessage({ type: 'error', message: String((e && e.message) || e), detail: W ? wasmError() : null });
  }

  usable() {
    return this.p && !broken && this.inst === inst;
  }

  onMessage(m) {
    if (m.type === 'config') {
      if (!this.usable()) return;
      try {
        const err = applyConfig(m.config, m.gen);
        if (err) this.port.postMessage({ type: 'config-error', detail: err });
      } catch (e) {
        this.fail(e);
      }
    } else if (m.type === 'free') {
      if (this.usable()) {
        try {
          W.asmr_proc_free(this.p);
        } catch {
          // 释放失败也不要紧：实例出错后整个会被换掉
        }
      }
      this.p = 0;
      this.alive = false;
    }
  }

  /** wasm 内存变大后旧的视图会失效，用前检查一下。 */
  views() {
    if (!this.io || this.io.buffer !== W.memory.buffer) {
      const cap = this.cap;
      const base = W.asmr_proc_io(this.p);
      this.io = new Float32Array(W.memory.buffer, base, 4 * cap);
      this.inL = this.io.subarray(0, cap);
      this.inR = this.io.subarray(cap, 2 * cap);
      this.outL = this.io.subarray(2 * cap, 3 * cap);
      this.outR = this.io.subarray(3 * cap, 4 * cap);
      this.tele = new Float32Array(W.memory.buffer, W.asmr_proc_tele(this.p), W.asmr_tele_len());
    }
  }

  process(inputs, outputs) {
    if (!this.alive) return false;
    const input = inputs[0];
    const out = outputs[0];
    if (!this.usable()) {
      passthrough(input, out);
      return true;
    }
    try {
      const n = out[0].length;
      for (let done = 0; done < n; ) {
        const m = Math.min(n - done, this.cap);
        this.views();
        if (input.length) {
          const l = input[0];
          const r = input[1] || l;
          this.inL.set(l.subarray(done, done + m));
          this.inR.set(r.subarray(done, done + m));
        } else {
          // 还没有输入（标签页的流还没接上）：送静音，引擎照常走（背景雨声不断）
          this.inL.fill(0, 0, m);
          this.inR.fill(0, 0, m);
        }
        W.asmr_proc_process(this.p, m);
        this.views();
        out[0].set(this.outL.subarray(0, m), done);
        if (out[1]) out[1].set(this.outR.subarray(0, m), done);
        done += m;
      }
      this.sinceTele += n;
      if (this.sinceTele >= TELE_EVERY) {
        this.sinceTele = 0;
        this.port.postMessage({ type: 'tele', t: Array.from(this.tele) });
      }
    } catch (e) {
      this.fail(e);
      passthrough(input, out);
    }
    return true;
  }
}

registerProcessor('auto-asmr', AsmrProcessor);
