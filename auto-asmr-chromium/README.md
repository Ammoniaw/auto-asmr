# auto-asmr-chromium（浏览器插件，开发中）

Chrome / Edge 插件：把标签页里的声音实时做成 KU100 双耳 ASMR 效果再播放。引擎是桌面 Rust 版的同一份代码（编译成 WebAssembly），配置文件与桌面版、安卓版通用。

## 构建

需要 Rust 和它的 wasm 目标（`rustup target add wasm32-unknown-unknown`）：

```bash
./build.sh
```

然后在 `chrome://extensions` 打开「开发者模式」，「加载已解压的扩展程序」选 `extension/` 目录。

## 文档

- [浏览器插件](../doc/chromium.md)：安装、使用、做法、延时、性能、测试、代码结构
