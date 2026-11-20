#!/usr/bin/env python3
"""开发用的本地服务器：在普通浏览器标签页里打开插件的页面和测试页，不用装插件。

    python3 tools/serve.py [端口，默认 8765]

- http://127.0.0.1:8765/tools/worklet-test.html：用离线渲染验证 wasm 引擎 + AudioWorklet（自动跑，结果显示在页面上）
- http://127.0.0.1:8765/extension/options.html、/extension/popup.html：插件界面。打开 extension/ 下的 HTML 时，
  自动在最前面插入 tools/chrome-shim.js（一个假的 chrome.* 接口，配置存在 localStorage 里），界面能照常操作；
  采集标签页的声音只有装成插件才能用。

只监听 127.0.0.1。
"""

import http.server
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SHIM = b'<script src="/tools/chrome-shim.js"></script>\n'


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript"}

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/extension/") and path.endswith(".html"):
            full = os.path.join(ROOT, path.lstrip("/"))
            if os.path.isfile(full):
                with open(full, "rb") as f:
                    body = f.read()
                i = body.find(b"<script")
                body = body[:i] + SHIM + body[i:] if i >= 0 else body + SHIM
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        super().do_GET()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"http://127.0.0.1:{port}/tools/worklet-test.html")
    print(f"http://127.0.0.1:{port}/extension/options.html")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
