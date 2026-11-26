"""Stdlib-only HTTP server hosting the mastering audition panel.

Start:  python3 -m saturation.panel
Then open the printed URL (http://localhost:8000) in a browser.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .panel_backend import render, SOURCES, FS
from .transfer import transfer_names

HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>非线性饱和母带试听面板</title>
<style>
  :root { --bg:#14181f; --card:#1d232c; --line:#2b333f; --fg:#e7ecf2;
          --muted:#94a0ae; --accent:#e05c5c; --ok:#4fc08d; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 -apple-system, "Segoe UI", "PingFang SC",
         "Microsoft YaHei", sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:18px 26px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  header h1 { font-size:18px; margin:0; }
  header span { color:var(--muted); font-size:12px; }
  main { display:grid; grid-template-columns:320px 1fr; gap:18px; padding:20px 26px; }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:10px; padding:16px; }
  label { display:block; font-size:12px; color:var(--muted); margin:12px 0 5px; }
  select, input[type=range] { width:100%; }
  select { background:#10141a; color:var(--fg); border:1px solid var(--line);
           border-radius:6px; padding:8px; }
  .row { display:flex; gap:8px; flex-wrap:wrap; }
  .row button { flex:1; }
  button { background:#26303d; color:var(--fg); border:1px solid var(--line);
           border-radius:7px; padding:9px 10px; cursor:pointer; font-size:13px; }
  button:hover { border-color:var(--accent); }
  button.primary { background:var(--accent); border-color:var(--accent);
                   color:#1a0d0d; font-weight:600; }
  button.factor.active { background:var(--accent); color:#1a0d0d;
                         border-color:var(--accent); }
  .val { float:right; color:var(--fg); font-variant-numeric:tabular-nums; }
  #error { display:none; margin-top:12px; padding:10px; border-radius:7px;
           background:#3a1d1d; border:1px solid #6e2f2f; color:#ffb4b4;
           font-size:12.5px; white-space:pre-wrap; }
  .metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:10px;
             margin-top:14px; }
  .metric { background:#10141a; border:1px solid var(--line); border-radius:8px;
            padding:10px; }
  .metric b { display:block; font-size:17px; font-variant-numeric:tabular-nums; }
  .metric small { color:var(--muted); }
  audio { width:100%; margin-top:12px; }
  img { width:100%; border-radius:8px; background:#0d1015; margin-top:12px; }
  .hint { color:var(--muted); font-size:12px; margin-top:10px; }
  .spinner { opacity:.55; pointer-events:none; }
</style>
</head>
<body>
<header>
  <h1>非线性饱和 · 母带试听面板</h1>
  <span>真实多速率链路（零填充 + Kaiser FIR 升降采样）· 采样率 __FS__ Hz · 控件由 SaturationProcessor 内核驱动</span>
</header>
<main>
  <section class="card">
    <label>测试信号源</label>
    <select id="source">__OPTIONS__</select>

    <label>传递函数</label>
    <select id="kind">__KINDS__</select>

    <label>驱动量 drive <span class="val" id="driveVal">4.0</span></label>
    <input id="drive" type="range" min="0.1" max="24" step="0.1" value="4">

    <label>过采样倍率</label>
    <div class="row" id="factors">
      <button data-f="1" class="factor active">x1 关闭</button>
      <button data-f="2" class="factor">x2</button>
      <button data-f="4" class="factor">x4</button>
      <button data-f="8" class="factor">x8</button>
    </div>

    <div class="row" style="margin-top:14px">
      <button id="render" class="primary">渲染并分析</button>
      <button id="ab">A/B 对比 x1↔x8</button>
    </div>
    <div id="error"></div>
    <p class="hint">A/B 会连续渲染 x1 与 x8 两份结果并依次播放，
       便于直接比较混叠与音色；失败信息会回显在控件下方而不是静默。</p>
  </section>

  <section class="card">
    <div id="status" class="hint">调整参数后点击“渲染并分析”。</div>
    <div class="metrics" id="metrics"></div>
    <label>湿声（饱和后）</label>
    <audio id="wet" controls></audio>
    <label>干声（原始信号）</label>
    <audio id="dry" controls></audio>
    <img id="plot" alt="谐波分布图">
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
let factor = 1;
document.querySelectorAll('#factors button').forEach(b => b.onclick = () => {
  document.querySelectorAll('#factors button').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); factor = parseInt(b.dataset.f);
});
$('drive').oninput = () => $('driveVal').textContent = parseFloat($('drive').value).toFixed(1);

function showError(msg) {
  const e = $('error'); e.style.display = 'block'; e.textContent = msg;
}
function clearError() { $('error').style.display = 'none'; }

async function callRender(ovFactor) {
  clearError();
  const body = new URLSearchParams({
    source: $('source').value, kind: $('kind').value,
    drive: $('drive').value, factor: ovFactor || factor,
  });
  $('status').textContent = '内核处理中…';
  document.body.classList.add('spinner');
  const res = await fetch('/render', {method:'POST', body});
  document.body.classList.remove('spinner');
  const data = await res.json();
  if (!data.ok) { showError('饱和内核拒绝该参数组合：\\n' + data.error);
    $('status').textContent = '处理失败，参数已回显，请调整后重试。'; return null; }
  return data;
}
function present(data) {
  const m = data.metrics;
  $('metrics').innerHTML = [
    ['THD', m.thd_db + ' dB'], ['直流偏移', m.dc_out],
    ['输出峰值', m.peak_out], ['RMS', m.rms_db + ' dBFS'],
    ['恒定延迟', m.latency_samples + ' 样本'], ['倍率 / 驱动', 'x'+m.factor+' / '+m.drive],
  ].map(([k,v]) => `<div class="metric"><b>${v}</b><small>${k}</small></div>`).join('');
  $('wet').src = 'data:audio/wav;base64,' + data.wav_b64;
  $('dry').src = 'data:audio/wav;base64,' + data.dry_b64;
  $('plot').src = 'data:image/png;base64,' + data.plot_b64;
  $('status').textContent = '完成：谐波分布与可试听湿声已更新。';
}
$('render').onclick = async () => { const d = await callRender(); if (d) present(d); };
$('ab').onclick = async () => {
  const a = await callRender(1); if (!a) return;
  present(a); $('wet').play();
  $('wet').onended = async () => {
    const b = await callRender(8); if (!b) { $('wet').onended = null; return; }
    present(b); $('wet').play(); $('wet').onended = null;
  };
};
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet console
        pass

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self._send(404, "text/plain", b"not found")
            return
        page = (HTML
                .replace("__FS__", str(FS))
                .replace("__OPTIONS__", "".join(
                    f'<option value="{s}">{s}</option>' for s in SOURCES))
                .replace("__KINDS__", "".join(
                    f'<option value="{k}">{k}</option>'
                    for k in transfer_names())))
        self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))

    def do_POST(self):
        if self.path != "/render":
            self._send(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        params = dict(p.split("=", 1) for p in
                      self.rfile.read(length).decode().split("&"))
        try:
            result = render(
                source=params.get("source", "pluck"),
                drive=float(params.get("drive", 4)),
                factor=int(params.get("factor", 1)),
                kind=params.get("kind", "tanh"),
            )
            payload = json.dumps({
                "ok": result.ok, "error": result.error,
                "wav_b64": result.wav_b64, "dry_b64": result.dry_b64,
                "plot_b64": result.plot_b64, "metrics": result.metrics,
            }).encode()
            self._send(200, "application/json", payload)
        except Exception as exc:
            self._send(200, "application/json", json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                 "wav_b64": "", "dry_b64": "", "plot_b64": "",
                 "metrics": {}}).encode())


def main():
    ap = argparse.ArgumentParser(description="Saturation mastering panel")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"饱和母带试听面板已启动: {url}")
    print("在浏览器打开上面的地址；Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
