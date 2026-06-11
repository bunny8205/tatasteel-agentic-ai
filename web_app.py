"""Standalone web app for the Tata Steel Agentic AI.

This avoids Streamlit entirely. It serves a plain HTML/CSS/JS frontend and a
small JSON API backed by the existing MaintenanceWizard agent.

Run:
    python web_app.py --port 8600
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent import MaintenanceWizard  # noqa: E402


WIZARD: MaintenanceWizard | None = None


def get_wizard() -> MaintenanceWizard:
    global WIZARD
    if WIZARD is None:
        wizard = MaintenanceWizard()
        wizard.initialize(load_llm=False)
        WIZARD = wizard
    return WIZARD


def jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tata Steel Agentic AI</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --panel-2: #f9fafb;
      --text: #111827;
      --muted: #5b6575;
      --line: #d9dee7;
      --blue: #1f4e79;
      --blue-2: #2563eb;
      --red: #b91c1c;
      --amber: #b45309;
      --green: #047857;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .shell {
      display: grid;
      grid-template-columns: 300px minmax(480px, 1fr) 420px;
      min-height: 100vh;
    }
    aside, main, .workspace { min-width: 0; }
    aside {
      background: #ffffff;
      border-right: 1px solid var(--line);
      padding: 18px 16px;
      overflow-y: auto;
    }
    main {
      display: flex;
      flex-direction: column;
      min-height: 100vh;
      background: #fbfcfe;
    }
    .workspace {
      background: #ffffff;
      border-left: 1px solid var(--line);
      padding: 16px;
      overflow-y: auto;
      max-height: 100vh;
    }
    .brand {
      font-weight: 800;
      font-size: 20px;
      color: var(--blue);
      margin-bottom: 4px;
    }
    .subtitle { color: var(--muted); font-size: 13px; line-height: 1.4; }
    .section-title {
      margin: 22px 0 8px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 800;
    }
    .asset {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 8px;
      background: var(--panel-2);
    }
    .asset strong { display: block; font-size: 14px; }
    .asset span { color: var(--muted); font-size: 12px; }
    .risk-critical { border-left: 4px solid var(--red); }
    .risk-high { border-left: 4px solid var(--amber); }
    .risk-medium { border-left: 4px solid var(--blue-2); }
    .starter {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      padding: 10px;
      border-radius: 8px;
      margin-bottom: 8px;
      cursor: pointer;
      font-size: 13px;
      line-height: 1.35;
    }
    .starter:hover { border-color: var(--blue-2); background: #f8fbff; }
    .topbar {
      border-bottom: 1px solid var(--line);
      padding: 16px 22px;
      background: #ffffff;
    }
    .topbar h1 { margin: 0; font-size: 22px; color: #111827; }
    .topbar p { margin: 4px 0 0; color: var(--muted); font-size: 14px; }
    .chat {
      flex: 1;
      padding: 20px 22px 120px;
      overflow-y: auto;
    }
    .msg {
      display: grid;
      grid-template-columns: 34px minmax(0, 1fr);
      gap: 10px;
      margin-bottom: 16px;
      align-items: start;
    }
    .avatar {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      font-weight: 800;
      color: white;
      background: var(--blue);
      font-size: 12px;
    }
    .msg.user .avatar { background: #374151; }
    .bubble {
      border: 1px solid var(--line);
      background: #ffffff;
      border-radius: 8px;
      padding: 12px 14px;
      line-height: 1.48;
      white-space: pre-wrap;
    }
    .msg.user .bubble { background: #eef4ff; border-color: #c8d8ff; }
    .composer {
      position: fixed;
      left: 300px;
      right: 420px;
      bottom: 0;
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(8px);
      padding: 14px 22px;
    }
    .composer-inner {
      display: grid;
      grid-template-columns: 1fr 92px;
      gap: 10px;
    }
    textarea {
      width: 100%;
      min-height: 54px;
      max-height: 140px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      outline: none;
    }
    textarea:focus { border-color: var(--blue-2); box-shadow: 0 0 0 3px rgba(37,99,235,0.12); }
    button.primary {
      border: none;
      border-radius: 8px;
      background: var(--blue);
      color: white;
      font-weight: 800;
      cursor: pointer;
    }
    button.primary:disabled { opacity: 0.55; cursor: wait; }
    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      margin-bottom: 12px;
    }
    .card h3 { margin: 0 0 8px; font-size: 15px; color: #111827; }
    .kv {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 6px 10px;
      font-size: 13px;
    }
    .kv b { color: var(--muted); }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      border-bottom: 1px solid #eef0f4;
      text-align: left;
      padding: 7px 6px;
      vertical-align: top;
    }
    th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
    pre {
      margin: 0;
      overflow-x: auto;
      background: #0b1220;
      color: #dbeafe;
      padding: 10px;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.45;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      margin-top: 10px;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); }
    .hidden { display: none; }
    @media (max-width: 1180px) {
      .shell { grid-template-columns: 260px 1fr; }
      .workspace { display: none; }
      .composer { left: 260px; right: 0; }
    }
    @media (max-width: 780px) {
      .shell { display: block; }
      aside { display: none; }
      .composer { left: 0; right: 0; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">Tata Steel Agentic AI</div>
      <div class="subtitle">Autonomous maintenance decision support for steel plant assets.</div>
      <div class="status"><span class="dot"></span><span id="healthText">Starting agent...</span></div>

      <div class="section-title">Live Assets</div>
      <div id="assetList"></div>

      <div class="section-title">Prompt Starters</div>
      <button class="starter">If I can maintain only one asset today, which one should I choose and why?</button>
      <button class="starter">Create a P1 alert report for GBX-17 abnormal vibration.</button>
      <button class="starter">MTR-204 is overheating. Diagnose root cause and give inspection plan.</button>
      <button class="starter">Design an agentic workflow for steel plant predictive maintenance using logs, SOPs, sensor alerts, and feedback.</button>
      <button class="starter">Plan spares and procurement for BOF trunnion bearing maintenance.</button>
    </aside>

    <main>
      <div class="topbar">
        <h1>Steel Maintenance Agent</h1>
        <p>Ask any maintenance, reliability, safety, spares, SOP, RCA, or plant-priority question.</p>
      </div>
      <div class="chat" id="chat">
        <div class="msg assistant">
          <div class="avatar">AI</div>
          <div class="bubble">Ask any steel-plant maintenance, operations, safety, spares, RCA, SOP, quality, or reliability question.</div>
        </div>
      </div>
      <div class="composer">
        <div class="composer-inner">
          <textarea id="prompt" placeholder="Ask the steel agent"></textarea>
          <button class="primary" id="send">Send</button>
        </div>
      </div>
    </main>

    <section class="workspace">
      <div class="card">
        <h3>Decision Packet</h3>
        <div id="packet" class="kv"><b>Status</b><span>No decision yet</span></div>
      </div>
      <div class="card">
        <h3>Agent Plan</h3>
        <div id="plan">No plan yet</div>
      </div>
      <div class="card">
        <h3>Tool Calls</h3>
        <div id="tools">No tool calls yet</div>
      </div>
      <div class="card">
        <h3>Verifier Checks</h3>
        <div id="checks">No checks yet</div>
      </div>
      <div class="card">
        <h3>Raw JSON</h3>
        <pre id="raw">{}</pre>
      </div>
    </section>
  </div>

  <script>
    const chat = document.getElementById("chat");
    const promptBox = document.getElementById("prompt");
    const sendBtn = document.getElementById("send");

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function addMessage(role, text) {
      const row = document.createElement("div");
      row.className = `msg ${role}`;
      row.innerHTML = `
        <div class="avatar">${role === "user" ? "YOU" : "AI"}</div>
        <div class="bubble">${escapeHtml(text).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")}</div>
      `;
      chat.appendChild(row);
      chat.scrollTop = chat.scrollHeight;
    }

    function riskClass(risk) {
      const r = String(risk || "").toLowerCase();
      if (r.includes("critical")) return "risk-critical";
      if (r.includes("high")) return "risk-high";
      return "risk-medium";
    }

    function table(rows) {
      if (!rows || !rows.length) return "None";
      const cols = Object.keys(rows[0]).slice(0, 6);
      return `<table><thead><tr>${cols.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>
        <tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${escapeHtml(r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    }

    function renderWorkspace(result) {
      const packet = result.decision_packet || {};
      document.getElementById("packet").innerHTML = Object.entries(packet).slice(0, 12)
        .map(([k, v]) => `<b>${escapeHtml(k)}</b><span>${escapeHtml(v)}</span>`).join("");
      document.getElementById("plan").innerHTML = table(result.agent_plan || []);
      document.getElementById("tools").innerHTML = table(result.tool_calls || []);
      document.getElementById("checks").innerHTML = table(result.verifier_checks || []);
      document.getElementById("raw").textContent = JSON.stringify(result, null, 2);
    }

    async function loadHealth() {
      try {
        const res = await fetch("/api/health");
        const data = await res.json();
        document.getElementById("healthText").textContent = "Agent ready";
        const assets = data.assets || [];
        document.getElementById("assetList").innerHTML = assets.map(a => `
          <div class="asset ${riskClass(a.risk_band)}">
            <strong>${escapeHtml(a.asset_id)} | ${escapeHtml(a.risk_band)}</strong>
            <span>RUL ${Number(a.estimated_rul_days || 0).toFixed(1)}d | Hybrid risk ${Number(a.hybrid_failure_risk || 0).toFixed(3)}</span>
          </div>
        `).join("");
      } catch (err) {
        document.getElementById("healthText").textContent = "Agent health check failed";
      }
    }

    async function sendPrompt() {
      const prompt = promptBox.value.trim();
      if (!prompt) return;
      promptBox.value = "";
      sendBtn.disabled = true;
      addMessage("user", prompt);
      addMessage("assistant", "Thinking: planning, retrieving evidence, checking risk, verifying action plan...");
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, user_id: "maintenance_engineer_01" })
        });
        const data = await res.json();
        chat.lastElementChild.remove();
        if (!res.ok) throw new Error(data.error || "Agent request failed");
        addMessage("assistant", data.answer || data.final_answer || "Agent completed.");
        renderWorkspace(data);
      } catch (err) {
        chat.lastElementChild.remove();
        addMessage("assistant", `The agent hit an error: ${err.message}`);
      } finally {
        sendBtn.disabled = false;
        promptBox.focus();
      }
    }

    sendBtn.addEventListener("click", sendPrompt);
    promptBox.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendPrompt();
      }
    });
    document.querySelectorAll(".starter").forEach(btn => {
      btn.addEventListener("click", () => {
        promptBox.value = btn.textContent.trim();
        promptBox.focus();
      });
    });
    loadHealth();
  </script>
</body>
</html>
"""


class AgentHandler(BaseHTTPRequestHandler):
    server_version = "TataSteelAgentHTTP/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")

    def send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(jsonable(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_html()
            return
        if path == "/api/health":
            try:
                wizard = get_wizard()
                assets = wizard.asset_health_table().sort_values("hybrid_health_score", ascending=False)
                self.send_json({"status": "ok", "assets": assets})
            except Exception as exc:
                self.send_json({"status": "error", "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/chat":
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self.read_json()
            prompt = str(payload.get("prompt", "")).strip()
            user_id = str(payload.get("user_id", "maintenance_engineer_01"))
            if not prompt:
                self.send_json({"error": "Prompt is required"}, HTTPStatus.BAD_REQUEST)
                return

            result = get_wizard().chat(prompt, user_id=user_id)
            self.send_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.send_json(
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "answer": "The agent hit an error while processing this prompt.",
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8600, type=int)
    args = parser.parse_args()

    print("Starting Tata Steel Agentic AI without Streamlit...")
    get_wizard()
    server = ThreadingHTTPServer((args.host, args.port), AgentHandler)
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
