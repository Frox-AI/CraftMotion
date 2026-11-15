"""Agent panel: MCP connection info, local bridge control, live tool-call log."""
from __future__ import annotations

import json
import sys
import tkinter as tk
from typing import List

from ..core.agent_bridge import AgentBridge
from ..core.registry import COMMANDS
from .theme import C, ScrollFrame, font, icon_button, label, mono, section


class AgentPanel(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=C.panel)
        self.app = app
        self.bridge = AgentBridge(app.reg, on_call=self._on_call)
        self._log_lines: List[str] = []

        head = tk.Frame(self, bg=C.panel, height=28)
        head.pack(fill="x")
        head.pack_propagate(False)
        tk.Label(head, text="AI AGENTS", bg=C.panel, fg=C.dim, font=font(8, True)).pack(side="left", padx=10)
        self.count_label = tk.Label(head, text="", bg=C.panel, fg=C.dim, font=mono(8))
        self.count_label.pack(side="right", padx=8)
        self.count_label.configure(text=f"{len(COMMANDS)} tools")

        body = ScrollFrame(self, bg=C.panel)
        body.pack(fill="both", expand=True)
        b = body.inner

        sec = section(b, "Both modes, one engine")
        label(sec, "Human mode = this window.\nAgent mode = MCP over stdio or HTTP.\n"
                   "Agents get every tool you have — no reduced toolset.", C.panel, C.muted, 9,
              justify="left", wraplength=290).pack(anchor="w")

        sec = section(b, "Connect an AI agent (stdio)")
        label(sec, "Add this to your MCP client config (Claude Desktop / Cursor / VS Code).",
              C.panel, C.muted, 8, wraplength=290, justify="left").pack(anchor="w")
        cfg = self._stdio_config()
        txt = tk.Text(sec, height=9, bg=C.panel2, fg=C.text, bd=0, font=mono(8), padx=6, pady=4)
        txt.insert("1.0", cfg)
        txt.configure(state="disabled")
        txt.pack(fill="x", pady=(4, 4))
        row = tk.Frame(sec, bg=C.panel)
        row.pack(fill="x")
        icon_button(row, "Copy config", lambda: self._copy(cfg), bg=C.accent, fg="#141414", size=8
                    ).pack(side="left")
        icon_button(row, "Save config file…", self._save_config, bg=C.panel2, size=8
                    ).pack(side="left", padx=4)

        sec = section(b, "Live bridge (HTTP)")
        label(sec, "Lets an agent edit this running window in real time.", C.panel, C.muted, 8,
              wraplength=290, justify="left").pack(anchor="w")
        self.bridge_status = label(sec, "Stopped", C.panel, C.dim, 9)
        self.bridge_status.pack(anchor="w", pady=(4, 2))
        self.bridge_info = tk.Text(sec, height=3, bg=C.panel2, fg=C.text, bd=0, font=mono(7),
                                   padx=6, pady=4)
        self.bridge_info.pack(fill="x")
        self.bridge_info.configure(state="disabled")
        row = tk.Frame(sec, bg=C.panel)
        row.pack(fill="x", pady=(4, 0))
        self.toggle_btn = icon_button(row, "Start bridge", self.toggle_bridge, bg=C.panel2, size=8)
        self.toggle_btn.pack(side="left")
        self.copy_btn = icon_button(row, "Copy endpoint", self.copy_endpoint, bg=C.panel2, size=8)
        self.copy_btn.pack(side="left", padx=4)

        sec = section(b, "Ready-made prompts")
        for name, desc in (("edit_like_a_human", "Footage → finished MP4, step by step"),
                           ("apply_am_preset", "Import an Alight Motion XML onto footage"),
                           ("review_render", "Storyboard, critique, fix, repeat")):
            label(sec, f"• {name} — {desc}", C.panel, C.muted, 8, wraplength=290,
                  justify="left").pack(anchor="w")

        sec = section(b, "Agent activity")
        self.log_text = tk.Text(sec, height=8, bg="#0c0e12", fg=C.good, bd=0, font=mono(7),
                                padx=6, pady=4)
        self.log_text.pack(fill="x")
        self.log_text.insert("1.0", "waiting for agent calls…\n")
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------ helpers
    def _cmd_path(self) -> List[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, "mcp"]
        return [sys.executable, "-m", "craftmotion.mcp"]

    def _stdio_config(self) -> str:
        cmd = self._cmd_path()
        proj = self.app.reg.project.path
        env = {"CRAFTMOTION_PROJECT": proj} if proj else {}
        cfg = {
            "mcpServers": {
                "craftmotion": {
                    "command": cmd[0],
                    "args": cmd[1:],
                    **({"env": env} if env else {}),
                }
            }
        }
        return json.dumps(cfg, indent=2)

    def _copy(self, text: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.app.log("Copied to clipboard")

    def _save_config(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(parent=self.winfo_toplevel(), defaultextension=".json",
                                            initialfile="craftmotion-mcp.json",
                                            filetypes=[("JSON", "*.json")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._stdio_config())
            self.app.log(f"Saved MCP config → {path}")

    def toggle_bridge(self):
        if self.bridge.running:
            self.bridge.stop()
            self.toggle_btn.configure(text="Start bridge")
            self.bridge_status.configure(text="Stopped", fg=C.dim)
            self._set_info("")
            self.app.log("Agent bridge stopped")
        else:
            info = self.bridge.start()
            self.toggle_btn.configure(text="Stop bridge")
            self.bridge_status.configure(text=f"● Live on port {info.get('port', self.bridge.port)}",
                                         fg=C.good)
            self._set_info(f"URL:   {self.bridge.url}\nAuth:  Bearer {self.bridge.token}\n"
                           f"Tools: {len(COMMANDS)}")
            self.app.log(f"Agent bridge live at {self.bridge.url}")

    def copy_endpoint(self):
        self._copy(f"{self.bridge.url}\nBearer {self.bridge.token}")

    def _set_info(self, text: str):
        self.bridge_info.configure(state="normal")
        self.bridge_info.delete("1.0", "end")
        self.bridge_info.insert("1.0", text)
        self.bridge_info.configure(state="disabled")

    def _on_call(self, tool: str):
        def add():
            self._log_lines.append(tool)
            self._log_lines = self._log_lines[-200:]
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.insert("1.0", "\n".join(self._log_lines[-40:]))
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        try:
            self.app.post(add)
        except Exception:
            pass
