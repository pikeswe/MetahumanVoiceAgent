"""Simple Tkinter control panel for the MetaHuman voice agent."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import requests
import tkinter as tk
from tkinter import messagebox, ttk

from utils import config as config_utils


class ControlPanel:
    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = Path(config_path or Path("config") / "default_config.json")
        self.config = config_utils.load_config(str(self.config_path))
        self.process: Optional[subprocess.Popen] = None
        self._status_poll = False

        orchestrator_cfg = self.config.get("orchestrator", {})
        self.api_host = orchestrator_cfg.get("api_host", "127.0.0.1")
        self.api_port = orchestrator_cfg.get("api_port", 8000)

        tts_cfg = self.config.get("tts", {})
        self.default_backend = tts_cfg.get("backend", "neutts")

        self.root = tk.Tk()
        self.root.title("Unreal Voice Agent Control Panel")
        self.backend_var = tk.StringVar(value=self.default_backend)
        self.status_var = tk.StringVar(value="Stopped")
        self.ws_status_var = tk.StringVar(value="WS: inactive")
        self.backend_status_var = tk.StringVar(value=f"Backend: {self.default_backend}")
        self.prompt_text = tk.Text(self.root, height=5, width=60)
        self.response_var = tk.StringVar(value="")

        self.rate_override = tk.BooleanVar(value=False)
        self.intensity_override = tk.BooleanVar(value=False)
        self.emotion_override_enabled = tk.BooleanVar(value=False)
        self.emotion_value = tk.StringVar(value="neutral")
        self.rate_scale_value = tk.DoubleVar(value=50.0)
        self.intensity_scale_value = tk.DoubleVar(value=50.0)

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ UI SETUP
    def _build_layout(self) -> None:
        top_frame = ttk.Frame(self.root, padding=12)
        top_frame.grid(row=0, column=0, sticky="nsew")

        status_frame = ttk.LabelFrame(top_frame, text="Status", padding=8)
        status_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.ws_status_var).grid(row=1, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.backend_status_var).grid(row=2, column=0, sticky="w")

        control_frame = ttk.Frame(top_frame)
        control_frame.grid(row=1, column=0, sticky="nw")
        ttk.Button(control_frame, text="Start", command=self.start_orchestrator).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(control_frame, text="Stop", command=self.stop_orchestrator).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(control_frame, text="Reload models", command=self.reload_models).grid(row=0, column=2, padx=4, pady=4)

        backend_frame = ttk.LabelFrame(top_frame, text="TTS Backend", padding=8)
        backend_frame.grid(row=1, column=1, sticky="ne")
        backend_options = ["neutts", "kani", "mock"]
        ttk.Combobox(backend_frame, values=backend_options, textvariable=self.backend_var, state="readonly").grid(
            row=0, column=0, padx=4, pady=4
        )
        ttk.Button(backend_frame, text="Apply", command=self.apply_backend).grid(row=0, column=1, padx=4, pady=4)

        prompt_frame = ttk.LabelFrame(top_frame, text="Chat", padding=8)
        prompt_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        prompt_frame.columnconfigure(0, weight=1)
        self.prompt_text.grid(in_=prompt_frame, row=0, column=0, sticky="ew")
        ttk.Button(prompt_frame, text="Send", command=self.send_prompt).grid(row=0, column=1, padx=6)
        ttk.Label(prompt_frame, textvariable=self.response_var, wraplength=420).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        override_frame = ttk.LabelFrame(top_frame, text="Overrides", padding=8)
        override_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        ttk.Checkbutton(
            override_frame, text="Emotion", variable=self.emotion_override_enabled
        ).grid(row=0, column=0, padx=4, sticky="w")
        emotion_combo = ttk.Combobox(
            override_frame,
            values=["happy", "sad", "angry", "surprised", "neutral"],
            textvariable=self.emotion_value,
            state="readonly",
        )
        emotion_combo.grid(row=0, column=1, padx=4, sticky="w")

        ttk.Checkbutton(override_frame, text="Rate", variable=self.rate_override).grid(row=1, column=0, padx=4, sticky="w")
        ttk.Scale(
            override_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.rate_scale_value,
        ).grid(row=1, column=1, padx=4, sticky="ew")

        ttk.Checkbutton(override_frame, text="Intensity", variable=self.intensity_override).grid(
            row=2, column=0, padx=4, sticky="w"
        )
        ttk.Scale(
            override_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.intensity_scale_value,
        ).grid(row=2, column=1, padx=4, sticky="ew")

        for column in range(2):
            top_frame.columnconfigure(column, weight=1)
        self.root.columnconfigure(0, weight=1)

    # ------------------------------------------------------------------ Backend helpers
    def start_orchestrator(self) -> None:
        if self.process and self.process.poll() is None:
            messagebox.showinfo("Already running", "The orchestrator is already running.")
            return
        command = [
            sys.executable,
            "-m",
            "server.agent_orchestrator",
            "--config",
            str(self.config_path),
            "--tts-backend",
            self.backend_var.get(),
        ]
        try:
            self.process = subprocess.Popen(command)
            self.status_var.set("Starting...")
            self._status_poll = True
            self.root.after(1000, self.update_status)
        except Exception as exc:  # pragma: no cover - UI feedback path
            messagebox.showerror("Failed to start", str(exc))

    def stop_orchestrator(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.status_var.set("Stopped")
        self.process = None
        self._status_poll = False

    def reload_models(self) -> None:
        try:
            response = requests.post(self._api_url("/reload"), timeout=5)
            response.raise_for_status()
            data = response.json()
            self.response_var.set(f"Reloaded ({data.get('backend')})")
        except Exception as exc:  # pragma: no cover - UI feedback path
            messagebox.showerror("Reload failed", str(exc))

    def apply_backend(self) -> None:
        backend = self.backend_var.get()
        if self.process is None or self.process.poll() is not None:
            self.response_var.set(f"Backend set to {backend} for next launch")
            return
        try:
            response = requests.post(self._api_url("/backend"), json={"backend": backend}, timeout=5)
            response.raise_for_status()
            data = response.json()
            self.backend_status_var.set(f"Backend: {data.get('active', backend)}")
            self.response_var.set(f"Switched backend: {json.dumps(data)}")
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Backend switch failed", str(exc))

    def send_prompt(self) -> None:
        text = self.prompt_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Empty prompt", "Please enter a prompt to send.")
            return
        payload: Dict[str, str] = {"prompt": text}
        if self.emotion_override_enabled.get():
            payload["emotion"] = self.emotion_value.get()
        if self.rate_override.get():
            payload["speaking_rate"] = self._rate_from_value(self.rate_scale_value.get())
        if self.intensity_override.get():
            payload["intensity"] = self._intensity_from_value(self.intensity_scale_value.get())
        try:
            response = requests.post(self._api_url("/ask"), json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            self.response_var.set(f"Response: {json.dumps(data)}")
        except Exception as exc:  # pragma: no cover - UI feedback path
            messagebox.showerror("Prompt failed", str(exc))

    def update_status(self) -> None:
        if not self._status_poll:
            return
        try:
            response = requests.get(self._api_url("/status"), timeout=2)
            response.raise_for_status()
            data = response.json()
            self.status_var.set("Running")
            self.ws_status_var.set(
                f"WS: {data.get('ws_host')}:{data.get('ws_port')} ({data.get('clients', 0)} clients)"
            )
            backend = data.get("tts_backend", self.backend_var.get())
            self.backend_status_var.set(f"Backend: {backend}")
        except Exception:
            self.status_var.set("Starting...")
        finally:
            if self._status_poll:
                self.root.after(1000, self.update_status)

    def _rate_from_value(self, value: float) -> str:
        if value < 33:
            return "slow"
        if value > 66:
            return "fast"
        return "normal"

    def _intensity_from_value(self, value: float) -> str:
        if value < 33:
            return "calm"
        if value > 66:
            return "excited"
        return "normal"

    def _api_url(self, path: str) -> str:
        return f"http://{self.api_host}:{self.api_port}{path}"

    def on_close(self) -> None:
        self.stop_orchestrator()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    panel = ControlPanel()
    panel.run()


if __name__ == "__main__":
    main()
