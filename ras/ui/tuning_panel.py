"""
tuning_panel.py
===============
Real-time parameter tuning panel using Tkinter.

Runs in a separate thread alongside the MuJoCo simulation, providing
sliders for adjusting MPC controller weights, planner parameters, wall
guard settings, and more — all without restarting the simulation.

Supports both MPCController and DifferentialDriveController (PID).

Usage:
    panel = TuningPanel(controller, planner)
    panel.start()   # launches the Tkinter window in a background thread
    # ... simulation runs ...
    panel.stop()    # closes the window
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ras.planning.path_planner import PathPlanner


class TuningPanel:
    """
    A floating Tkinter window with sliders for real-time parameter tuning.

    All slider changes are applied immediately to the live controller
    and planner objects (thread-safe for numeric attribute writes).
    """

    def __init__(self, controller, planner: "PathPlanner"):
        self.controller = controller
        self.planner = planner
        self._thread: threading.Thread | None = None
        self._root: tk.Tk | None = None
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self):
        """Launch the tuning panel in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_tk,
                                        daemon=True, name="TuningPanel")
        self._thread.start()

    def stop(self):
        """Close the panel window."""
        self._running = False
        if self._root is not None:
            try:
                self._root.quit()
            except Exception:
                pass

    # ── Tkinter setup ─────────────────────────────────────────────

    def _run_tk(self):
        """Build and run the Tkinter main loop (called in bg thread)."""
        root = tk.Tk()
        self._root = root
        root.title("🎛  Parameter Tuning Panel")
        root.geometry("380x720")
        root.resizable(False, True)
        root.configure(bg="#1e1e2e")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Title.TLabel",
                        font=("Segoe UI", 11, "bold"),
                        foreground="#cdd6f4", background="#1e1e2e")
        style.configure("Param.TLabel",
                        font=("Segoe UI", 9),
                        foreground="#a6adc8", background="#1e1e2e")
        style.configure("Val.TLabel",
                        font=("Consolas", 9),
                        foreground="#89b4fa", background="#1e1e2e")
        style.configure("TFrame", background="#1e1e2e")
        style.configure("Sep.TFrame", background="#313244")
        style.configure("Horizontal.TScale",
                        background="#1e1e2e", troughcolor="#313244")

        # Scrollable canvas
        canvas = tk.Canvas(root, bg="#1e1e2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(root, orient="vertical",
                                  command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        parent = scroll_frame

        # ── Controller section (auto-detect PID vs MPC) ───────────
        from ras.control.controller import MPCController, DifferentialDriveController

        if isinstance(self.controller, MPCController):
            self._add_section(parent, "MPC Controller")
            self._add_slider(parent, "Q_track", 1.0, 200.0,
                             self.controller.Q_track,
                             lambda v: setattr(self.controller, 'Q_track', v))
            self._add_slider(parent, "R_effort", 0.01, 5.0,
                             self.controller.R_effort,
                             lambda v: setattr(self.controller, 'R_effort', v))
            self._add_slider(parent, "Q_jerk", 0.0, 20.0,
                             self.controller.Q_jerk,
                             lambda v: setattr(self.controller, 'Q_jerk', v))
            self._add_slider(parent, "Horizon N", 3, 20,
                             self.controller.N,
                             lambda v: setattr(self.controller, 'N', int(v)))
            self._add_slider(parent, "Torque limit", 1.0, 15.0,
                             self.controller.torque_limit,
                             lambda v: setattr(self.controller, 'torque_limit', v))
        elif isinstance(self.controller, DifferentialDriveController):
            self._add_section(parent, "PID Controller")
            self._add_slider(parent, "Kp", 0.0, 30.0,
                             self.controller.pid_left.kp,
                             self._set_pid_kp)
            self._add_slider(parent, "Ki", 0.0, 10.0,
                             self.controller.pid_left.ki,
                             self._set_pid_ki)
            self._add_slider(parent, "Kd", 0.0, 10.0,
                             self.controller.pid_left.kd,
                             self._set_pid_kd)

        self._add_separator(parent)

        # ── Speed Profile (SPD) ───────────────────────────────────
        self._add_section(parent, "Speed Profile (SPD)")
        self._add_slider(parent, "v_max (m/s)", 0.1, 3.0,
                         self.planner.spd.v_max,
                         lambda v: setattr(self.planner.spd, 'v_max', v))
        self._add_slider(parent, "a_x_ub (m/s²)", 0.01, 0.5,
                         self.planner.spd.a_x_ub,
                         lambda v: setattr(self.planner.spd, 'a_x_ub', v))
        self._add_slider(parent, "v_min_turn", 0.01, 0.5,
                         self.planner.spd.v_min_turn,
                         lambda v: setattr(self.planner.spd, 'v_min_turn', v))

        self._add_separator(parent)

        # ── Angular Rate (ARGA) ───────────────────────────────────
        self._add_section(parent, "Angular Rate (ARGA)")
        self._add_slider(parent, "K_default", 0.5, 8.0,
                         self.planner.arga.K_default,
                         lambda v: setattr(self.planner.arga, 'K_default', v))
        self._add_slider(parent, "K_min", 0.1, 3.0,
                         self.planner.arga.K_min,
                         lambda v: setattr(self.planner.arga, 'K_min', v))
        self._add_slider(parent, "a_y_max (m/s²)", 0.5, 5.0,
                         self.planner.arga.a_y_max,
                         lambda v: setattr(self.planner.arga, 'a_y_max', v))

        self._add_separator(parent)

        # ── Wall Proximity Guard ──────────────────────────────────
        self._add_section(parent, "Wall Proximity Guard")
        wg = self.planner._wall_guard
        self._add_slider(parent, "danger_dist (m)", 0.5, 5.0,
                         wg.danger_dist,
                         lambda v: setattr(wg, 'danger_dist', v))
        self._add_slider(parent, "critical_dist (m)", 0.2, 3.0,
                         wg.critical_dist,
                         lambda v: setattr(wg, 'critical_dist', v))
        self._add_slider(parent, "repulsion_gain", 0.5, 20.0,
                         wg.repulsion_gain,
                         lambda v: setattr(wg, 'repulsion_gain', v))

        self._add_separator(parent)

        # ── Waypoint / Navigation ─────────────────────────────────
        self._add_section(parent, "Navigation")
        self._add_slider(parent, "capture_radius (m)", 0.3, 5.0,
                         self.planner.wap.capture_radius,
                         lambda v: setattr(self.planner.wap,
                                           'capture_radius', v))

        self._add_separator(parent)

        # ── Stuck Recovery ────────────────────────────────────────
        self._add_section(parent, "Stuck Recovery")
        sr = self.planner._recovery
        self._add_slider(parent, "stuck_timeout (s)", 2.0, 15.0,
                         sr.stuck_timeout,
                         lambda v: setattr(sr, 'stuck_timeout', v))
        self._add_slider(parent, "v_escape (m/s)", 0.1, 1.0,
                         sr.v_escape,
                         lambda v: setattr(sr, 'v_escape', v))
        self._add_slider(parent, "omega_pivot (rad/s)", 0.5, 4.0,
                         sr.omega_pivot,
                         lambda v: setattr(sr, 'omega_pivot', v))

        # Run Tk loop
        try:
            root.mainloop()
        except Exception:
            pass
        self._running = False

    # ── Widget builders ───────────────────────────────────────────

    def _add_section(self, parent: ttk.Frame, title: str):
        """Add a section header label."""
        lbl = ttk.Label(parent, text=f"── {title} ──",
                        style="Title.TLabel")
        lbl.pack(anchor="w", padx=10, pady=(12, 4))

    def _add_separator(self, parent: ttk.Frame):
        """Add a visual separator."""
        sep = ttk.Frame(parent, style="Sep.TFrame", height=1)
        sep.pack(fill="x", padx=10, pady=6)

    def _add_slider(self, parent: ttk.Frame, label: str,
                    lo: float, hi: float, initial: float,
                    callback):
        """Add a labelled slider with live value display."""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=10, pady=2)

        # Label
        lbl = ttk.Label(frame, text=label, style="Param.TLabel")
        lbl.pack(anchor="w")

        # Value display
        val_var = tk.StringVar(value=f"{initial:.3f}")
        val_lbl = ttk.Label(frame, textvariable=val_var,
                            style="Val.TLabel")
        val_lbl.pack(anchor="e")

        # Slider
        var = tk.DoubleVar(value=initial)

        def _on_change(v):
            fv = float(v)
            val_var.set(f"{fv:.3f}")
            callback(fv)

        scale = ttk.Scale(frame, from_=lo, to=hi, variable=var,
                          orient="horizontal", command=_on_change)
        scale.pack(fill="x", pady=(0, 2))

    # ── Parameter setters (PID) ───────────────────────────────────

    def _set_pid_kp(self, v: float):
        self.controller.pid_left.kp = v
        self.controller.pid_right.kp = v

    def _set_pid_ki(self, v: float):
        self.controller.pid_left.ki = v
        self.controller.pid_right.ki = v

    def _set_pid_kd(self, v: float):
        self.controller.pid_left.kd = v
        self.controller.pid_right.kd = v

    def _on_close(self):
        """Handle window close."""
        self._running = False
        if self._root:
            self._root.destroy()
