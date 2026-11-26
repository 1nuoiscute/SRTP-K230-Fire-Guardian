"""Desktop console for K230 sensor and RT-Smart diagnostics."""

import argparse
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from board_control import ACTIONS, RELEASE, PendingCommand, send_command
from board_data import (DEFAULT_SERIAL, Freshness, MonitorError, adb_snapshot,
                        board_info, demo_snapshot, find_adb, guard_info, parse_status)
from board_hardware import parse_hardware, read_chip_temperature
from board_resources import parse_resources, size_text
from rt_serial import SerialReader
from session import ADB_KINDS, Timeline
from version import VERSION


BG = "#edf2f6"
INK = "#14293b"
MUTED = "#4e6477"
BLUE = "#175d85"
TEAL = "#007a77"
AMBER = "#92510c"
NAV = "#112d43"
WHITE = "#ffffff"
GRID = "#d5e1ea"
UI = "Microsoft YaHei UI"
MONO = "Consolas"
METRICS = {
    "温度": ("temperature", "°C", ".1f"),
    "湿度": ("humidity", "%RH", ".1f"),
    "气压": ("pressure", "hPa", ".1f"),
    "MQ-2": ("mq2", "mV", "d"),
    "火焰模块": ("flame", "mV", "d"),
}
VISION = re.compile(r"\b(SINGLE_PROGRESS|PREVIEW_PROGRESS) frames=(\d+) elapsed=(\d+(?:\.\d+)?)\b")


class Monitor:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.live = Timeline("模拟数据" if args.demo else args.serial)
        self.view = None
        self.selected = -1
        self.inbox = queue.Queue()
        self.reader = None
        self.pending = None
        self.hardware_pending = False
        self.freshness = Freshness()
        self.polling = False
        self.next_read = 0
        self.adb_identity = None
        self.adb_issue = None
        self.board_log_tails = {}
        self.last_panel = {}
        self.build()
        if args.replay:
            self.open_recording(args.replay, required=True)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    def build(self):
        r = self.root
        r.title("K230 调试台")
        r.geometry("1180x805")
        r.minsize(980, 710)
        r.configure(bg=BG)

        style = ttk.Style(r)
        style.theme_use("clam")
        style.configure(".", font=(UI, 10))
        style.configure("TButton", padding=(10, 6), foreground=INK)
        style.configure("Accent.TButton", background=TEAL, foreground=WHITE)
        style.map("Accent.TButton", background=[("active", "#086462"), ("disabled", GRID)],
                  foreground=[("disabled", MUTED)])
        style.configure("TNotebook", background=WHITE, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(11, 5))
        style.configure("Treeview", rowheight=27, fieldbackground=WHITE, foreground=INK)
        style.map("Treeview", background=[("selected", BLUE)])

        header = tk.Frame(r, bg=NAV, padx=22, pady=14)
        header.pack(fill="x")
        tk.Label(header, text="K230 调试台", font=(UI, 21, "bold"), fg=WHITE, bg=NAV).pack(side="left")
        self.mode = tk.Label(header, font=(UI, 10), fg="#bdeee8", bg=NAV)
        self.mode.pack(side="left", padx=16)
        tk.Label(header, text=f"v{VERSION}", font=(MONO, 10), fg="#bdd1df", bg=NAV).pack(side="left")
        self.clock = tk.Label(header, font=(MONO, 11), fg=WHITE, bg=NAV)
        self.clock.pack(side="right")

        actions = tk.Frame(header, bg=NAV)
        actions.pack(side="right", padx=(0, 16))
        self.save_button = ttk.Button(actions, text="保存会话", command=self.save)
        self.save_button.pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="打开会话", command=self.open_recording).pack(side="left", padx=(0, 6))
        self.live_button = ttk.Button(actions, text="回到实时", command=self.return_live)
        self.live_button.pack(side="left")

        tools = tk.Frame(r, bg=BG, padx=22, pady=9)
        tools.pack(fill="x")
        self.adb_button = ttk.Button(tools, text="选择 ADB", command=self.choose_adb)
        self.adb_button.pack(side="left")
        self.serial_button = ttk.Button(tools, text="连接视觉串口", command=self.toggle_serial)
        self.serial_button.pack(side="right")
        self.port = ttk.Entry(tools, width=8)
        self.port.insert(0, self.args.port)
        self.port.pack(side="right", padx=6)
        self.label(tools, "视觉串口", bg=BG).pack(side="right")

        self.pages = ttk.Notebook(r, padding=(3, 3, 3, 0))
        self.pages.pack(fill="both", expand=True, padx=19, pady=(0, 2))
        overview_page = tk.Frame(self.pages, bg=BG, padx=8, pady=8)
        debug_page = tk.Frame(self.pages, bg=BG, padx=8, pady=8)
        hardware_page = tk.Frame(self.pages, bg=BG, padx=8, pady=8)
        self.pages.add(overview_page, text="运行总览")
        self.pages.add(debug_page, text="设备调试")
        self.pages.add(hardware_page, text="硬件监测")
        self.pages.select(debug_page)
        overview = tk.Frame(overview_page, bg=BG)
        overview.pack(fill="x")
        self.tiles = {}
        for col, (key, title) in enumerate((
            ("usb", "USB 连接"), ("sensor", "传感器采集"),
            ("inference", "视觉推理"), ("preview", "预览提交"),
        )):
            overview.columnconfigure(col, weight=1, uniform="overview")
            card = tk.Frame(overview, bg=WHITE, padx=12, pady=8,
                            highlightthickness=1, highlightbackground=GRID)
            card.grid(row=0, column=col, padx=(0, 8 if col < 3 else 0), sticky="nsew")
            self.label(card, title, size=10, bold=True).pack(fill="x")
            value = self.label(card, size=16, bold=True)
            value.pack(fill="x", pady=(3, 0))
            detail = self.label(card)
            detail.configure(height=2)
            detail.pack(fill="x")
            card.bind("<Configure>", lambda e, target=detail: target.configure(wraplength=max(e.width - 26, 1)))
            self.tiles[key] = value, detail

        body = tk.Frame(overview_page, bg=BG)
        body.pack(fill="both", expand=True, pady=(10, 0))
        body.columnconfigure(0, weight=3, minsize=490)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        sensor = tk.Frame(body, bg=WHITE, padx=14, pady=10)
        sensor.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        sensor.rowconfigure(4, weight=1)
        self.label(sensor, "传感器读数", size=12, bold=True).grid(row=0, column=0, columnspan=5, sticky="w")
        self.values = {}
        for column, name in enumerate(METRICS):
            sensor.columnconfigure(column, weight=1)
            self.label(sensor, name).grid(row=1, column=column, sticky="w", pady=(9, 1))
            self.values[name] = self.label(sensor, size=13, bold=True)
            self.values[name].grid(row=2, column=column, sticky="w")

        chart_bar = tk.Frame(sensor, bg=WHITE)
        chart_bar.grid(row=3, column=0, columnspan=5, sticky="ew", pady=(14, 2))
        self.label(chart_bar, "近 120 秒趋势", bold=True).pack(side="left")
        self.metric = ttk.Combobox(chart_bar, values=list(METRICS), state="readonly", width=10)
        self.metric.set("MQ-2")
        self.metric.pack(side="right")
        self.metric.bind("<<ComboboxSelected>>", lambda _: self.render_chart())
        self.chart = tk.Canvas(sensor, height=100, background=WHITE, highlightthickness=0)
        self.chart.grid(row=4, column=0, columnspan=5, sticky="nsew")
        self.chart.bind("<Configure>", lambda _: self.render_chart())
        self.label(sensor, "— 表示无有效读数。MQ-2 与火焰模块显示传感器电压。").grid(
            row=5, column=0, columnspan=5, sticky="w", pady=(4, 0))

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        debug_page.columnconfigure(0, weight=1, minsize=335)
        debug_page.columnconfigure(1, weight=2)
        debug_page.rowconfigure(0, weight=1)
        operations = tk.Frame(debug_page, bg=BG, width=360)
        operations.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        controls = tk.Frame(operations, bg=WHITE, padx=12, pady=8)
        controls.pack(fill="x", pady=(0, 8))
        controls.columnconfigure(0, weight=1)
        self.label(controls, "板端控制", size=12, bold=True).grid(row=0, column=0, columnspan=2, sticky="w")
        self.control_buttons = {}
        self.control_labels = {}
        for row, (code, note) in enumerate((
            ("k", "启动或停止摄像头与识别程序"),
            ("1", "显示或隐藏摄像头页上的传感器参数"),
            ("2", "在摄像头页与全屏传感器数据页之间切换"),
        ), 1):
            group = tk.Frame(controls, bg=WHITE)
            group.grid(row=row, column=0, sticky="w", pady=(8, 0))
            current = self.label(group, size=10, bold=True)
            current.pack(anchor="w")
            self.label(group, note, wrap=196).pack(anchor="w")
            button = ttk.Button(controls, width=16, command=lambda action=code: self.issue_control(action))
            button.grid(row=row, column=1, sticky="e", padx=(4, 0), pady=(8, 0))
            self.control_buttons[code] = button
            self.control_labels[code] = current
        self.control_state = self.label(controls)
        self.control_state.configure(height=2, wraplength=310)
        self.control_state.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        valve = tk.Frame(right, bg=WHITE, padx=12, pady=8)
        valve.pack(fill="x", pady=(0, 8))
        self.label(valve, "阀门联动", size=11, bold=True).pack(anchor="w")
        self.guard = self.label(valve, size=10, bold=True)
        self.guard.pack(anchor="w", pady=(3, 0))
        self.label(valve, "此处记录程序指令，物理阀位需现场确认。").pack(anchor="w")

        self.overview_identity = self.label(right, bg=BG, wrap=350)
        self.overview_identity.pack(fill="x", pady=(2, 0))
        events = tk.Frame(operations, bg=WHITE, padx=12, pady=9)
        events.pack(fill="both", expand=True)
        event_header = tk.Frame(events, bg=WHITE)
        event_header.pack(fill="x", pady=(0, 6))
        self.label(event_header, "故障记录", size=12, bold=True).pack(side="left")
        self.mark_button = ttk.Button(event_header, text="标记故障", style="Accent.TButton", command=self.mark)
        self.mark_button.pack(side="right")

        note_row = tk.Frame(events, bg=WHITE)
        note_row.pack(fill="x", pady=(0, 7))
        self.label(note_row, "现场现象").pack(side="left", padx=(0, 8))
        self.note = ttk.Entry(note_row)
        self.note.pack(side="left", fill="x", expand=True)
        self.note.bind("<Return>", lambda _: self.mark())
        detail = tk.Frame(events, bg=WHITE)
        detail.pack(side="bottom", fill="x", pady=(6, 0))
        self.event_detail = tk.Text(detail, height=2, width=1, wrap="word", bg="#f6f9fc",
                                    fg=INK, font=(UI, 9), relief="flat", padx=6, pady=3,
                                    state="disabled")
        detail_scroll = ttk.Scrollbar(detail, orient="vertical", command=self.event_detail.yview)
        self.event_detail.configure(yscrollcommand=detail_scroll.set)
        detail_scroll.pack(side="right", fill="y")
        self.event_detail.pack(side="left", fill="x", expand=True)
        self.events = ttk.Treeview(events, columns=("time", "event"), show="headings", height=4)
        self.events.heading("time", text="接收时间")
        self.events.heading("event", text="事件（双击回看）")
        self.events.column("time", width=86, stretch=False)
        self.events.column("event", width=190)
        event_scroll = ttk.Scrollbar(events, orient="vertical", command=self.events.yview)
        self.events.configure(yscrollcommand=event_scroll.set)
        event_scroll.pack(side="right", fill="y")
        self.events.pack(fill="both", expand=True)
        self.events.bind("<Double-1>", self.jump_to_event)
        self.events.bind("<Return>", self.jump_to_event)
        self.events.bind("<<TreeviewSelect>>", self.show_event_detail)

        self.log_panel = tk.Frame(debug_page, bg=WHITE, padx=12, pady=8)
        self.log_panel.grid(row=0, column=1, sticky="nsew")
        log_header = tk.Frame(self.log_panel, bg=WHITE)
        log_header.pack(fill="x", pady=(0, 7))
        self.label(log_header, "调试输出", size=12, bold=True).pack(side="left")
        self.serial_summary = self.label(log_header)
        self.serial_summary.pack(side="left", padx=9)
        ttk.Button(log_header, text="复制日志", command=self.copy_log).pack(side="right")
        self.follow_log = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_header, text="跟随最新", variable=self.follow_log,
                        command=self.follow_output).pack(side="right", padx=6)
        self.log = {}
        self.log_tabs = ttk.Notebook(self.log_panel)
        self.log_tabs.pack(fill="both", expand=True)
        for name in ("实时输出", "阀门日志", "传感器日志", "连接详情"):
            frame = tk.Frame(self.log_tabs, bg=WHITE)
            self.log_tabs.add(frame, text=name)
            field = tk.Text(frame, font=(MONO, 10), bg="#f6f9fc", fg=INK, relief="flat",
                            wrap="word", padx=9, pady=7, state="disabled", height=6, width=12)
            bar = ttk.Scrollbar(frame, orient="vertical", command=field.yview)
            field.configure(yscrollcommand=bar.set)
            field.tag_configure("control", foreground=TEAL)
            field.tag_configure("rt_state", foreground=BLUE)
            field.tag_configure("adb_state", foreground=BLUE)
            bar.pack(side="right", fill="y")
            field.pack(side="left", fill="both", expand=True)
            self.log[name] = field

        self.build_hardware(hardware_page)
        footer = tk.Frame(r, bg=BG, padx=22, pady=8)
        footer.pack(fill="x")
        self.label(footer, "会话时间轴", bg=BG).pack(side="left")
        self.seek = ttk.Scale(footer, from_=0, to=1, command=self.scrub)
        self.seek.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self.updating_seek = False

    def copy_log(self):
        widget = self.log[self.log_tabs.tab(self.log_tabs.select(), "text")]
        selection = widget.tag_ranges("sel")
        text = widget.get(*selection) if selection else widget.get("1.0", "end-1c")
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

    def follow_output(self):
        if self.follow_log.get():
            for widget in self.log.values():
                widget.see("end")

    def build_hardware(self, page):
        page.columnconfigure(0, weight=1, uniform="hardware")
        page.columnconfigure(1, weight=1, uniform="hardware")
        page.rowconfigure(0, weight=1)
        left = tk.Frame(page, bg=WHITE, padx=14, pady=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = tk.Frame(page, bg=WHITE, padx=14, pady=10)
        right.grid(row=0, column=1, sticky="nsew")
        self.label(left, "Linux 小核内存", size=12, bold=True).pack(anchor="w")
        self.memory_value = self.label(left, size=16, bold=True)
        self.memory_value.pack(anchor="w", pady=(3, 0))
        self.memory_usage = ttk.Progressbar(left, orient="horizontal", mode="determinate")
        self.memory_usage.pack(fill="x", pady=6)
        self.memory_detail = self.label(left)
        self.memory_detail.pack(anchor="w")
        self.label(left, "已挂载文件系统", size=11, bold=True).pack(anchor="w", pady=(15, 5))
        self.filesystems = self.hardware_table(left, (
            ("目标", 65), ("来源", 116), ("总量", 80), ("可用", 80), ("占用率", 63)), 3)
        self.label(left, "tmpfs / RAM 来源表示以内存提供的空间。", wrap=380).pack(anchor="w", pady=(4, 0))
        self.label(left, "识别到的存储设备", size=11, bold=True).pack(anchor="w", pady=(12, 5))
        self.disks = self.hardware_table(left, (("设备", 105), ("介质", 85), ("容量", 100)), 3)

        header = tk.Frame(right, bg=WHITE)
        header.pack(fill="x")
        self.label(header, "K230 芯片结温", size=12, bold=True).pack(side="left")
        self.hardware_button = ttk.Button(header, text="读取芯片结温", command=self.request_hardware)
        self.hardware_button.pack(side="right")
        self.chip_temperature = self.label(right, size=16, bold=True)
        self.chip_temperature.pack(anchor="w", pady=(2, 0))
        self.chip_detail = self.label(right, wrap=400)
        self.chip_detail.pack(anchor="w", pady=(0, 8))
    @staticmethod
    def hardware_table(parent, columns, height):
        table = ttk.Treeview(parent, columns=[name for name, _ in columns],
                             show="headings", height=height, selectmode="browse")
        for name, width in columns:
            table.heading(name, text=name)
            table.column(name, width=width, minwidth=45)
        table.pack(fill="x")
        return table

    def table_rows(self, table, rows):
        key = str(table)
        signature = tuple(rows)
        if self.last_panel.get(key) == signature:
            return
        self.last_panel[key] = signature
        selected = table.selection()
        table.delete(*table.get_children())
        for item, values in rows:
            table.insert("", "end", iid=item, values=values)
        if selected and table.exists(selected[0]):
            table.selection_set(selected[0])

    def request_hardware(self):
        if self.hardware_pending or self.hardware_button.instate(["disabled"]):
            return
        self.hardware_pending = True
        self.render()

        def collect():
            try:
                raw = read_chip_temperature(self.args.adb, self.args.serial)
                data = {"text": raw}
            except MonitorError as exc:
                data = {"text": str(exc), "error": True}
            self.inbox.put(("hardware", data))

        threading.Thread(target=collect, daemon=True).start()

    def render_hardware(self, timeline, index, raw):
        live = self.view is None and not self.args.demo
        self.hardware_button.state(["!disabled"] if live and raw and not self.hardware_pending else ["disabled"])
        self.hardware_button.configure(text="正在读取…" if self.hardware_pending else "读取芯片结温")
        resource = parse_resources(raw.get("resources", ""))
        total, available = resource.total_kib, resource.available_kib
        used = total - available if total is not None and available is not None else None
        self.memory_value.configure(text=f"{size_text(available)} 可用")
        self.memory_detail.configure(text=(f"总量 {size_text(total)}  ·  估算占用 {size_text(used)}"
                                           if total is not None else "等待板端内存信息"))
        self.memory_usage["value"] = (used / total * 100) if used is not None and total else 0
        self.table_rows(self.filesystems, [
            (path, (path, "tmpfs / RAM" if source == "tmpfs" else source,
                    size_text(capacity), size_text(free), pct))
            for path, source, capacity, _, free, pct in resource.filesystems
        ])
        self.table_rows(self.disks, [(name, (name, kind, size_text(capacity)))
                                     for name, kind, capacity in resource.disks])

        reading = timeline.last({"hardware"}, index)
        chip_value, chip_detail = "—", "点「读取芯片结温」采样。环境温度在运行总览中显示。"
        if reading >= 0:
            event = timeline.events[reading]
            stamp = local_clock(event)
            data = event["data"]
            if data.get("error"):
                chip_detail = f"{stamp} 读取失败 · {data['text']}"
            else:
                snap = parse_hardware(data["text"])
                boot = board_info(raw)[0] if raw else ""
                if snap.temperature is not None and snap.boot == boot:
                    chip_value = f"{snap.temperature:.1f} °C"
                    chip_detail = f"{stamp} 采样 · {snap.thermal_source}"
                elif snap.boot and snap.boot != boot:
                    chip_detail = f"{stamp} 的快照已过时，请重新读取。"
                else:
                    chip_detail = f"{stamp} 未取得有效的片上传感器读数。"
        self.chip_temperature.configure(text=chip_value)
        self.chip_detail.configure(text=chip_detail[:180])

    @staticmethod
    def label(parent, text="", size=9, bold=False, bg=WHITE, wrap=0):
        return tk.Label(parent, text=text, font=(UI, size, "bold" if bold else "normal"),
                        fg=INK if bold else MUTED, bg=bg, anchor="w", justify="left",
                        wraplength=wrap)

    def get_sample(self):
        try:
            raw = demo_snapshot() if self.args.demo else adb_snapshot(self.args.adb, self.args.serial)
            self.inbox.put(("sample", raw))
        except MonitorError as exc:
            self.inbox.put(("adb_error", str(exc)))

    def tick(self):
        now = time.monotonic()
        changed = False
        for _ in range(200):
            try:
                kind, data = self.inbox.get_nowait()
            except queue.Empty:
                break
            if kind == "hardware":
                self.hardware_pending = False
                self.live.append(kind, data)
                changed = True
                continue
            if kind == "control_result":
                sequence, error = data
                if error:
                    self.finish_control(f"未能确认送达。{error}")
                elif self.pending is not None:
                    self.pending.sequence = sequence
                    self.live.append("control", {"text": "命令已写入，等待板端回执"})
                changed = True
                continue
            if kind in ADB_KINDS:
                self.polling = False
                self.next_read = now + 2
            self.accept(kind, data)
            changed = True
        if not self.args.replay and not self.polling and now >= self.next_read:
            self.polling = True
            threading.Thread(target=self.get_sample, daemon=True).start()
        if self.reader is not None and self.reader.process.poll() is not None:
            self.reader = None
            self.serial_button.configure(text="连接视觉串口")
            changed = True
        if self.check_control(now):
            changed = True
        if changed or not self.last_panel:
            self.render()
        self.timer = self.root.after(125, self.tick)

    def accept(self, kind, data):
        if kind == "sample":
            try:
                status = parse_status(data["status"])
                state = self.freshness.observe(board_info(data)[0], status.sequence, time.monotonic())
                message = {"fresh": "序号 %s" % status.sequence, "waiting": "等待序号更新",
                           "stale": "状态文件停止更新"}[state]
            except ValueError as exc:
                self.freshness.reset()
                state, message = "invalid", str(exc)
            self.live.append(kind, {"raw": data, "state": state, "message": message})
            identity = board_info(data)
            if self.adb_identity != identity:
                action = "ADB 已读取" if self.adb_identity is None else "启动号或版本变化"
                self.live.append("adb_state", {"text": f"{action}  {identity[0][:8] or '—'}"})
                self.adb_identity = identity
            self.accept_board_logs(identity[0], data)
            self.adb_issue = None
            if self.args.demo:
                seconds = time.monotonic() - self.live.start
                self.live.append("rt_line", {"text":
                    f"SINGLE_PROGRESS frames={int(seconds * 6)} elapsed={seconds:.3f} mode=camera"})
                self.live.append("rt_line", {"text":
                    f"PREVIEW_PROGRESS frames={int(seconds * 30)} elapsed={seconds:.3f}"})
        else:
            if kind == "adb_error":
                self.freshness.reset()
                self.adb_identity = None
                if data != self.adb_issue:
                    self.live.append("adb_state", {"text": f"ADB 读取失败  {data}"})
                self.adb_issue = data
            self.live.append(kind, {"text": data})

    def accept_board_logs(self, boot, snapshot):
        for key, title in (("guard_log", "守护"), ("launcher_log", "启动器")):
            lines = snapshot[key].splitlines()
            old_boot, previous = self.board_log_tails.get(key, (None, []))
            if old_boot != boot:
                previous = []
            if lines == previous:
                continue
            overlap = next((n for n in range(min(len(lines), len(previous)), 0, -1)
                            if previous[-n:] == lines[:n]), 0)
            for line in lines[overlap:]:
                self.live.append("board_log", {"text": f"{title}  {line}"})
            self.board_log_tails[key] = (boot, lines)

    def context(self):
        timeline = self.view if self.view is not None else self.live
        index = self.selected if self.view is not None else len(timeline.events) - 1
        return timeline, index

    def render(self):
        timeline, index = self.context()
        live = self.view is None
        replay = not live and timeline is not self.live
        self.mode.configure(text=("模拟演示" if self.args.demo else "实时监测") if live
                            else ("离线回放" if replay else "回看实时会话"))
        self.mark_button.state(["!disabled"] if live else ["disabled"])
        self.note.state(["!disabled"] if live else ["disabled"])
        self.save_button.state(["!disabled"] if not replay and self.live.events else ["disabled"])
        self.live_button.state(["!disabled"] if not live and not self.args.replay else ["disabled"])
        available = live and not self.args.demo and not self.args.replay
        self.serial_button.state(["!disabled"] if available else ["disabled"])
        allow_adb = available and self.pending is None and not self.hardware_pending
        self.adb_button.state(["!disabled"] if allow_adb else ["disabled"])
        self.port.state(["!disabled"] if available and self.reader is None else ["disabled"])
        self.updating_seek = True
        self.seek.configure(to=max(timeline.times[-1], 0.001) if timeline.times else 1)
        self.seek.set(timeline.times[index] if index >= 0 else 0)
        self.seek.state(["!disabled"] if timeline.events else ["disabled"])
        self.updating_seek = False
        at = local_clock(timeline.events[index]) if index >= 0 else "--:--:--"
        self.clock.configure(text=at)

        adb_index = timeline.last(ADB_KINDS, index)
        sample = timeline.events[adb_index] if adb_index >= 0 else None
        status, raw, message, state = self.render_sample(sample)
        self.render_connection(sample, timeline.source, replay, raw)
        for name, (attr, unit, fmt) in METRICS.items():
            value = getattr(status, attr, None)
            self.values[name].configure(text=(f"{value:{fmt}} {unit}" if value is not None else "—"),
                                        fg=INK if value is not None else MUTED)
        self.render_sensor(state, status)
        active_vision = self.render_board(timeline, index, raw)
        self.render_controls(timeline, index, raw, active_vision)
        self.render_output(timeline, index)
        self.render_hardware(timeline, index, raw)
        self.set_log("阀门日志", "\n\n".join(filter(None, [raw.get("guard_log"), raw.get("launcher_log")])))
        self.set_log("传感器日志", raw.get("sensor_log", ""))
        snapshot_text = "\n\n".join(
            f"[{key}]\n{raw.get(key) or '—'}" for key in ("board", "status", "guard", "processes", "resources"))
        self.set_log("连接详情", f"{timeline.source}\n{message}\n\n{snapshot_text}" if raw else message)
        self.render_events(timeline)
        self.render_chart()

    def set_tile(self, key, headline, detail, color=MUTED):
        value, explanation = self.tiles[key]
        value.configure(text=headline, fg=color)
        explanation.configure(text=detail)

    def render_sample(self, sample):
        if sample is None:
            return None, {}, "等待采样", "waiting"
        data = sample["data"]
        if sample["kind"] == "adb_error":
            return None, {}, data["text"], "invalid"
        state, raw = data["state"], data["raw"]
        if state == "fresh":
            try:
                return parse_status(raw["status"]), raw, data["message"], state
            except ValueError as exc:
                return None, raw, str(exc), "invalid"
        return None, raw, data["message"], state

    def render_connection(self, sample, source, replay, raw):
        if source == "模拟数据":
            self.set_tile("usb", "模拟会话", "本窗口使用模拟数据。", BLUE)
        elif sample is None:
            self.set_tile("usb", "等待连接", "通过 ADB 读取板端状态。")
        elif not raw:
            self.set_tile("usb", "读取失败", "检查 USB 连接或查看连接详情。", AMBER)
        else:
            boot, _ = board_info(raw)
            action = "历史快照" if replay else "已连接"
            self.set_tile("usb", action, f"板端状态可读 · 启动号 {boot[:8] or '—'}", TEAL)

    def render_sensor(self, state, status):
        if state == "fresh":
            count = sum(getattr(status, attr) is not None for attr, _, _ in METRICS.values())
            if count:
                self.set_tile("sensor", f"{count} 项有效", "最近一次板端采样含有效读数。", TEAL)
            else:
                self.set_tile("sensor", "无有效读数", "板端状态在更新，各传感器均未上报有效值。", AMBER)
            return
        headline, detail = {
            "waiting": ("等待采样", "收到连续更新后显示读数。"),
            "stale": ("数据过期", "约 6 秒未收到新的状态序号。"),
            "invalid": ("不可读取", "板端状态缺失或格式异常。"),
        }[state]
        self.set_tile("sensor", headline, detail, AMBER if state != "waiting" else MUTED)

    def render_board(self, timeline, index, raw):
        decision, reason = guard_info(raw) if raw else ("", "")
        valve = {"开阀": "开阀指令", "关阀": "关阀指令"}.get(decision, "暂无指令记录")
        if reason.startswith("no-data"):
            valve += " · 传感器无有效读数"
        self.guard.configure(text=valve)
        boot, release = board_info(raw) if raw else ("", "")
        self.overview_identity.configure(text=f"启动号  {boot or '—'}\n演示版本  {release or '—'}")
        serial_state = timeline.last({"rt_state"}, index)
        connected = timeline.source == "模拟数据" or (
            serial_state >= 0 and timeline.events[serial_state]["data"]["text"].endswith("正在接收"))
        if not connected:
            self.serial_summary.configure(text="视觉串口未连接")
            self.set_tile("inference", "串口未连接", "连接视觉串口以读取推理进度。")
            self.set_tile("preview", "串口未连接", "通过串口日志检查送显链路。")
            return False

        active_lines = [timeline.events[i] for i in timeline.recent("rt_line", index, 160)
                        if i > serial_state]
        if timeline.source == "模拟数据":
            summary = "模拟串口"
        else:
            summary = "正在接收视觉日志" if active_lines else "串口已连接 · 暂无视觉日志"
        self.serial_summary.configure(text=summary)
        end = timeline.times[index] if index >= 0 else 0
        running = False
        for key, tag, limit, stop in (
            ("inference", "SINGLE_PROGRESS", 6, "SINGLE_END"),
            ("preview", "PREVIEW_PROGRESS", 12, "PREVIEW_END"),
        ):
            title, detail, color = progress_state(active_lines, tag, limit, stop, end)
            self.set_tile(key, title, detail, color)
            if key == "inference":
                running = color == TEAL
        return running

    def controls_available(self, raw):
        if self.args.demo or self.view is not None or not raw or self.reader is None:
            return False
        state = self.live.last({"rt_state"}, len(self.live.events) - 1)
        boot, release = board_info(raw)
        return bool(boot and release == RELEASE and state >= 0
                    and self.live.events[state]["data"]["text"].endswith("正在接收")
                    and self.reader.process.poll() is None)

    def render_controls(self, timeline, index, raw, active_vision):
        available = self.controls_available(raw) and self.pending is None
        vision = timeline.device_state("vision", index)
        panel = timeline.device_state("panel", index)
        page = timeline.device_state("page", index)
        vision_label = "运行中" if active_vision else {
            "running": "进度未更新", "stopped": "已停止", "starting": "启动中",
            "stopping": "退出中", "fault": "异常退出",
        }.get(vision, "状态未知")
        states = {
            "k": ("视觉程序", vision_label,
                  "停止视觉" if active_vision else "启动视觉" if vision == "stopped" else "切换视觉启停"),
            "1": ("参数面板", {"shown": "已显示", "hidden": "已隐藏"}.get(panel, "状态未知"),
                  {"shown": "隐藏参数面板", "hidden": "显示参数面板"}.get(panel, "显隐参数面板")),
            "2": ("板载 LCD", {"camera": "摄像头页", "data": "传感器数据页"}.get(page, "页面未知"),
                  {"camera": "打开数据页", "data": "返回摄像头页"}.get(page, "切换 LCD 页面")),
        }
        for code, button in self.control_buttons.items():
            title, state, action = states[code]
            enabled = available and (
                active_vision if code != "k" else vision not in ("starting", "stopping", "fault"))
            self.control_labels[code].configure(text=f"{title} · {state}")
            button.configure(text=action)
            button.state(["!disabled"] if enabled else ["disabled"])
        latest = timeline.last({"control"}, index)
        if latest >= 0:
            text = timeline.events[latest]["data"]["text"]
        elif self.args.demo or self.view is not None:
            text = "设备控制仅在实板联机时可用。"
        elif available:
            text = "每次操作的发送与回执会出现在实时输出中。"
        else:
            text = "连接 USB 和视觉串口后可调试。"
        self.control_state.configure(text=text[:200])

    def issue_control(self, action):
        if self.control_buttons[action].instate(["disabled"]):
            return
        last = self.live.last(ADB_KINDS, len(self.live.events) - 1)
        raw = self.live.events[last]["data"].get("raw") if last >= 0 else None
        if not self.controls_available(raw) or self.pending is not None:
            return
        boot = board_info(raw)[0]
        self.pending = PendingCommand(action, len(self.live.events), time.monotonic() + 18)
        self.live.append("control", {"text": f"正在发送{ACTIONS[action]}命令"})
        self.render()

        def send():
            try:
                sequence = send_command(self.args.adb, self.args.serial, boot, action)
                self.inbox.put(("control_result", (sequence, None)))
            except MonitorError as exc:
                self.inbox.put(("control_result", (None, str(exc))))

        threading.Thread(target=send, daemon=True).start()

    def finish_control(self, result):
        if self.pending is not None:
            title = ACTIONS[self.pending.action]
            self.live.append("control", {"text": f"{title} · {result}"})
            self.pending = None

    def check_control(self, now):
        pending = self.pending
        if pending is None:
            return False
        if now >= pending.deadline:
            result = "板端已接收，未确认动作结果" if pending.received else "未收到板端回执"
            self.finish_control(result)
            return True
        if pending.sequence is None:
            return False
        indices = self.live.indices["rt_line"]
        start = pending.next_index if pending.next_index is not None else pending.start_index
        changed = False
        for i in indices[bisect_left(indices, start):]:
            received = pending.received
            result = pending.observe(self.live.events[i]["data"]["text"])
            if result:
                self.finish_control(result)
                return True
            if pending.received and not received:
                self.live.append("control", {"text": "板端已接收，等待动作结果"})
                changed = True
        pending.next_index = len(self.live.events)
        return changed

    def render_output(self, timeline, index):
        tags = {"rt_line": "RT", "rt_state": "串口", "adb_state": "USB",
                "control": "操作", "board_log": "板端"}
        rows = sorted(i for tag in tags for i in timeline.recent(tag, index, 300))[-300:]
        key = (id(timeline), tuple(rows))
        previous = self.last_panel.get("output")
        if key == previous:
            return
        self.last_panel["output"] = key
        field = self.log["实时输出"]
        first, bottom = field.yview()
        field.configure(state="normal")
        keep = 0
        if previous and previous[0] == key[0] and previous[1] and rows:
            old = previous[1]
            retired = bisect_left(old, rows[0])
            retained = old[retired:]
            if retained and tuple(rows[:len(retained)]) == retained:
                if retired:
                    field.delete("1.0", f"{retired + 1}.0")
                keep = len(retained)
        if keep == 0:
            field.delete("1.0", "end")
        for i in rows[keep:]:
            event = timeline.events[i]
            kind = event["kind"]
            message = " ".join(event["data"]["text"].splitlines())
            field.insert("end", f"{local_clock(event, millis=True)}  [{tags[kind]}]  {message}\n", kind)
        if not rows:
            field.insert("end", "等待 USB 或视觉串口输出。")
        field.configure(state="disabled")
        if self.follow_log.get() and bottom > 0.99:
            field.see("end")
        elif keep == 0:
            field.yview_moveto(first)

    def set_log(self, name, text):
        if name in self.last_panel and self.last_panel[name] == text:
            return
        self.last_panel[name] = text
        field = self.log[name]
        previous_fraction, bottom = field.yview()
        field.configure(state="normal")
        field.delete("1.0", "end")
        field.insert("1.0", text or "暂无记录")
        field.configure(state="disabled")
        field.yview_moveto(1 if self.follow_log.get() and bottom > 0.99 else previous_fraction)

    def render_events(self, timeline):
        entries = []
        indices = sorted((i for kind in ("adb_state", "rt_state", "mark", "control")
                          for i in timeline.indices[kind][-100:]), reverse=True)
        for i in indices[:100]:
            event = timeline.events[i]
            text = event["data"]["text"].replace("\n", " ")
            if event["kind"] == "mark":
                text = f"标记  {text}"
            entries.append((str(i), local_clock(event), text))
        signature = (id(timeline), tuple(entries))
        if self.last_panel.get("events") == signature:
            return
        self.last_panel["events"] = signature
        selected = self.events.selection()
        self.events.delete(*self.events.get_children())
        for identifier, at, text in entries:
            self.events.insert("", "end", iid=identifier, values=(at, text))
        if selected and self.events.exists(selected[0]):
            self.events.selection_set(selected[0])
        self.show_event_detail()

    def show_event_detail(self, _=None):
        selected = self.events.selection()
        timeline, _ = self.context()
        text = timeline.events[int(selected[0])]["data"]["text"] if selected else ""
        self.event_detail.configure(state="normal")
        self.event_detail.delete("1.0", "end")
        self.event_detail.insert("1.0", text)
        self.event_detail.configure(state="disabled")

    def render_chart(self):
        if not hasattr(self, "chart"):
            return
        canvas = self.chart
        timeline, index = self.context()
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 110 or height < 40:
            return
        left, top, right, bottom = 54, 10, width - 8, height - 19
        attr, unit, _ = METRICS[self.metric.get()]
        end = timeline.times[index] if index >= 0 else 0
        points = []
        previous_boot = None
        for i in timeline.window(ADB_KINDS, index, 120):
            event = timeline.events[i]
            data = event["data"]
            value = None
            if event["kind"] == "sample" and data["state"] == "fresh":
                boot, _ = board_info(data["raw"])
                if previous_boot != boot and points:
                    points.append((event["t"], None))
                previous_boot = boot
                try:
                    value = getattr(parse_status(data["raw"]["status"]), attr)
                except ValueError:
                    pass
            points.append((event["t"], value))
        values = [value for _, value in points if value is not None]
        if not values:
            canvas.create_text((left + right) / 2, (top + bottom) / 2, text="暂无有效趋势数据",
                               font=(UI, 9), fill=MUTED)
        else:
            lo, hi = min(values), max(values)
            pad = max(1 if unit == "mV" else 0.2, (hi - lo) * 0.12)
            lo, hi = lo - pad, hi + pad
            for y, label in [(top, hi), (bottom, lo)]:
                canvas.create_text(left - 5, y, text=f"{label:.5g}", anchor="e", fill=MUTED, font=(MONO, 8))
                canvas.create_line(left, y, right, y, fill=GRID)
            last = None
            for stamp, value in points:
                if value is None:
                    last = None
                    continue
                x = right - (end - stamp) / 120 * (right - left)
                y = bottom - (value - lo) / (hi - lo) * (bottom - top)
                if last is not None and stamp - last[0] < 6:
                    canvas.create_line(last[1], last[2], x, y, fill=TEAL, width=2)
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=TEAL, outline="")
                last = stamp, x, y
        canvas.create_text(left, height - 5, text="−120 s", anchor="sw", fill=MUTED, font=(MONO, 8))
        canvas.create_text(right, height - 5, text="0 s", anchor="se", fill=MUTED, font=(MONO, 8))

    def scrub(self, value):
        if self.updating_seek:
            return
        timeline, _ = self.context()
        if timeline.events:
            self.select_event(timeline, max(0, bisect_right(timeline.times, float(value)) - 1))

    def select_event(self, timeline, index):
        self.view = timeline
        self.selected = index
        self.render()

    def jump_to_event(self, _):
        selected = self.events.selection()
        if selected:
            self.select_event(self.context()[0], int(selected[0]))

    def return_live(self):
        self.view = None
        self.render()

    def choose_adb(self):
        path = filedialog.askopenfilename(parent=self.root, title="选择 Android Platform-Tools 中的 adb.exe",
                                           filetypes=[("ADB", "adb.exe")])
        if path:
            self.args.adb = path
            self.freshness.reset()
            self.adb_identity = None
            self.next_read = 0

    def toggle_serial(self):
        if self.reader is not None and self.reader.process.poll() is None:
            self.reader.close()
            self.serial_button.configure(text="正在关闭")
        else:
            try:
                self.reader = SerialReader(self.port.get().strip().upper(),
                                           lambda kind, data: self.inbox.put((kind, data)))
            except (OSError, ValueError) as exc:
                self.reader = None
                self.accept("rt_state", str(exc))
            if self.reader is not None:
                self.serial_button.configure(text="断开视觉串口")
        self.render()

    def mark(self):
        if self.view is None:
            self.live.append("mark", {"text": self.note.get().strip()[:500] or "故障标记"})
            self.note.delete(0, "end")
            self.render()

    def save(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".jsonl",
                                            filetypes=[("K230 会话", "*.jsonl")],
                                            initialfile=f"k230-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
        if path:
            try:
                self.live.save(path)
                messagebox.showinfo("会话已保存", Path(path).name, parent=self.root)
            except OSError as exc:
                messagebox.showerror("保存失败", str(exc), parent=self.root)

    def open_recording(self, path=None, required=False):
        path = path or filedialog.askopenfilename(parent=self.root,
                                                   filetypes=[("K230 会话", "*.jsonl")])
        if path:
            try:
                self.view = Timeline.open(path)
            except (ValueError, OSError) as exc:
                if required:
                    raise SystemExit(f"无法打开会话（{exc}）") from exc
                messagebox.showerror("打开失败", str(exc), parent=self.root)
                return
            self.selected = len(self.view.events) - 1
            self.render()

    def close(self):
        self.root.after_cancel(self.timer)
        if self.reader is not None:
            self.reader.close()
        self.root.destroy()


def progress_state(lines, tag, age_limit, stop, now):
    reports = []
    for event in reversed(lines):
        text = event["data"]["text"]
        if stop in text:
            if not reports:
                return "已停止", "板端日志已报告程序或送显结束。", MUTED
            break
        match = VISION.search(text)
        if match and match[1] == tag:
            reports.append((int(match[2]), float(match[3]), event["t"]))
            if len(reports) == 2:
                break
    if not reports:
        return "等待进度", "尚未收到对应的板端进度日志。", MUTED
    count, elapsed, stamp = reports[0]
    evidence = f"累计 {count:,} 帧"
    if now - stamp > age_limit:
        return "近期无更新", f"{evidence} · 等待新的进度日志。", AMBER
    if len(reports) == 2:
        previous, previous_time, _ = reports[1]
        span = elapsed - previous_time
        if count > previous and span >= 0.001:
            return f"{(count - previous) / span:.1f} 帧/秒", evidence + " · 由相邻上报计算。", TEAL
    return "已收到进度", evidence + " · 等待下一次上报。", TEAL


def local_clock(event, millis=False):
    instant = datetime.fromisoformat(event["at"])
    return instant.isoformat(timespec="milliseconds" if millis else "seconds").split("T")[1][:12 if millis else 8]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Launch with simulated board data")
    mode.add_argument("--replay", type=Path, help="Open a saved session offline")
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--adb", default=find_adb(), help="Path to the ADB executable")
    parser.add_argument("--serial", default=DEFAULT_SERIAL, help="ADB device serial")
    parser.add_argument("--port", default="COM6", help="Initial RT-Smart UART port")
    parser.add_argument("--listen", action="store_true", help="Connect RT-Smart logs at startup")
    args = parser.parse_args()
    if args.listen and (args.demo or args.replay):
        parser.error("--listen requires live USB mode")
    root = tk.Tk()
    app = Monitor(root, args)
    if args.listen:
        app.toggle_serial()
    root.mainloop()


if __name__ == "__main__":
    main()
