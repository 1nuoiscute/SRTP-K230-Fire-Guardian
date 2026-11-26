"""In-memory timeline and portable JSON Lines recordings."""

from bisect import bisect_left, bisect_right
from datetime import datetime
import json
import math
import re
from pathlib import Path
import tempfile
import time

from board_data import SECTIONS


KINDS = {"sample", "adb_error", "adb_state", "rt_line", "rt_state", "mark", "control", "hardware", "voltage", "board_log"}
ADB_KINDS = {"sample", "adb_error"}
HEADER = "srtp-board-monitor/v1"


class Timeline:
    def __init__(self, source):
        self.source = source
        self.start = time.monotonic()
        self.events = []
        self.indices = {kind: [] for kind in KINDS}
        self.times = []
        self.device_states = {key: [] for key in ("vision", "panel", "page")}

    def append(self, kind, data):
        self._add({
            "t": round(time.monotonic() - self.start, 3),
            "at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "kind": kind,
            "data": data,
        })

    def _add(self, event):
        position = len(self.events)
        self.indices[event["kind"]].append(position)
        if event["kind"] == "rt_state":
            for updates in self.device_states.values():
                updates.append((position, None))
        elif event["kind"] == "rt_line":
            for key, value in display_updates(event["data"]["text"]).items():
                updates = self.device_states[key]
                if not updates or updates[-1][1] != value:
                    updates.append((position, value))
        self.times.append(event["t"])
        self.events.append(event)

    def device_state(self, key, index):
        updates = self.device_states[key]
        right = bisect_right(updates, index, key=lambda update: update[0])
        return updates[right - 1][1] if right else None

    def recent(self, kind, index, limit):
        indices = self.indices[kind]
        end = bisect_right(indices, index)
        return indices[max(0, end - limit):end]

    def last(self, kinds, index):
        return max((i for kind in kinds for i in self.recent(kind, index, 1)), default=-1)

    def window(self, kinds, index, seconds):
        if index < 0:
            return []
        first = bisect_left(self.times, self.times[index] - seconds)
        return sorted(i for kind in kinds
                      for i in self.indices[kind][bisect_left(self.indices[kind], first):
                                                   bisect_right(self.indices[kind], index)])

    def save(self, path):
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                header = {"format": HEADER, "source": self.source}
                stream.write(json.dumps(header, ensure_ascii=False) + "\n")
                for record in self.events:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def open(cls, path):
        with Path(path).open(encoding="utf-8-sig") as stream:
            header = json.loads(stream.readline(16384))
            if (not isinstance(header, dict) or header.get("format") != HEADER
                    or not isinstance(header.get("source"), str)):
                raise ValueError("未知的上位机会话格式")
            session = cls(header["source"])
            previous = 0
            for number, line in enumerate(iter(lambda: stream.readline(65537), ""), 2):
                try:
                    if len(line) > 65536:
                        raise ValueError("事件行超过长度限制")
                    event = json.loads(line)
                    previous = validate_event(event, previous)
                    session._add(event)
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"第 {number} 行会话事件无效") from exc
            return session


def validate_event(event, previous):
    t, kind, data = event["t"], event["kind"], event["data"]
    if type(t) not in (float, int) or not math.isfinite(t) or t < previous:
        raise ValueError("事件时间无效")
    datetime.fromisoformat(event["at"])
    if kind not in KINDS or not isinstance(data, dict):
        raise ValueError("事件类型无效")
    if kind == "sample":
        raw = data["raw"]
        if (not isinstance(raw, dict) or (raw.keys() | {"resources"}) != SECTIONS.keys()
                or not all(isinstance(value, str) for value in raw.values())
                or data["state"] not in ("waiting", "stale", "fresh", "invalid")
                or not isinstance(data["message"], str)):
            raise ValueError("采样记录无效")
    elif not isinstance(data["text"], str):
        raise ValueError("日志文本无效")
    if kind == "voltage":
        volts = data["volts"]
        if (not isinstance(data["rail"], str) or type(volts) not in (float, int)
                or not math.isfinite(volts) or abs(volts) > 25):
            raise ValueError("电压记录无效")
    return t



def display_updates(text):
    if "DEMO_CONTROLLER_ON " in text:
        return {"vision": "starting", "panel": None, "page": None}
    if "DEMO_CONTROLLER_OFF " in text:
        stopped = re.search(r"\bstatus=0\b", text) is not None
        return {"vision": "stopped" if stopped else "fault", "panel": None, "page": None}
    if "DEMO_CHILD_UNEXPECTED_EXIT" in text:
        return {"vision": "fault", "panel": None, "page": None}
    if re.search(r"\bSINGLE_END status=[1-9]", text):
        return {"vision": "fault", "panel": None, "page": None}
    if "SINGLE_END " in text:
        return {"vision": "stopping", "panel": None, "page": None}
    if "SINGLE_READY " in text:
        return {"vision": "starting", "panel": "shown", "page": "camera"}
    if match := re.search(r"\bSINGLE_KEY1 (shown|hidden)\b", text):
        return {"panel": match[1]}
    if match := re.search(r"\bSINGLE_KEY2 (camera|data)\b", text):
        return {"page": match[1]}
    if "SINGLE_PROGRESS " in text:
        match = re.search(r"\bmode=(camera|data)\b", text)
        return {"vision": "running", "page": match[1]} if match else {"vision": "running"}
    return {}
