"""Use the recording demo controller's sequenced K1/KEY1/KEY2 mailbox."""

from dataclasses import dataclass
import re

from board_data import MonitorError, run_adb_shell


ACTIONS = {"k": "视觉启停", "1": "参数面板显隐", "2": "LCD 页面切换"}
CONTROLLER = "252148307857e596f58702a77eb5e96e7286993402fc6f821db9d26b17cb5fd1"
RELEASE = "hardware-preview-last640-200-nv12-20261125"


def control_script(boot, action):
    if action not in ACTIONS or not re.fullmatch(r"[0-9a-f-]{36}", boot):
        raise MonitorError("缺少有效的板端启动号或控制命令")
    return f"""
set -e
cd /tmp/srtp_demo_last640_20261124
test "$(cat /proc/sys/kernel/random/boot_id)" = '{boot}' ||
    {{ echo '板子已重启' >&2; exit 20; }}
test "$(cat READY)" = '{RELEASE}' ||
    {{ echo '当前固件版本未经控制兼容性验证' >&2; exit 21; }}
digest=$(sha256sum single_owner_controller.elf)
test "${{digest%% *}}" = '{CONTROLLER}' ||
    {{ echo '当前控制器未启用已验证的命令接口' >&2; exit 22; }}
exec 9>.board_monitor.lock
flock -xn 9 ||
    {{ echo '另一个调试台正在发送命令' >&2; exit 23; }}
read previous unused < parent_control.txt
case "$previous" in ''|*[!0-9]*) exit 24;; esac
test "${{#previous}}" -le 10 && test "$previous" -lt 2147483646 || exit 24
sequence=$((previous + 1))
tmp=$(mktemp .board_monitor.XXXXXX)
trap 'rm -f "$tmp"' 0
printf '%s %s\\n' "$sequence" '{action}' > "$tmp"
chmod 644 "$tmp"
mv -f "$tmp" parent_control.txt
printf 'SRTP_SENT %s\\n' "$sequence"
"""


def send_command(adb, serial, boot, action):
    output = run_adb_shell(adb, serial, control_script(boot, action))
    match = re.fullmatch(r"SRTP_SENT ([1-9][0-9]*)", output.strip())
    if match is None:
        raise MonitorError("未收到命令写入凭据")
    return int(match[1])


@dataclass
class PendingCommand:
    action: str
    start_index: int
    deadline: float
    next_index: int | None = None
    sequence: int | None = None
    received: bool = False
    starting: bool = False

    def observe(self, text):
        if not self.received:
            self.received = (
                self.sequence is not None
                and re.search(rf"\bDEMO_PARENT_EVENT seq={self.sequence} action={self.action}\b", text)
                is not None
            )
            return None
        if self.action == "1":
            match = re.search(r"\bSINGLE_KEY1 (shown|hidden)\b", text)
            return {"shown": "板端已显示面板", "hidden": "板端已隐藏面板"}[match[1]] if match else None
        if self.action == "2":
            match = re.search(r"\bSINGLE_KEY2 (data|camera)\b", text)
            return {"data": "板端已切换至数据页", "camera": "板端已切换至摄像头页"}[match[1]] if match else None

        if "DEMO_CONTROLLER_ON " in text:
            self.starting = True
        if self.starting and "SINGLE_PROGRESS frames=" in text:
            return "视觉已启动，收到推理进度"
        match = re.search(r"\bDEMO_CONTROLLER_OFF\b.*\bstatus=(\d+)", text)
        if match:
            return "视觉已正常停止" if match[1] == "0" else "视觉停止时返回异常"
        if "DEMO_CHILD_UNEXPECTED_EXIT" in text or re.search(r"\bSINGLE_END status=[1-9]", text):
            return "视觉程序异常退出"
        return None
