"""Optional receive-only Windows UART reader using the system's .NET serial API."""

import os
import re
import subprocess
import threading


SCRIPT = """
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$port = [IO.Ports.SerialPort]::new('__PORT__', 115200, 'None', 8, 'One')
$port.DtrEnable = $false
$port.RtsEnable = $false
try {
    $port.Open()
    [Console]::WriteLine('===SRTP_UART_OPEN===')
    while ($true) {
        $chunk = $port.ReadExisting()
        if ($chunk.Length) {
            [Console]::Write($chunk)
            [Console]::Out.Flush()
        }
        Start-Sleep -Milliseconds 50
    }
} catch {
    [Console]::WriteLine()
    [Console]::WriteLine('===SRTP_UART_ERROR===' + $_.Exception.Message)
    exit 1
} finally {
    $port.Dispose()
}
"""


class SerialReader:
    def __init__(self, port, emit):
        if os.name != "nt" or not re.fullmatch(r"COM[1-9]\d{0,3}", port):
            raise ValueError("串口监听需要 Windows 串口，例如 COM6")
        self.port = port
        self.emit = emit
        self.process = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             SCRIPT.replace("__PORT__", port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW,
        )
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        self.emit("rt_state", f"{self.port} 正在打开")
        failed = False
        with self.process.stdout as stream:
            for line in iter(lambda: stream.readline(4096), ""):
                line = line.strip()
                if line == "===SRTP_UART_OPEN===":
                    self.emit("rt_state", f"{self.port} 正在接收")
                elif line.startswith("===SRTP_UART_ERROR==="):
                    failed = True
                    self.emit("rt_state", f"{self.port} {line.partition('===SRTP_UART_ERROR===')[2]}")
                elif line:
                    self.emit("rt_line", line)
        self.process.wait()
        if not failed:
            self.emit("rt_state", f"{self.port} 已关闭")

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
