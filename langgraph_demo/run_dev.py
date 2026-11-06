"""PyCharm 一键启动入口，等价于命令行执行 `langgraph dev`。

在 PyCharm 里右键本文件 -> Run，即可启动 LangGraph 开发服务器
（默认 http://127.0.0.1:2024），无需激活 conda 环境。

说明：
- 直接调用 langgraph-cli 的 click 入口 `langgraph_cli.cli:cli`，
  与 `Scripts\\langgraph.exe` 完全等价。
- 未传参数时默认追加 `dev --no-browser`；想换参数可在
  Run Configuration 的 "Parameters" 里覆盖。
- 强制 stdout 使用 UTF-8，避免 Windows 中文控制台（cp936/GBK）
  在打印 banner 里的 🚀 / 🎨 等 emoji 时抛 UnicodeEncodeError
  （这正是服务启动后立刻退出的常见原因）。
"""

import os
import sys

# 尽量让子进程也继承 UTF-8 设置
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# 当前进程的 stdio 已经是 GBK 时，重新配置为 UTF-8 且不因编码失败而崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # 极老版本或被重定向时忽略
        pass

from langgraph_cli.cli import cli

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += ["dev", "--no-browser"]
    cli()
