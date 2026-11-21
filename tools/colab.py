"""Thin client for the local Colab bridge daemon (127.0.0.1:8787).

The daemon (D:/ragproj/build/colab_daemon.py serve) keeps one colab-mcp process
connected to a Colab browser tab. Restarting it means clicking Accept in the tab
again, so this client only talks to it and never restarts it.

    python tools/colab.py status
    python tools/colab.py run "print(1)"
    python tools/colab.py runf path/to/cell.py
    python tools/colab.py bg <name> <script.py> [args...]   # background python job
    python tools/colab.py bgsh <name> "<shell command>"     # background shell job
    python tools/colab.py tail <name> [lines]
    python tools/colab.py wait <name> <regex> [seconds]     # block (<= 540 s) until regex appears in the log
    python tools/colab.py ps
    python tools/colab.py push <local file> [remote path]   # copy one file into the runtime (default: same path under /content/sealedread)
    python tools/colab.py pull <remote file> <local file>   # copy a runtime file here (gzipped in transit, sha256-checked)

Background jobs run detached on the runtime (start_new_session), log to
/content/logs/<name>.log and record their exit code in /content/logs/<name>.exit,
so no cell ever runs long enough to hit the bridge's RPC timeout.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import pathlib
import re
import socket
import sys

HOST, PORT = "127.0.0.1", 8787
LOGDIR = "/content/logs"
REPO = "/content/sealedread"
MAX_CODE = 300 * 1024  # one websocket frame is 1 MiB; stay well below it


def send(op: dict, timeout: float = 1200) -> tuple[bool, str]:
    conn = socket.create_connection((HOST, PORT), timeout=timeout)
    conn.sendall((json.dumps(op) + "\n").encode("utf-8"))
    chunks = []
    while True:
        b = conn.recv(65536)
        if not b:
            break
        chunks.append(b)
        if b.endswith(b"\n"):
            break
    conn.close()
    res = json.loads(b"".join(chunks).decode("utf-8") or "{}")
    return bool(res.get("ok")), res.get("out", "")


def run(code: str) -> int:
    if len(code.encode("utf-8")) > MAX_CODE:
        print(f"cell is {len(code.encode())} B, over the {MAX_CODE} B limit")
        return 1
    ok, out = send({"op": "run", "code": code})
    sys.stdout.buffer.write((out + "\n").encode("utf-8", errors="replace"))
    return 0 if ok else 1


def _launch(name: str, argv_expr: str, preamble: str = "") -> int:
    code = (
        "import os, pathlib, subprocess, sys\n"
        f"os.makedirs({LOGDIR!r}, exist_ok=True)\n"
        f"{preamble}"
        # keep the previous run's log as <name>.prev.log so waits never match stale lines
        f"logp = pathlib.Path({LOGDIR!r}) / {name + '.log'!r}\n"
        f"if logp.exists(): logp.replace(logp.with_name({name + '.prev.log'!r}))\n"
        "log = open(logp, 'wb')\n"
        f"exitf = pathlib.Path({LOGDIR!r}) / {name + '.exit'!r}\n"
        "exitf.unlink(missing_ok=True)\n"
        f"argv = {argv_expr}\n"
        # a wrapper shell records the exit code when the job finishes
        "wrapped = ['bash', '-c', 'cd \"$0\" 2>/dev/null || cd /content; \"${@}\"; echo $? > ' + str(exitf),"
        f" {REPO!r}] + argv\n"
        "p = subprocess.Popen(wrapped, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)\n"
        f"print('started', {name!r}, 'pid', p.pid)\n"
    )
    return run(code)


def bg(name: str, script: str, args: list[str]) -> int:
    src = pathlib.Path(script).read_bytes()
    pre = (
        f"script = pathlib.Path({LOGDIR!r}) / {name + '.py'!r}\n"
        f"script.write_bytes(__import__('base64').b64decode({base64.b64encode(src).decode()!r}))\n"
    )
    return _launch(name, f"[sys.executable, '-u', str(script)] + {args!r}", pre)


def bgsh(name: str, command: str) -> int:
    return _launch(name, f"['bash', '-lc', {command!r}]")


def tail(name: str, lines: int = 40) -> int:
    code = (
        "import pathlib\n"
        f"p = pathlib.Path({LOGDIR!r}) / {name + '.log'!r}\n"
        f"e = pathlib.Path({LOGDIR!r}) / {name + '.exit'!r}\n"
        f"txt = p.read_text(errors='replace').splitlines()[-{lines}:] if p.exists() else ['(no log yet)']\n"
        "print('\\n'.join(l[-400:] for l in txt))\n"
        "print('[exit code: %s]' % (e.read_text().strip() if e.exists() else 'running'))\n"
    )
    return run(code)


def wait(name: str, pattern: str, seconds: float = 540) -> int:
    seconds = min(float(seconds), 540.0)
    code = (
        "import pathlib, re, time\n"
        f"p = pathlib.Path({LOGDIR!r}) / {name + '.log'!r}\n"
        f"e = pathlib.Path({LOGDIR!r}) / {name + '.exit'!r}\n"
        f"rx = re.compile({pattern!r})\n"
        "t0 = time.time(); hit = False\n"
        f"while time.time() - t0 < {seconds}:\n"
        "    txt = p.read_text(errors='replace') if p.exists() else ''\n"
        "    if rx.search(txt): hit = True; break\n"
        "    if e.exists(): break\n"
        "    time.sleep(5)\n"
        "txt = p.read_text(errors='replace').splitlines()[-15:] if p.exists() else []\n"
        "print('\\n'.join(l[-400:] for l in txt))\n"
        "print('[matched: %s after %.0fs; exit code: %s]' % (hit, time.time() - t0, e.read_text().strip() if e.exists() else 'running'))\n"
    )
    return run(code)


def push(local: str, remote: str | None = None) -> int:
    src = pathlib.Path(local).read_bytes()
    if remote is None:
        rel = pathlib.Path(local).resolve().relative_to(pathlib.Path(__file__).resolve().parents[1])
        remote = f"{REPO}/{rel.as_posix()}"
    code = (
        "import base64, pathlib\n"
        f"p = pathlib.Path({remote!r}); p.parent.mkdir(parents=True, exist_ok=True)\n"
        f"p.write_bytes(base64.b64decode({base64.b64encode(src).decode()!r}))\n"
        "print('wrote', p, p.stat().st_size, 'bytes')\n"
    )
    return run(code)


def _cell_value(code: str, tag: str) -> str:
    ok, out = send({"op": "run", "code": code})
    m = re.search(rf"{tag}:(.*?):END", out, re.S)
    if not ok or not m:
        raise RuntimeError(out[-1500:])
    return m.group(1)


def pull(remote: str, local: str, chunk: int = 180_000) -> int:
    """Gzip the remote file to /tmp, stream it back in base64 chunks, gunzip and verify sha256."""
    meta = _cell_value(
        "import gzip, hashlib, shutil\n"
        f"src = {remote!r}; gz = '/tmp/pull.gz'\n"
        "h = hashlib.sha256(open(src, 'rb').read()).hexdigest()\n"
        "with open(src, 'rb') as a, gzip.open(gz, 'wb', 6) as b: shutil.copyfileobj(a, b)\n"
        # tags are split in the source so the regex only matches the printed output, not an echo of the code
        "import os; print('ME' 'TA:%s %d:E' 'ND' % (h, os.path.getsize(gz)))\n", "META")
    sha, size = meta.split()
    size = int(size)
    buf = bytearray()
    while len(buf) < size:
        part = _cell_value(
            "import base64\n"
            f"f = open('/tmp/pull.gz', 'rb'); f.seek({len(buf)}); d = f.read({chunk})\n"
            "print('B6' '4:' + base64.b64encode(d).decode() + ':E' 'ND')\n", "B64")
        buf += base64.b64decode(part)
        print(f"\rpulled {len(buf) / 1e6:.1f}/{size / 1e6:.1f} MB", end="", flush=True)
    data = gzip.decompress(bytes(buf))
    if hashlib.sha256(data).hexdigest() != sha:
        print("\nsha256 mismatch, not written")
        return 1
    out = pathlib.Path(local)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"\nwrote {out} ({len(data) / 1e6:.1f} MB, sha256 ok)")
    return 0


def ps() -> int:
    code = (
        "import subprocess, pathlib\n"
        "print(subprocess.run(\"ps -eo pid,etime,pcpu,rss,args --sort=-pcpu | head -12\", shell=True,"
        " capture_output=True, text=True).stdout)\n"
        "print(subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',"
        " '--format=csv,noheader'], capture_output=True, text=True).stdout)\n"
        f"d = pathlib.Path({LOGDIR!r})\n"
        "print('jobs:', {f.stem: (f.with_suffix('.exit').read_text().strip() if f.with_suffix('.exit').exists() else 'running')"
        " for f in d.glob('*.log')} if d.exists() else {})\n"
    )
    return run(code)


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "status":
        ok, out = send({"op": "status"}, timeout=60)
        print(out)
        return 0 if ok else 1
    if cmd == "run":
        return run(argv[2])
    if cmd == "runf":
        return run(pathlib.Path(argv[2]).read_text(encoding="utf-8"))
    if cmd == "bg":
        return bg(argv[2], argv[3], argv[4:])
    if cmd == "bgsh":
        return bgsh(argv[2], argv[3])
    if cmd == "tail":
        return tail(argv[2], int(argv[3]) if len(argv) > 3 else 40)
    if cmd == "wait":
        return wait(argv[2], argv[3], float(argv[4]) if len(argv) > 4 else 540)
    if cmd == "ps":
        return ps()
    if cmd == "push":
        return push(argv[2], argv[3] if len(argv) > 3 else None)
    if cmd == "pull":
        return pull(argv[2], argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
