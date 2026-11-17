#!/usr/bin/env python3
"""UltraRig read-only probe.

Talks to a Tank-G running UltraRig over its USB serial port and asks it to
describe itself. It sends ONLY query commands taken from the UltraRig control
page (PING, MODEL, MEM, FXT, FXS, PLIST, CLIST, LVL). Nothing that edits,
saves, uploads, deletes or changes audio is in the allow-list, and the script
refuses to send anything outside it.

Setup (Windows):
    pip install pyserial
    Close the UltraRig control page / Android app first (only one program can
    hold the port), then:
    python ultrarig_probe.py            (auto-finds the pedal)
    python ultrarig_probe.py COM5       (or name the port)

Writes everything it hears to ultrarig_probe_<timestamp>.txt next to itself.
"""
import sys, time, datetime, pathlib

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is missing: run  pip install pyserial")

VID = 0x3654   # the control page filters on this USB vendor id
READ_ONLY = {  # command -> predicate that marks the end of its reply
    "PING":  lambda l: l.startswith("PONG"),
    "MODEL": lambda l: l.startswith("MODEL"),
    "MEM":   lambda l: l.startswith("MEM"),
    "FXT":   lambda l: l in ("FXT end", "FXT fin"),
    "FXS":   None,                                   # end marker unknown: stop when the line goes quiet
    "PLIST": lambda l: l.startswith(("PLIST end", "PLIST fin")),
    "CLIST": lambda l: l.startswith(("CLIST end", "CLIST fin")),
    "LVL":   lambda l: l.startswith("LVL"),
}

def find_port():
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid == VID:
            return p.device
    print("No port with USB vendor id 0x3654 found. Ports seen:")
    for p in ports:
        print(f"  {p.device}  vid={p.vid} pid={p.pid}  {p.description}")
    sys.exit("Plug the pedal in (data cable, no hub) or pass the COM port as an argument.")

class Pedal:
    def __init__(self, port):
        self.s = serial.Serial(port, 115200, timeout=0.05)
        self.buf = b""

    def lines(self, end, timeout, idle):
        out, t0, last = [], time.time(), time.time()
        while time.time() - t0 < timeout:
            chunk = self.s.read(4096)
            if chunk:
                last = time.time()
                self.buf += chunk
                while b"\n" in self.buf:
                    raw, self.buf = self.buf.split(b"\n", 1)
                    ln = raw.decode("utf-8", "replace").rstrip("\r")
                    if not ln.strip() or ln.startswith("@"):   # '@' = unsolicited push lines
                        continue
                    out.append(ln)
                    if ln.startswith("ERR") or (end and end(ln)):
                        return out
            elif end is None and out and time.time() - last > idle:
                return out
        return out

    def cmd(self, c, timeout=4.0, idle=0.6):
        if c.split()[0] not in READ_ONLY:
            raise RuntimeError(f"refusing non-read-only command {c!r}")
        self.s.reset_input_buffer(); self.buf = b""
        self.s.write((c + "\n").encode())
        return self.lines(READ_ONLY[c.split()[0]], timeout, idle)

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_port()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = pathlib.Path(__file__).with_name(f"ultrarig_probe_{stamp}.txt")
    print(f"Opening {port} ...")
    try:
        p = Pedal(port)
    except serial.SerialException as e:
        sys.exit(f"Couldn't open {port}: {e}\nIs the control page or another program still connected?")
    with open(log, "w", encoding="utf-8") as f:
        def emit(s):
            print(s); f.write(s + "\n")
        pong = []
        for _ in range(5):                       # first packet after opening is sometimes lost
            pong = p.cmd("PING", timeout=0.8)
            if any(l.startswith("PONG") for l in pong):
                break
        emit(f"### PING\n" + "\n".join(pong))
        if not any(l.startswith("PONG") for l in pong):
            emit("No PONG: wrong port, or the pedal isn't running UltraRig.")
            return
        for c in ["MODEL", "MEM", "LVL", "FXT", "FXS", "PLIST", "CLIST"]:
            emit(f"\n### {c}")
            lines = p.cmd(c)
            emit("\n".join(lines) or "(no reply)")
            if c == "FXT" and lines:                  # the block catalogue the bridge can use offline (bridge.py --catalogue)
                (pathlib.Path(__file__).resolve().parent / "fxt_catalogue.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {log}")

if __name__ == "__main__":
    main()
