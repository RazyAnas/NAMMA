"""Run the laptop side of "Send for render" and print the link to open on the phone.

Starts the NAMMA server with a fresh pairing key, opens a Cloudflare quick tunnel to it (free, no
account; needs `cloudflared`: `brew install cloudflared` on a Mac), and prints:

  * the phone link: the hosted recorder (GitHub Pages) with ?server=<tunnel>&key=<key>, which pairs
    the phone with this laptop in one tap and is remembered on the phone;
  * the same page served by the laptop itself through the tunnel, as a fallback.

The tunnel is what makes this work on networks that block phone-to-laptop traffic (campus Wi-Fi),
and it gives the HTTPS address WebXR needs. Uploads go in 8 MB chunks, under the tunnel's 100 MB
request limit. Without cloudflared it falls back to the local network address.

    python scripts/serve_render.py                 # tunnel + key
    python scripts/serve_render.py --no-tunnel     # same Wi-Fi only
    python scripts/serve_render.py --port 8001 --key mykey
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PAGES = "https://razyanas.github.io/namma-space-pages/record/"


def lan_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def wait_for_server(port: int, key: str, timeout: float = 120) -> dict | None:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/remote/hello", headers={"X-Namma-Key": key})
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return json.loads(response.read())
        except OSError:
            time.sleep(0.5)
    return None


def start_tunnel(port: int) -> tuple[subprocess.Popen, str | None]:
    process = subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    found: list[str] = []
    errors: list[str] = []

    def read() -> None:
        for line in process.stdout:
            # The tunnel address is the random four-word host; api.trycloudflare.com is where it asks.
            match = re.search(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com", line)
            if match and not found:
                found.append(match.group(0))
            elif "failed" in line.lower() or " ERR " in line:
                errors.append(line.strip())

    threading.Thread(target=read, daemon=True).start()
    deadline = time.time() + 45
    while not found and time.time() < deadline and process.poll() is None:
        time.sleep(0.2)
    if not found and errors:
        print("cloudflared: " + errors[-1])
    return process, found[0] if found else None


def print_qr(text: str) -> None:
    try:
        import qrcode
    except ImportError:
        print("  (pip install qrcode to get a QR code here)")
        return
    qr = qrcode.QRCode(border=1)
    qr.add_data(text)
    qr.print_ascii(invert=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--key", default=None, help="pairing key (default: a fresh random one)")
    parser.add_argument("--no-tunnel", action="store_true", help="local network only")
    parser.add_argument("--pages", default=PAGES, help="hosted recorder page")
    args = parser.parse_args()

    key = args.key or secrets.token_urlsafe(9)
    env = {**os.environ, "NAMMA_REMOTE_KEY": key}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", str(args.port)],
                              cwd=ROOT, env=env)
    tunnel = None
    try:
        hello = wait_for_server(args.port, key)
        if hello is None:
            print("The server did not start; see the messages above.")
            return 1
        public = None
        if not args.no_tunnel:
            if shutil.which("cloudflared"):
                tunnel, public = start_tunnel(args.port)
                if not public:
                    print("cloudflared started but printed no tunnel address; falling back to the local network.")
                    tunnel.terminate()
                    tunnel = None
            else:
                print("cloudflared is not installed (Mac: brew install cloudflared), so only the local network works.")
        server_url = public or f"http://{lan_address()}:{args.port}"
        query = urllib.parse.urlencode({"server": server_url, "key": key})
        what = f"photoreal splats with {hello['trainer']}" if hello["photoreal"] else (
            "dense points (no splat trainer found: install OpenSplat, or use the Colab notebook for gsplat)")
        print()
        print(f"Laptop ready: {hello['machine']}, can make {what}.")
        if public:
            phone = f"{args.pages}?{query}"
            print(f"\nOpen on the phone:\n  {phone}\n")
            print_qr(phone)
            print(f"\nSame page served by this laptop:\n  {public}/record/?key={urllib.parse.quote(key)}")
        else:
            print(f"\nOn the same Wi-Fi, open on the phone:\n  {server_url}/record/?{query}")
            print("  WebXR needs HTTPS, so record on the hosted page, save the bundle, then open it on this page to send it.")
        print(f"\nPairing key: {key}    Ctrl+C stops the server{' and the tunnel' if tunnel else ''}.\n")
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for process in (tunnel, server):
            if process and process.poll() is None:
                process.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
