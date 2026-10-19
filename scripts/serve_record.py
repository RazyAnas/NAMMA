"""Serve only the phone sensor recorder (/record/) over HTTPS on the local network.

Needs nothing beyond the Python that ships with macOS and the system `openssl`, unlike
scripts/serve_live.py, which loads the whole backend (COLMAP, Open3D). The recorder is a static
page that saves its bundle on the phone, so no backend is required. Phone browsers only allow
WebXR, camera and motion sensors on https:// pages, hence the self-signed certificate.

    python3 scripts/serve_record.py            # then open the printed URL on the phone
"""

from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path
import socket
import ssl
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "app" / "static" / "record"


def lan_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))  # no packet is sent; this only picks the outbound interface
        return probe.getsockname()[0]


def ensure_certificate(directory: Path, address: str) -> tuple:
    cert, key = directory / f"record-{address}.crt", directory / f"record-{address}.key"
    if not (cert.is_file() and key.is_file()):
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
             "-keyout", str(key), "-out", str(cert), "-subj", "/CN=NAMMA recorder"],
            check=True, capture_output=True,
        )
    return cert, key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8443)
    args = parser.parse_args()
    address = lan_address()
    cert, key = ensure_certificate(ROOT / "data" / "tls", address)
    http.server.SimpleHTTPRequestHandler.extensions_map[".js"] = "text/javascript"  # module scripts need it
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(PAGE))
    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert), str(key))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print(f"\n  On your phone (same Wi-Fi), open:  https://{address}:{args.port}/\n  Press Ctrl+C here to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
