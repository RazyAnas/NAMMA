"""Serve NAMMA over HTTPS on the local network so a phone browser can stream live capture.

Phone browsers only allow camera and motion-sensor access on https:// pages, so this
creates a self-signed certificate for this computer's LAN address (once, under data/tls)
and prints the capture URL. Accept the certificate warning on the phone once.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import ipaddress
from pathlib import Path
import socket
import sys

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def lan_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))  # no packet is sent; this only picks the outbound interface
        return probe.getsockname()[0]


def ensure_certificate(directory: Path, address: str) -> tuple[Path, Path]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    cert_path, key_path = directory / f"{address}.crt", directory / f"{address}.key"
    if cert_path.is_file() and key_path.is_file():
        return cert_path, key_path
    directory.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "NAMMA live capture")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(address)), x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--address", default=None, help="LAN address to advertise (auto-detected by default)")
    args = parser.parse_args()
    address = args.address or lan_address()
    cert, key = ensure_certificate(ROOT / "data" / "tls", address)
    print(f"\n  On your phone (same Wi-Fi), open:  https://{address}:{args.port}/capture/\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=args.port, ssl_certfile=str(cert), ssl_keyfile=str(key))


if __name__ == "__main__":
    main()
