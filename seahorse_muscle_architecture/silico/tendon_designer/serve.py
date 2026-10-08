"""Serve the Tendon Designer web app locally and open it in the browser.

    python -m seahorse_muscle_architecture.silico.tendon_designer.serve [--port 8765] [--no-browser]

Plain standard library. Serves ``web/`` over HTTP (WebAssembly and ES modules do not work
from ``file://``), with the right MIME types for .wasm/.js/.mjs and caching switched off so
that edits show up on reload. ``/README.md`` is served from this package folder so the Help
overlay can link to the student guide. Stop the server with Ctrl+C.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socket
import sys
import threading
import webbrowser
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
WEB_DIR = PACKAGE_DIR / "web"

EXTRA_TYPES = {
    ".wasm": "application/wasm",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".css": "text/css",
    ".svg": "image/svg+xml",
    ".stl": "application/octet-stream",
    ".xml": "application/xml",
    ".md": "text/plain; charset=utf-8",  # shown as text instead of downloaded
    ".csv": "text/csv",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **EXTRA_TYPES}

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean == "/README.md" and (PACKAGE_DIR / "README.md").is_file():
            return str(PACKAGE_DIR / "README.md")
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - signature from base class
        # Only report errors; a page load requests ~40 files.
        if len(args) >= 2 and str(args[1]).startswith(("4", "5")):
            sys.stderr.write(f"[tendon designer] {self.address_string()} {format % args}\n")


def find_free_port(host: str, start: int, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in {start}..{start + tries - 1}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the Tendon Designer web app.")
    parser.add_argument("--port", type=int, default=8765, help="port to use (the next free one is taken if busy)")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: only this computer)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    parser.add_argument("--root", type=Path, default=WEB_DIR, help="folder to serve (default: web/ next to this file)")
    args = parser.parse_args(argv)
    web_dir = args.root.resolve()

    if not (web_dir / "index.html").is_file():
        sys.exit(f"Cannot find {web_dir / 'index.html'}.")
    if not (web_dir / "model" / "catalog.json").is_file():
        print("Warning: web/model/ is missing. Build it first with\n"
              "  python -m seahorse_muscle_architecture.silico.tendon_designer.export_assets", file=sys.stderr)

    port = find_free_port(args.host, args.port)
    handler = functools.partial(Handler, directory=str(web_dir))
    server = http.server.ThreadingHTTPServer((args.host, port), handler)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{port}/"
    print(f"Tendon Designer running at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
