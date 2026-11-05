"""Serve hsview's browser version locally: ``python web/serve.py`` and open the page it prints.

The page loads the Python package from ``src/hsview``, so the whole repository
is served, not only ``web/``. Any static web server rooted at the repository
works the same way (for example ``python -m http.server`` there, then open
``/web/``), as does GitHub Pages publishing the repository.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                      ".mjs": "text/javascript", ".wasm": "application/wasm", ".py": "text/plain"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")  # always pick up edited Python and JS files
        super().end_headers()


def main(argv=None):
    p = argparse.ArgumentParser(description="Serve the hsview browser version")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = p.parse_args(argv)
    server = http.server.ThreadingHTTPServer((args.host, args.port), functools.partial(Handler, directory=str(ROOT)))
    url = f"http://{args.host}:{server.server_address[1]}/web/"
    print(f"hsview: serving {ROOT} at {url} (Ctrl+C stops)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
