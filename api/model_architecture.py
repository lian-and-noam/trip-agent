"""GET /api/model_architecture — the architecture diagram as image/png."""
import os
from http.server import BaseHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PNG_PATH = os.path.join(ROOT, "architecture.png")


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            with open(PNG_PATH, "rb") as f:
                png = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(png)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(('{"error":"architecture.png not found: %s"}' % e).encode())
