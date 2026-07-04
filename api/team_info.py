"""GET /api/team_info — student names and emails."""
import json
from http.server import BaseHTTPRequestHandler

TEAM = {
    "group_batch_order_number": "1_10",
    "team_name": "Group B",
    "students": [
        {"name": "Noam Tsemah", "email": "noam.university@gmail.com"},
        {"name": "Lian Sigawi", "email": "liansigawi696@gmail.com"},
    ],
}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._respond(200, TEAM)

    def do_OPTIONS(self):
        self._respond(204, None)

    def _respond(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if data is not None:
            self.wfile.write(json.dumps(data).encode())
