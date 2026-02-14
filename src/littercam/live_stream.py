"""Standalone MJPEG live stream for camera alignment."""
from __future__ import annotations

import io
import logging
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

logger = logging.getLogger(__name__)

Picamera2 = None
try:
    from picamera2 import Picamera2
except ImportError:
    pass


class MJPEGHandler(BaseHTTPRequestHandler):
    camera = None

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<!doctype html>
<html><head><title>LitterCam Alignment</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:100vh}</style></head>
<body><img src="/stream" /></body></html>""")
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    frame = self.camera.capture_array("main")
                    from PIL import Image
                    img = Image.fromarray(frame)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=70)
                    jpeg = buf.getvalue()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args) -> None:
        pass


def main() -> None:
    if Picamera2 is None:
        print("Error: picamera2 is required. Run on the Raspberry Pi.")
        return

    import socket
    host_ip = socket.gethostbyname(socket.gethostname())
    port = 8001

    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"size": (1280, 720)},
    )
    camera.configure(config)
    camera.start()
    time.sleep(1.0)

    MJPEGHandler.camera = camera

    server = HTTPServer(("0.0.0.0", port), MJPEGHandler)
    print(f"Live stream at http://{host_ip}:{port}")
    print("Stop the capture service first if it's running:")
    print("  systemctl --user stop littercam-capture")
    print()
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.shutdown()
        camera.stop()
        camera.close()


if __name__ == "__main__":
    main()
