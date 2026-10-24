"""Start the real Streamlit server and check its HTTP health endpoint."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

root = Path(__file__).resolve().parents[1]
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
log_path = root / 'reports' / 'runtime.log'
with log_path.open('w', encoding='utf-8') as log:
    process = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', str(root / 'app.py'),
         '--server.address=127.0.0.1', f'--server.port={port}', '--server.headless=true'],
        cwd=root, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    try:
        deadline = time.monotonic() + 50
        while True:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=2) as response:
                    assert response.status == 200
                    assert response.read().decode() == 'ok'
                break
            except (OSError, AssertionError):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f'Server failed; see {log_path}')
                time.sleep(.3)
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=5) as response:
            assert response.status == 200
            assert 'streamlit' in response.read().decode().lower()
        result = {'health_http_status': 200, 'health_body': 'ok', 'homepage_http_status': 200,
                  'note': 'HTTP startup smoke check; interactive rendering is covered separately by Streamlit AppTest.'}
        (root / 'reports' / 'runtime_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
