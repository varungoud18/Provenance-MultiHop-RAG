"""
Stage 3 Launcher: Interactive Web UI Dashboard
Starts the FastAPI backend and launches the browser to http://127.0.0.1:8000
"""

import sys
import time
import webbrowser
import threading
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

def open_browser():
    time.sleep(1.2)
    url = "http://127.0.0.1:8000"
    print(f"\n[LAUNCHER] Opening browser at: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[LAUNCHER] Note: Could not auto-launch browser ({e}). Please visit {url} manually.")

def main():
    print("=" * 80)
    print("   STARTING PROVENANCE INTERACTIVE WEB UI DASHBOARD (STAGE 3)")
    print("=" * 80)
    print(" Serving local API & UI at: http://127.0.0.1:8000")
    print(" Press Ctrl+C in terminal to stop server.")
    print("=" * 80)

    # Launch browser in separate background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start Uvicorn ASGI server
    uvicorn.run(
        "src.server:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    main()
