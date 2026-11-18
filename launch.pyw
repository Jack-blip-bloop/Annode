"""Desktop launcher: starts the Study Annotator website in the background (if it isn't running yet) and opens it in
your browser. Double-click it, or use the desktop shortcut."""
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:5000"


def running():
    try:
        urllib.request.urlopen(URL + "/privacy", timeout=1.5)
        return True
    except Exception:
        return False


if not running():
    env = dict(os.environ, ADMIN_KEY=os.environ.get("ADMIN_KEY", "local-admin"))
    pyw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    log = open(os.path.join(HERE, "web", "server.log"), "a")
    subprocess.Popen([pyw if os.path.exists(pyw) else sys.executable, os.path.join(HERE, "web", "app.py")],
                     cwd=HERE, env=env, stdout=log, stderr=log,
                     creationflags=0x00000008 | 0x00000200 | 0x08000000)   # detached, own group, no window
    for _ in range(40):                    # wait up to ~20 s for it to start
        time.sleep(0.5)
        if running():
            break
webbrowser.open(URL)
