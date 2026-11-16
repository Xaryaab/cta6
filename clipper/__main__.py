import sys

# Windows consoles default to a legacy codepage; stream titles often contain emoji.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .cli import main  # noqa: E402

main()
