"""Patch aiperf 1.0.7 so --tokenizer accepts a local directory.

Upstream `_resolve_local_snapshot()` always routes through
`huggingface_hub.snapshot_download(local_files_only=True)`, which validates the
argument as a repo id and therefore rejects an absolute model directory.  A
local directory needs no snapshot resolution at all, so short-circuit it.
"""

from pathlib import Path

TARGET = Path("/usr/local/lib/python3.11/site-packages/aiperf/common/tokenizer.py")
OLD = "        candidates = [name]\n"
NEW = (
    "        if __import__('os').path.isdir(name):\n"
    "            return name\n"
    "        candidates = [name]\n"
)

text = TARGET.read_text(encoding="utf-8")
if NEW in text:
    print("already patched")
else:
    assert OLD in text, "unexpected aiperf tokenizer.py layout"
    TARGET.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print("patched local-directory tokenizer support")
