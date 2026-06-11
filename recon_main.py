#!/usr/bin/env python3
"""Reconstruct the truncated manuscript/main.tex by replaying all successful Write/Edit tool
calls (in timestamp order across every session transcript) recorded in the .claude transcripts."""
import json, glob, sys

TARGET = "/home/users/ybi3/mechinterp_brain/manuscript/main.tex"
TX_DIR = "/home/users/ybi3/.claude/projects/-home-users-ybi3"

events = []   # (timestamp, kind, payload, tool_use_id)
results = {}  # tool_use_id -> is_error (bool)

for f in glob.glob(f"{TX_DIR}/*.jsonl"):
    with open(f) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ts = rec.get("timestamp", "")
            msg = rec.get("message", rec)
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                t = c.get("type")
                if t == "tool_use" and c.get("name") in ("Write", "Edit", "MultiEdit"):
                    inp = c.get("input", {})
                    fp = inp.get("file_path", "")
                    if fp != TARGET:
                        continue
                    events.append((ts, c.get("name"), inp, c.get("id")))
                elif t == "tool_result":
                    tid = c.get("tool_use_id")
                    if tid is not None:
                        results[tid] = bool(c.get("is_error", False))

events.sort(key=lambda e: e[0])
content = None
applied = 0
skipped = 0
for ts, kind, inp, tid in events:
    if results.get(tid, False):       # tool errored -> never applied to disk
        skipped += 1; continue
    if kind == "Write":
        content = inp.get("content", "")
        applied += 1
    elif kind == "Edit":
        if content is None:
            continue
        old, new = inp.get("old_string", ""), inp.get("new_string", "")
        if old not in content:
            print(f"  [warn] edit old_string not found @ {ts}: {old[:60]!r}", file=sys.stderr)
            skipped += 1; continue
        if inp.get("replace_all"):
            content = content.replace(old, new)
        else:
            content = content.replace(old, new, 1)
        applied += 1
    elif kind == "MultiEdit":
        for e in inp.get("edits", []):
            old, new = e.get("old_string", ""), e.get("new_string", "")
            if old in content:
                content = content.replace(old, new, 1 if not e.get("replace_all") else -1) if not e.get("replace_all") else content.replace(old, new)
        applied += 1

if content is None:
    print("FATAL: no Write event found for main.tex", file=sys.stderr); sys.exit(1)
open("manuscript/main.tex.recon", "w").write(content)
print(f"applied={applied} skipped={skipped} reconstructed_bytes={len(content)} lines={content.count(chr(10))+1}")
print("wrote manuscript/main.tex.recon")
