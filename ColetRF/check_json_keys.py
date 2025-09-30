#!/usr/bin/env python3
import json, sys, pathlib
from collections import Counter

def main(d):
    d = pathlib.Path(d)
    c = Counter()
    for p in d.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"[ERR] {p.name}: cannot parse ({e})")
            continue
        for row in data:
            for k in ("t","x","y","pupil_size"):
                if k not in row:
                    c[(p.name,k)] += 1
    # print summary
    missing_by_file = {}
    for (fname, k), n in c.items():
        missing_by_file.setdefault(fname, []).append((k, n))
    for fname, lst in missing_by_file.items():
        print(fname, "->", ", ".join(f"{k}:{n}" for k,n in lst))
    if not missing_by_file:
        print("All JSON entries contain t,x,y,pupil_size.")
if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: check_json_keys.py /path/to/json_dir")
        sys.exit(1)
    main(sys.argv[1])
