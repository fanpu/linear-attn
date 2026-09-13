"""Collect trained deep-LSA / GD-k points for the explorer widget -> widgets/data_explorer.js"""
import json, pathlib
HERE = pathlib.Path(__file__).parent
pts = []
for f in sorted((HERE / "cache").glob("lsa_deep_*.json")):
    for r in json.load(open(f))["runs"]:
        kind, L, d, n, sigma, cov = r["cfg"]
        if cov == "iso" and d == 10 and kind in ("gd", "dense"):
            pts.append({"kind": kind, "L": L, "n": n, "sigma": sigma, "risk": round(r["risk"], 5)})
(HERE / "widgets" / "data_explorer.js").write_text("window.EXPLORER_DATA = " + json.dumps({"points": pts}) + ";\n")
print(len(pts), "points")
