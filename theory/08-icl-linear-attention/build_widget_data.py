"""Collect trained deep-LSA / GD-k points for the explorer widget -> widgets/data_explorer.js
Uses the large-sample re-evaluation (eval_deep.py) when available: mean squared error over 2^18 prompts."""
import json, pathlib
HERE = pathlib.Path(__file__).parent
ev = json.load(open(HERE / "cache" / "deep_eval.json"))
pts = []
for v in ev.values():
    if v["cov"] == "iso" and v["d"] == 10 and v["kind"] in ("gd", "dense"):
        pts.append({"kind": v["kind"], "L": v["L"], "n": v["n"], "sigma": v["sigma"], "risk": round(v["mean"], 5),
                    "trim": round(v["trim999"], 5), "sem": round(v["sem"], 5)})
(HERE / "widgets" / "data_explorer.js").write_text("window.EXPLORER_DATA = " + json.dumps({"points": pts}) + ";\n")
print(len(pts), "points")
