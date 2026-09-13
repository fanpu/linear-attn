"""Write precomputed data for the widgets as widgets/data_*.js (window.X = {...}), loadable from file:// via <script>."""
import json
import numpy as np
from core import geometry_2d, svm


def dump(name, var, obj):
    with open(f"widgets/{name}.js", "w") as f:
        f.write(f"window.{var} = {json.dumps(obj, separators=(',', ':'))};\n")


if __name__ == "__main__":
    X, y = geometry_2d()
    ref = {k: svm(X, y, k)[0].round(10).tolist() for k in ("l2", "linf", "l1")}
    dump("data_geometry", "GEOM_DATA", {"X": X.round(6).tolist(), "y": y.tolist(), "ref": ref})
    print("wrote widgets/data_geometry.js", ref)
