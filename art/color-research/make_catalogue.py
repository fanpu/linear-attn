"""Writes COLOR_SCHEMES.md from palettes.SCHEMES / PAIRINGS plus metrics computed here."""
import palettes as P

FAM = [("scientific", "Scientific / cartographic"), ("print", "Print traditions"), ("painting", "Painting"),
       ("design", "Design"), ("film", "Film and photography"), ("nature", "Nature"), ("digital", "Digital"),
       ("astronomy", "Astronomy"), ("textile", "Textiles and natural dyes")]

HEAD = open("catalogue_head.md").read()
out = [HEAD]
for fam, title in FAM:
    out.append(f"\n## {title}\n")
    for s in P.list_schemes(family=fam):
        m = P.metrics(s)
        hexes = " ".join(f"`{h}`" for h in s.sample_hex(9))
        if s.kind in ("categorical", "inks"):
            diag = (f"min pairwise dE2000 {m['min_dE']:.0f} (deuteranopia {m['min_dE_deut']:.0f}, protanopia "
                    f"{m['min_dE_prot']:.0f}); min dL* {m['min_dL']:.0f}")
        else:
            diag = (f"L* {m['L0']:.0f} -> {m['L1']:.0f} (range {m['Lmin']:.0f}-{m['Lmax']:.0f}); J' reversals "
                    f"{m['J_reversals']}; CAM02-UCS speed CV {m['speed_cv']:.2f}, spike {m['spike']:.1f}; "
                    f"deuteranopia retention {m['deut_retention']:.2f}; greyscale retention {m['grey_retention']:.2f}")
        src = ", ".join(f"<{u}>" for u in s.sources)
        ground = f" Ground/paper `{s.ground}`." if s.ground else ""
        lib = f" (library map `{s.lib}`; stops sampled)" if s.lib else ""
        out.append(f"\n### `{s.name}` - {s.title}\n"
                   f"- **Type:** {s.kind}{lib}.{ground}\n- **Stops:** {hexes}\n"
                   f"- **Measured diagnostics:** {diag}.\n- **Caveats:** {s.caveat}\n"
                   f"- **Suits:** {s.suits}\n- **Lineage:** {s.lineage} Sources: {src}\n")
out.append("\n## Split-at-the-boundary pairings (`PAIRINGS`)\n\n"
           "Each side is listed from the seam (boundary) outward. Use `P.render_split(x, name, near_boundary=...)`.\n\n"
           "| name | x < 0 side (seam -> pastel) | x > 0 side (seam -> pastel) | idea |\n|---|---|---|---|\n")
for k, p in P.PAIRINGS.items():
    def fmt(side):
        if isinstance(side, list):
            return " ".join(f"`{h}`" for h in side)
        cm = P.as_cmap(side)
        return " ".join(f"`{P.rgb2hex(c)}`" for c in cm([0, .25, .5, .75, 1.0])[:, :3])
    out.append(f"| `{k}` | {fmt(p['neg'])} | {fmt(p['pos'])} | {p['title']}: {p['lineage']} |\n")
open("COLOR_SCHEMES.md", "w").write("".join(out))
print("ok")
