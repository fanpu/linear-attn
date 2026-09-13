"""Render a theory/ blog post (post.md) to a standalone post.html, optionally screenshotting it.

    .venv/bin/python _shared/render_post.py 03-saxe-dynamics/post.md            # -> post.html next to it
    .venv/bin/python _shared/render_post.py 03-saxe-dynamics/post.md --shot     # + _preview/*.png page slices

Math: $inline$ and $$display$$ (pymdownx.arithmatex, rendered client-side by vendored KaTeX, works offline).
Raw HTML blocks (figures, <video>, widgets) pass through untouched; markdown inside <div markdown="1"> is parsed.
$...$ inside raw HTML (captions, widget labels) is also typeset. Write a literal dollar sign as &#36;.

Check widgets for JS errors (prints any console.* / uncaught errors):
    google-chrome --headless=new --no-sandbox --disable-gpu --enable-logging=stderr --v=0 \\
        --virtual-time-budget=8000 --dump-dom file://$PWD/post.html 2>&1 >/dev/null | grep CONSOLE
"""
import argparse, os, pathlib, subprocess, sys

import markdown

SHARED = pathlib.Path(__file__).resolve().parent

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{rel}/vendor/katex/dist/katex.min.css">
<link rel="stylesheet" href="{rel}/post.css">
<script defer src="{rel}/vendor/katex/dist/katex.min.js"></script>
<script defer src="{rel}/vendor/katex/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [{{left: '\\\\[', right: '\\\\]', display: true}}, {{left: '\\\\(', right: '\\\\)', display: false}}, {{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}], ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'], throwOnError: false}})"></script>
</head><body>
{body}
</body></html>
"""


def render(md_path: pathlib.Path) -> pathlib.Path:
    text = md_path.read_text()
    title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("# ")), md_path.parent.name)
    body = markdown.markdown(
        text,
        extensions=["extra", "md_in_html", "toc", "sane_lists", "pymdownx.arithmatex", "pymdownx.superfences"],
        extension_configs={"pymdownx.arithmatex": {"generic": True}},
    )
    rel = os.path.relpath(SHARED, md_path.parent)
    out = md_path.with_suffix(".html")
    out.write_text(TEMPLATE.format(title=title, rel=rel, body=body))
    return out


def shoot(html: pathlib.Path, width: int, height: int, slices: int) -> None:
    """Headless-Chrome screenshots of consecutive page slices (scrolls via #fragment-free JS offset)."""
    prev = html.parent / "_preview"
    prev.mkdir(exist_ok=True)
    for i in range(slices):
        wrapper = prev / f"_shot{i}.html"
        wrapper.write_text(
            f'<html><body style="margin:0;overflow:hidden"><iframe src="../{html.name}" '
            f'style="border:0;width:{width}px;height:{height * slices}px;margin-top:-{i * height}px"></iframe></body></html>'
        )
        png = prev / f"page_{i:02d}.png"
        subprocess.run(
            ["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
             "--allow-file-access-from-files", f"--window-size={width},{height}", "--virtual-time-budget=8000",
             f"--screenshot={png}", wrapper.resolve().as_uri()],
            check=True, capture_output=True, timeout=120,
        )
        wrapper.unlink()
        print(png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("post", type=pathlib.Path)
    ap.add_argument("--shot", action="store_true", help="also save _preview/page_XX.png slices")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=1600)
    ap.add_argument("--slices", type=int, default=8)
    a = ap.parse_args()
    out = render(a.post.resolve())
    print(out)
    if a.shot:
        shoot(out, a.width, a.height, a.slices)
