"""Scrollable HTML gallery of a whole generated dataset (open it in a browser).

One row per sample: RS A | RS B (as camera B sees it) | RS B rotated back | GS A (target),
plus the sample's parameters. Images load lazily while scrolling, so thousands of
samples are fine. The HTML only links to the PNGs, so keep it inside the dataset folder.

  python src/gallery.py rs_layers                 # -> rs_layers/gallery.html
"""
import argparse
import json
from pathlib import Path

PAGE = r"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Galeria zbioru</title>
<style>
:root { --bg:#f6f6f4; --fg:#1d1d1b; --muted:#6b6b66; --card:#ffffff; --line:#deded8; --acc:#2f6fdf; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ececea; --muted:#9a9a94;
  --card:#21211f; --line:#33332f; --acc:#7aa7ff; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line);
  padding:10px 16px; display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; }
h1 { font-size:16px; margin:0 8px 0 0; }
.grp { display:flex; gap:4px; align-items:center; }
button, input { font:inherit; color:var(--fg); background:var(--card); border:1px solid var(--line);
  border-radius:6px; padding:4px 10px; cursor:pointer; }
button.on { border-color:var(--acc); color:var(--acc); font-weight:600; }
input { width:90px; cursor:text; }
.cnt { color:var(--muted); }
main { padding:12px 16px 40px; max-width:1200px; margin:0 auto; }
.heads, .row .imgs { display:grid; grid-template-columns:repeat(4, 1fr); gap:6px; }
.heads { color:var(--muted); font-size:12px; text-align:center; margin-bottom:6px; }
.row { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px; margin-bottom:10px; }
.row img { width:100%; aspect-ratio:1; display:block; image-rendering:pixelated; background:var(--line); border-radius:3px; }
.info { margin-top:6px; color:var(--muted); font-size:12px; }
.info b { color:var(--fg); }
.tag { display:inline-block; padding:0 6px; border-radius:4px; border:1px solid var(--line); margin-right:6px; }
#more { text-align:center; color:var(--muted); padding:20px; }
@media (max-width:600px) { .heads, .row .imgs { grid-template-columns:repeat(2, 1fr); } }
</style></head><body>
<header>
  <h1>__TITLE__</h1>
  <div class="grp">zbiór: <button data-f="split" data-v="all" class="on">wszystko</button>
    <button data-f="split" data-v="train">train</button><button data-f="split" data-v="val">val</button></div>
  <div class="grp">tekstury: <button data-f="tex" data-v="all" class="on">wszystkie</button>
    <button data-f="tex" data-v="photos">zdjęcia</button><button data-f="tex" data-v="procedural">kształty</button></div>
  <div class="grp">skocz do nr: <input id="jump" placeholder="np. 1234"></div>
  <span class="cnt" id="cnt"></span>
</header>
<main>
  <div class="heads"><div>RS A (wejście 1)</div><div>RS B, tak jak widzi kamera (wejście 2)</div>
    <div>RS B obrócony z powrotem</div><div>GS A (cel)</div></div>
  <div id="list"></div><div id="more"></div>
</main>
<script>
const DATA = __DATA__;
const f = { split: "all", tex: "all" };
let rows = [], shown = 0;
const list = document.getElementById("list"), more = document.getElementById("more");

function card(s) {
  const d = document.createElement("div");
  d.className = "row"; d.id = "s" + s.id;
  const p = s.split + "/" + s.id + "/";
  const rot = "transform:rotate(" + (90 * s.k) + "deg)";
  d.innerHTML =
    '<div class="imgs"><img loading="lazy" src="' + p + 'rs_a.png">' +
    '<img loading="lazy" src="' + p + 'rs_b.png">' +
    '<img loading="lazy" style="' + rot + '" src="' + p + 'rs_b.png">' +
    '<img loading="lazy" src="' + p + 'gs_a.png"></div>' +
    '<div class="info"><span class="tag">' + s.split + '</span><span class="tag">' + s.tex + '</span>' +
    '<b>' + s.id + '</b> &nbsp; ' + s.txt + '</div>';
  return d;
}
function addMore(n) {
  const frag = document.createDocumentFragment();
  for (let i = 0; i < n && shown < rows.length; i++, shown++) frag.appendChild(card(rows[shown]));
  list.appendChild(frag);
  more.textContent = shown < rows.length ? "przewiń dalej…" : "koniec";
}
function apply() {
  rows = DATA.filter(s => (f.split === "all" || s.split === f.split) && (f.tex === "all" || s.tex === f.tex));
  list.innerHTML = ""; shown = 0;
  document.getElementById("cnt").textContent = rows.length + " próbek";
  addMore(60);
}
document.querySelectorAll("button[data-f]").forEach(b => b.onclick = () => {
  f[b.dataset.f] = b.dataset.v;
  document.querySelectorAll('button[data-f="' + b.dataset.f + '"]').forEach(x => x.classList.toggle("on", x === b));
  apply();
});
document.getElementById("jump").addEventListener("keydown", e => {
  if (e.key !== "Enter") return;
  const id = e.target.value.trim().padStart(6, "0");
  const i = rows.findIndex(s => s.id === id);
  if (i < 0) { e.target.value = "brak"; return; }
  if (i >= shown) addMore(i - shown + 20);
  document.getElementById("s" + id).scrollIntoView({ block: "start" });
  window.scrollBy(0, -70);
});
new IntersectionObserver(es => { if (es[0].isIntersecting) addMore(60); }, { rootMargin: "1500px" }).observe(more);
apply();
</script></body></html>
"""


def describe(m):
    parts = [f"odczyt {m['readout_a_s'] * 1e3:.0f} ms"]
    if "sync_offset_s" in m:
        parts.append(f"sync {m['sync_offset_s'] * 1e3:+.1f} ms")
    if "baseline_px" in m:
        parts.append(f"baza {m['baseline_px']:.1f} px")
    if "objects" in m:
        objs = m["objects"]
        parts.append(f"{len(objs)} obiekt(y)")
        if objs:
            parts.append("ruch obiektów do " + f"{max(o['speed_px_per_readout'] for o in objs):.0f} px, "
                         + "obrót do " + f"{max(abs(o['spin_rad_per_readout']) for o in objs):.2f} rad")
    w = m.get("cam_w_rad_s", m.get("w_rad_s"))
    if w:
        parts.append(f"drgania {abs(complex(w[0], w[1])):.1f} rad/s, roll {w[2]:+.1f} rad/s")
    return " · ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="dataset folder with train/ and val/")
    ap.add_argument("--out", default=None, help="default: <folder>/gallery.html")
    a = ap.parse_args()
    root = Path(a.folder)
    data = []
    for split in ("train", "val"):
        if not (root / split).is_dir():
            continue
        for d in sorted(p for p in (root / split).iterdir() if p.is_dir()):
            try:
                m = json.loads((d / "meta.json").read_text())
            except (OSError, ValueError):
                continue
            data.append(dict(id=d.name, split=split, k=m.get("b_rot_k", 1),
                             tex=m.get("textures", m.get("kind", "-")), txt=describe(m)))
    data.sort(key=lambda s: s["id"])
    out = Path(a.out) if a.out else root / "gallery.html"
    html = PAGE.replace("__TITLE__", root.resolve().name).replace("__DATA__", json.dumps(data))
    out.write_text(html, encoding="utf-8")
    print(f"{len(data)} samples -> {out}")


if __name__ == "__main__":
    main()
