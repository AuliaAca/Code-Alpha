"""Generate the README visuals from REAL outputs of the code (nothing is mocked).

    python scripts/make_visuals.py

Design language: a dark "night" surface, one fixed accent colour per task
(so a colour always means the same task in every picture), direct labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.paths import add_task_paths  # noqa: E402

add_task_paths()

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Wedge  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

ASSETS = ROOT / "docs" / "assets"
BG, INK, MUTED, GRID = "#0b0f19", "#e5e7eb", "#94a3b8", "#1e293b"
TASK = {1: "#2dd4bf", 2: "#fbbf24", 3: "#f472b6", 4: "#60a5fa"}   # teal, amber, pink, blue
TITLE = {"family": "DejaVu Sans", "weight": "bold"}


def _canvas(w, h, title, subtitle):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    fig.text(0.04, 0.955, title, color=INK, fontsize=20, **TITLE, va="top")
    fig.text(0.04, 0.905, subtitle, color=MUTED, fontsize=11, va="top")
    return fig


def _save(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / name, dpi=170, facecolor=BG)
    plt.close(fig)
    print("wrote", ASSETS / name)


# --------------------------------------------------------------------------
# 1. Architecture as a metro map
# --------------------------------------------------------------------------
def metro_map():
    fig = _canvas(14, 8, "CodeLab Metro", "four AI lines, three transfer stations - the hub connects them")
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.80], facecolor=BG)
    ax.set_xlim(0, 100), ax.set_ylim(0, 54), ax.axis("off")

    rows = {2: 44, 1: 32, 4: 20, 3: 8}   # row order chosen so the transfers are short
    lines = {
        2: ("TASK 2  FAQ Chatbot", ["faqs.csv", "NLTK clean\n+ stem", "TF-IDF\nvectors", "cosine\nmatch", "Answer"]),
        1: ("TASK 1  Translator", ["Text +\nlanguages", "Google /\nMicrosoft", "fallback\n+ cache", "Translated\ntext", "Copy /\nspeak"]),
        4: ("TASK 4  Tracking", ["Webcam /\nvideo", "Detect\nYOLO | motion", "SORT\nKalman + IoU", "SceneReport", "Annotated\nvideo"]),
        3: ("TASK 3  Music", ["Bach\nchorales", "Note\ntokens", "LSTM\ntraining", "Sample\nnew notes", "MIDI +\nWAV"]),
    }
    xs = [22, 35, 48, 61, 74]
    for task, y in rows.items():
        name, stations = lines[task]
        c = TASK[task]
        ax.plot([xs[0] - 3, xs[-1]], [y, y], color=c, lw=9, solid_capstyle="round", zorder=2)
        ax.text(1, y, name.replace("  ", "\n"), color=c, fontsize=10.5, va="center", **TITLE)
        for x, label in zip(xs, stations):
            ax.add_patch(Circle((x, y), 1.25, fc=BG, ec=INK, lw=2.4, zorder=4))
            beside = x == xs[3] and task in (4, 3)   # keep labels clear of the transfer arrows
            ax.text(x - 1.9 if beside else x, y + 2.9, label, color=INK, fontsize=8.6,
                    ha="right" if beside else "center", va="bottom", zorder=5, linespacing=1.15)

    # transfers: SceneReport (T4) feeds the translator's "Translated text" and the sampler
    def transfer(x, y0, y1, text, side=1):
        ax.plot([x, x], [y0, y1], color=INK, lw=2.2, ls=(0, (2, 2)), zorder=3)
        ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2), zorder=3)
        ax.text(x + side * 1.6, (y0 + y1) / 2, text, color=INK, fontsize=8.6, style="italic",
                va="center", ha="left" if side > 0 else "right")

    transfer(xs[3], 20 - 1.4, 8 + 1.4, "seed notes +\nsampling temperature")
    transfer(xs[3], 20 + 1.4, 32 - 1.4, "object names\nas a sentence")
    for y in (32, 20, 8):   # interchange rings
        ax.add_patch(Circle((xs[3], y), 1.9, fc="none", ec=INK, lw=1.6, zorder=3))

    # the router bus: Task2 answers questions and hands work to the other lines
    bx = 92
    ax.plot([bx, bx], [8, 44], color=INK, lw=2.2, ls=(0, (4, 3)), zorder=1)
    for y in (32, 20, 8):
        ax.plot([xs[-1], bx], [y, y], color=INK, lw=1.6, ls=(0, (2, 2)), zorder=1)
    ax.plot([xs[-1], bx], [44, 44], color=TASK[2], lw=9, solid_capstyle="round", zorder=2)
    ax.add_patch(FancyBboxPatch((86.4, 2.2), 11.5, 46.2, boxstyle="round,pad=0.3,rounding_size=1.6",
                                fc="none", ec=GRID, lw=1.4, zorder=0))
    ax.text(92.1, 51.6, "ROUTER", color=INK, fontsize=10, ha="center", **TITLE)
    ax.text(92.1, 49.4, "one chat box", color=MUTED, fontsize=8, ha="center")
    for y, t in ((44, "FAQ"), (32, "translate ..."), (20, "describe scene"), (8, "compose music")):
        ax.text(bx + 1.4, y + 1.5, t, color=MUTED, fontsize=7.5, ha="left")
        ax.add_patch(Circle((bx, y), 0.9, fc=INK, ec=BG, lw=1.5, zorder=4))
    _save(fig, "metro_map.png")


# --------------------------------------------------------------------------
# 2. Task 1: translation constellation (real provider output)
# --------------------------------------------------------------------------
def language_constellation():
    from translator import build_default_service

    service = build_default_service(offline_only=True)
    sentence, targets = "Hello my friend", ["id", "es", "fr", "de"]
    names = {"id": "Indonesian", "es": "Spanish", "fr": "French", "de": "German"}
    words = {c: service.translate(sentence, "en", c).text for c in targets}

    fig = _canvas(11, 8, "Language Constellation", "one sentence, four translations - real output of the Task 1 service")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.84], facecolor=BG)
    ax.set_xlim(-1.75, 1.75), ax.set_ylim(-1.3, 1.3), ax.set_aspect("equal"), ax.axis("off")
    rng = np.random.default_rng(4)
    ax.scatter(rng.uniform(-1.75, 1.75, 320), rng.uniform(-1.3, 1.3, 320), s=rng.uniform(0.3, 3, 320),
               c=MUTED, alpha=0.35, lw=0)                                     # background stars
    c = TASK[1]
    for r in (0.55, 1.0):
        ax.add_patch(Circle((0, 0), r, fc="none", ec=GRID, lw=1.2, ls=(0, (2, 4))))
    for glow, alpha in ((0.42, 0.06), (0.32, 0.10), (0.25, 0.9)):              # sun glow
        ax.add_patch(Circle((0, 0), glow, fc=c, alpha=alpha, lw=0))
    ax.text(0, 0.0, "Hello\nmy friend", color=BG, fontsize=10, ha="center", va="center", linespacing=1.2, **TITLE)
    ax.text(0, -0.36, "ENGLISH", color=c, fontsize=8.5, ha="center")
    for i, code in enumerate(targets):
        ang = np.pi / 4 + i * np.pi / 2 + 0.25
        r = 0.55 if i % 2 == 0 else 1.0
        px, py = r * np.cos(ang), r * np.sin(ang)
        ax.plot([0, px], [0, py], color=c, lw=0.9, alpha=0.35)
        ax.add_patch(Circle((px, py), 0.055, fc=c, ec=BG, lw=2, zorder=3))
        dx, dy = (0.09 if px > 0 else -0.09), 0.02
        ha = "left" if px > 0 else "right"
        ax.text(px + dx, py + dy + 0.05, words[code], color=INK, fontsize=12.5, ha=ha, va="center", **TITLE)
        ax.text(px + dx, py + dy - 0.045, names[code].upper(), color=c, fontsize=8, ha=ha, va="center")
    ax.text(0, -1.25, "offline lexicon provider shown - with API keys the same chain uses Google / Microsoft",
            color=MUTED, fontsize=8.5, ha="center")
    _save(fig, "translator_constellation.png")


# --------------------------------------------------------------------------
# 3. Task 2: string loom (query -> FAQ similarity threads)
# --------------------------------------------------------------------------
LOOM_QUERIES = [
    "how do I install it", "which languages are supported", "can I use my webcam",
    "what is SORT", "how is music generated", "where does the training data come from",
    "does it read text aloud", "add my own questions", "why do IDs change",
    "how do all four work together", "what is the weather", "who won the football match",
]


def string_loom():
    from faq_bot import FAQMatcher

    m = FAQMatcher()
    sim = m.similarity_matrix(LOOM_QUERIES)                       # (queries, faqs)
    cats = [f["category"] for f in m.faqs]
    order = np.argsort([["General", "Translator", "Chatbot", "Music", "Tracking", "Hub"].index(c) for c in cats],
                       kind="stable")
    sim = sim[:, order]
    n_q, n_f = sim.shape
    fig = _canvas(14, 8.6, "String Loom", "every thread is a cosine similarity between a user question and an FAQ - thickness = score")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.84], facecolor=BG)
    ax.set_xlim(0, 118), ax.set_ylim(-3, 100), ax.axis("off")
    qy = np.linspace(94, 6, n_q)
    fy = np.linspace(97, 3, n_f)
    cat_color = {"General": "#94a3b8", "Translator": TASK[1], "Chatbot": TASK[2 - 1 + 1] if False else TASK[2],
                 "Music": TASK[3], "Tracking": TASK[4], "Hub": "#e5e7eb"}
    x0, x1 = 30, 78
    for i in range(n_q):
        best = sim[i].argmax()
        for j in range(n_f):
            s = sim[i, j]
            if s < 0.06:
                continue
            verts = [(x0, qy[i]), (54, qy[i]), (54, fy[j]), (x1, fy[j])]
            patch = PathPatch(MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]),
                              fc="none", ec=cat_color[cats[order[j]]] if j == best else TASK[2],
                              lw=0.4 + 6.5 * s, alpha=0.95 if j == best else 0.28, capstyle="round")
            ax.add_patch(patch)
        low = sim[i, best] < 0.35
        ax.add_patch(Circle((x0, qy[i]), 0.9, fc=INK if not low else MUTED, ec=BG, lw=1.5, zorder=4))
        ax.text(x0 - 1.8, qy[i], LOOM_QUERIES[i], color=INK if not low else MUTED, fontsize=9.4, ha="right", va="center",
                style="italic" if low else "normal")
        if low:
            ax.text(x0 - 1.8, qy[i] - 2.5, "below threshold - bot says \"I don't know\"", color=MUTED, fontsize=6.8, ha="right")
    for j in range(n_f):
        cat = cats[order[j]]
        ax.add_patch(Circle((x1, fy[j]), 0.55, fc=cat_color[cat], ec=BG, lw=1, zorder=4))
        q = m.faqs[order[j]]["question"]
        ax.text(x1 + 1.5, fy[j], q if len(q) < 44 else q[:43] + "...", color=MUTED, fontsize=6.8, va="center")
    for cat, col in cat_color.items():
        idx = [j for j in range(n_f) if cats[order[j]] == cat]
        ax.plot([x1 + 21.5] * 2, [fy[idx[0]] + 1, fy[idx[-1]] - 1], color=col, lw=3, solid_capstyle="round")
        ax.text(x1 + 23, np.mean([fy[idx[0]], fy[idx[-1]]]), cat, color=col, fontsize=8, va="center", rotation=0)
    _save(fig, "faq_string_loom.png")


# --------------------------------------------------------------------------
# 4. Task 3: vinyl record (generated notes wound onto a disc) + training curves
# --------------------------------------------------------------------------
def music_vinyl():
    from music_gen.data import parse_token
    from music_gen.generate import MusicGenerator

    gen = MusicGenerator.load()
    tokens = gen.generate(96, temperature=0.9, rng_seed=11)
    history = json.loads((ROOT / "Task3/models/bach_lstm.history.json").read_text(encoding="utf-8"))

    fig = _canvas(14, 7.4, "Vinyl of Generated Bach", "96 notes sampled from the trained LSTM - angle = time, radius = pitch, arc length = duration")
    ax = fig.add_axes([0.02, 0.02, 0.56, 0.84], facecolor=BG, aspect="equal")
    ax.set_xlim(-1.1, 1.1), ax.set_ylim(-1.1, 1.1), ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fc="#111827", ec=GRID, lw=2))
    for r in np.linspace(0.36, 0.98, 22):                           # record grooves
        ax.add_patch(Circle((0, 0), r, fc="none", ec="#182033", lw=0.6))
    parsed = [parse_token(t) for t in tokens]
    pitches = [p[0][0] for p in parsed if p[0]]
    lo, hi = min(pitches), max(pitches)
    total = sum(d for _, d in parsed)
    cmap = {0.5: "#f9a8d4", 1.0: TASK[3], 2.0: "#db2777", 4.0: "#831843"}
    t = 0.0
    for ps, d in parsed:
        if ps:
            r = 0.42 + 0.52 * (ps[0] - lo) / max(hi - lo, 1)
            a0 = 90 - 360 * t / total
            a1 = 90 - 360 * (t + d * 0.86) / total
            ax.add_patch(Wedge((0, 0), r + 0.012, a1, a0, width=0.024, fc=cmap[d], ec="none"))
        t += d
    ax.add_patch(Circle((0, 0), 0.28, fc=TASK[3], ec=BG, lw=3))
    ax.add_patch(Circle((0, 0), 0.03, fc=BG))
    ax.text(0, 0.11, "BACH-ISH", color=BG, fontsize=11, ha="center", **TITLE)
    ax.text(0, -0.13, "LSTM  t=0.9", color=BG, fontsize=8.5, ha="center")
    for i, (d, col) in enumerate(cmap.items()):   # legend: dot, then label, evenly spaced
        lx = -0.95 + i * 0.5
        ax.add_patch(Circle((lx, -1.06), 0.018, fc=col, ec="none", clip_on=False))
        ax.text(lx + 0.05, -1.06, f"{d:g} beat", color=MUTED, fontsize=8.5, va="center")

    ax2 = fig.add_axes([0.63, 0.20, 0.33, 0.55], facecolor=BG)
    ep = np.arange(1, len(history["train_loss"]) + 1)
    ax2.plot(ep, history["train_loss"], color=TASK[3], lw=2.2)
    ax2.plot(ep, history["val_loss"], color=INK, lw=2.2)
    b = history["best_epoch"]
    ax2.scatter([b], [history["val_loss"][b - 1]], s=90, color=BG, ec=INK, lw=2, zorder=5)
    ax2.annotate(f"kept epoch {b}\n(val loss lowest)", (b, history["val_loss"][b - 1]), (b - 15, history["val_loss"][b - 1] + 0.9),
                 color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax2.text(ep[-1] + 0.4, history["train_loss"][-1], "train", color=TASK[3], fontsize=10, va="center")
    ax2.text(ep[-1] + 0.4, history["val_loss"][-1], "validation", color=INK, fontsize=10, va="center")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax2.spines[s].set_color(GRID)
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.set_xlabel("epoch", color=MUTED), ax2.set_ylabel("cross-entropy loss", color=MUTED)
    ax2.set_xlim(1, ep[-1] + 7)
    ax2.grid(color=GRID, lw=0.6)
    ax2.set_title("Where it stopped learning and started memorising", color=INK, fontsize=11, loc="left")
    _save(fig, "music_vinyl.png")


# --------------------------------------------------------------------------
# 5. Task 4: light painting (tracker trails as a long exposure)
# --------------------------------------------------------------------------
def light_painting():
    from make_demo_video import make_demo_video  # noqa: E402
    from tracking import ColorLabeler, MotionDetector, Sort, TrackingPipeline, build_detector
    from tracking.annotate import id_color

    sys.path.insert(0, str(ROOT / "Task4"))
    video = ROOT / "Task4" / "samples" / "demo_scene.mp4"
    if not video.exists():
        make_demo_video(video)
    frames = []
    pipeline = TrackingPipeline(build_detector("motion", color_labels=True), Sort())
    report = pipeline.run(video, on_frame=lambda i, f, t: frames.append(f) if i == 100 else None)
    h, w = 360, 640
    scale = 2
    canvas = np.zeros((h * scale, w * scale, 3), np.uint8)
    glow = np.zeros_like(canvas)
    for tid, pts in report.trails.items():
        col = id_color(tid)
        p = (np.array(pts) * scale).astype(np.int32).reshape(-1, 1, 2)
        for thick, alpha in ((22, 0.25), (12, 0.45), (5, 1.0)):       # layered strokes = neon glow
            layer = np.zeros_like(canvas)
            cv2.polylines(layer, [p], False, tuple(int(c * alpha) for c in col), thick, cv2.LINE_AA)
            glow = cv2.add(glow, layer)
    glow = cv2.add(glow, cv2.GaussianBlur(glow, (0, 0), 14))
    canvas = cv2.add(canvas, glow)
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    fig = _canvas(14, 8.4, "Light Painting", "the trails SORT reconstructed from the demo video, like a long-exposure photograph")
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.82], facecolor=BG)
    ax.imshow(img, extent=(0, w, h, 0)), ax.axis("off")
    for tid, pts in report.trails.items():
        x, y = pts[-1]
        label = report.tracks[tid]
        ax.text(min(x + 8, w - 8), y - 10, f"#{tid} {label}", color=INK, fontsize=11, **TITLE,
                ha="right" if x + 8 >= w - 8 else "left")
        ax.scatter([pts[0][0]], [pts[0][1]], s=50, color=tuple(c / 255 for c in id_color(tid)[::-1]), zorder=5, ec=BG)
    _save(fig, "tracking_light_painting.png")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "Task4"))
    which = sys.argv[1:] or ["metro", "constellation", "loom", "vinyl", "light"]
    fns = {"metro": metro_map, "constellation": language_constellation, "loom": string_loom,
           "vinyl": music_vinyl, "light": light_painting}
    for name in which:
        fns[name]()
