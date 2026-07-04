"""Generates <repo-root>/architecture.png — the diagram returned by GET /api/model_architecture."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

W, H = 14, 8
fig, ax = plt.subplots(figsize=(W, H), dpi=150)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("#F4F8FE")
ax.set_facecolor("#F4F8FE")

def box(x, y, w, h, title, sub, fill, edge, tcol):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                linewidth=2, edgecolor=edge, facecolor=fill))
    ax.text(x + w/2, y + h - 0.42, title, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=tcol)
    ax.text(x + w/2, y + h/2 - 0.28, sub, ha="center", va="center",
            fontsize=8.4, color="#33415C", wrap=True)

ax.text(0.4, H - 0.5, "Trip Planning AI Agent — Architecture", fontsize=20,
        fontweight="bold", color="#16243F")

# ---- Pipeline row ----
modules = [
    ("Conversational\nIntake", "Collects traveller\nneeds via dialogue", "#EDE6FB", "#7C5CD6", "#5B36B8"),
    ("Preference\nProfiler", "Builds typed JSON\ntraveller profile", "#DCEBF7", "#2E7FC2", "#1F5E94"),
    ("ReAct\nPlanner", "Thought -> Action ->\nObservation loop", "#CFEFE9", "#138A7A", "#0C6457"),
    ("Reflection\nLayer", "Critic checks plan\nbefore delivery", "#FCEBC8", "#D79A2B", "#9C6E12"),
    ("Output\nFormatter", "Day-by-day plan\nwith times & costs", "#D6EBD9", "#3E9B53", "#2C6E3B"),
]
bw, bh, gap = 2.25, 1.7, 0.35
x0, ytop = 0.5, 5.2
xs = []
for i, (t, s, f, e, tc) in enumerate(modules):
    x = x0 + i * (bw + gap)
    xs.append(x)
    box(x, ytop, bw, bh, t, s, f, e, tc)
    if i > 0:
        ax.add_patch(FancyArrowPatch((x - gap, ytop + bh/2), (x, ytop + bh/2),
                     arrowstyle="-|>", mutation_scale=16, color="#5A6B86", linewidth=1.8))

# Re-plan loop: Reflection -> ReAct
rx, rrx = xs[2] + bw/2, xs[3] + bw/2
ax.add_patch(FancyArrowPatch((rrx, ytop), (rx, ytop), connectionstyle="arc3,rad=-0.45",
             arrowstyle="-|>", mutation_scale=15, color="#D79A2B", linewidth=1.8, linestyle=(0, (5, 3))))
ax.text((rx + rrx)/2, ytop - 0.95, "Re-plan", ha="center", color="#9C6E12",
        fontsize=10, fontstyle="italic", fontweight="bold")

# ---- Tools row (under ReAct Planner) ----
tools = ["maps_tool", "booking_tool", "flights_tool", "search_tool",
         "reviews_tool", "weather_tool", "calendar_tool"]
ax.text(0.5, 3.05, "Tools available to the ReAct Planner", fontsize=11,
        fontweight="bold", color="#16243F")
tw, th = 1.78, 0.62
for i, name in enumerate(tools):
    tx = 0.5 + i * (tw + 0.18)
    ax.add_patch(FancyBboxPatch((tx, 2.25), tw, th, boxstyle="round,pad=0.02,rounding_size=0.1",
                 linewidth=1.3, edgecolor="#138A7A", facecolor="#EAF7F4"))
    ax.text(tx + tw/2, 2.25 + th/2, name, ha="center", va="center",
            fontsize=8.2, color="#0C6457", fontfamily="monospace")

# ---- Human-in-the-loop tiers ----
ax.text(0.5, 1.55, "Human-in-the-Loop", fontsize=11, fontweight="bold", color="#16243F")
ax.add_patch(FancyBboxPatch((0.5, 0.35), 6.3, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12",
             linewidth=1.4, edgecolor="#3E9B53", facecolor="#EAF6EC"))
ax.text(0.75, 1.12, "Tier 1 — Autonomous (read-only)", fontsize=9, fontweight="bold", color="#2C6E3B")
ax.text(0.75, 0.66, "maps · search · reviews · weather · flights_search", fontsize=8, color="#33415C")

ax.add_patch(FancyBboxPatch((7.1, 0.35), 6.3, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12",
             linewidth=1.4, edgecolor="#C2434C", facecolor="#FBECEC"))
ax.text(7.35, 1.12, "Tier 2 — Gated (requires user approval)", fontsize=9, fontweight="bold", color="#9C2A33")
ax.text(7.35, 0.66, "booking_confirm_tool · flight_book_tool", fontsize=8, color="#33415C")

# Write to the repo root, where GET /api/model_architecture reads architecture.png from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "architecture.png")
fig.savefig(_OUT, bbox_inches="tight", facecolor="#F4F8FE")
print("wrote", _OUT)
