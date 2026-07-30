"""Generates <repo-root>/architecture.png — the diagram returned by GET /api/model_architecture.

Module names here are the contract: they must match the `module` field of every step in
/api/execute and the names used in the README and /api/agent_info. Only real LLM calls are
drawn as modules; the deterministic validation layer is shown separately, hatched, because
it makes no model call and therefore never appears in the steps trace.

Run:  python scripts/make_architecture.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch     # noqa: E402

W, H = 16, 11.5
fig, ax = plt.subplots(figsize=(W, H), dpi=150)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("#F4F8FE")
ax.set_facecolor("#F4F8FE")

INK = "#16243F"
MUTED = "#33415C"
EDGE = "#5A6B86"


def box(x, y, w, h, title, sub, fill, edge, tcol, hatch=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.16",
                                linewidth=2, edgecolor=edge, facecolor=fill, hatch=hatch))
    ax.text(x + w / 2, y + h - 0.36, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=tcol)
    ax.text(x + w / 2, y + h / 2 - 0.30, sub, ha="center", va="center",
            fontsize=7.8, color=MUTED)


def arrow(p1, p2, color=EDGE, style="-|>", rad=0.0, dashed=False, lw=1.8):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=15, color=color, linewidth=lw,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=(0, (5, 3)) if dashed else "solid"))


def label(x, y, text, color=MUTED, size=8.2, weight="bold", style="normal"):
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            color=color, fontweight=weight, fontstyle=style)


ax.text(0.4, H - 0.45, "Trip Planning AI Agent — Architecture", fontsize=21,
        fontweight="bold", color=INK)
ax.text(0.4, H - 0.95, "One intake call routes every turn. Only the branch that needs the "
                       "planner pays for it.", fontsize=9.5, color=MUTED)

# ---- Column 1: the router ----
BW, BH = 2.7, 1.45
X1 = 0.4
box(X1, 6.40, BW, 1.80, "Conversational\nIntake",
    "Reads the dialogue.\nExtracts the profile,\ndetects intent.",
    "#EDE6FB", "#7C5CD6", "#5B36B8")

# ---- Column 2: the three work modules, one row per branch ----
X2 = 4.5
Y_PLAN, Y_EDIT, Y_QA = 8.85, 6.55, 4.60
box(X2, Y_PLAN, BW, BH, "ReAct Planner", "Thought -> Action ->\nObservation loop",
    "#CFEFE9", "#138A7A", "#0C6457")
box(X2, Y_EDIT, BW, BH, "Plan Editor", "Patches only the days\nthe user asked about",
    "#FCE3E9", "#C2436A", "#8E2748")
box(X2, Y_QA, BW, BH, "Itinerary Q&A", "Answers from the plan.\nChanges nothing.",
    "#E4EEDA", "#6C8C3A", "#4B6626")

# ---- Columns 3 and 4 ----
X3, X4 = 8.6, 12.7
Y_FMT = 7.00
box(X3, Y_PLAN, BW, BH, "Reflection Layer", "Critic checks the draft\nbefore delivery",
    "#FCEBC8", "#D79A2B", "#9C6E12")
box(X4, Y_FMT, BW, BH, "Output Formatter", "Day-by-day Markdown\nwith times & costs",
    "#D6EBD9", "#3E9B53", "#2C6E3B")

# ---- Routing edges, labelled with branch + cost ----
MIDX = (X1 + BW + X2) / 2
arrow((X1 + BW, 7.70), (X2, Y_PLAN + 0.72))
label(MIDX, 9.05, "C · new trip", "#0C6457")
label(MIDX, 8.77, "~9 calls", "#0C6457", size=7.2, weight="normal")

arrow((X1 + BW, 7.30), (X2, Y_EDIT + 0.72))
label(MIDX, 7.55, "E · revise", "#8E2748")
label(MIDX, 7.27, "3 calls", "#8E2748", size=7.2, weight="normal")

arrow((X1 + BW, 6.90), (X2, Y_QA + 0.72))
label(MIDX, 5.85, "D · question", "#4B6626")
label(MIDX, 5.57, "2 calls", "#4B6626", size=7.2, weight="normal")

# A and B stop at intake — no downstream module runs at all.
arrow((X1 + BW / 2, 6.40), (X1 + BW / 2, 5.62), color="#7C5CD6")
label(X1 + BW / 2, 5.32, "A · clarify   B · confirm", "#5B36B8", size=8)
label(X1 + BW / 2, 5.04, "1 call — stops here", "#5B36B8", size=7.2, weight="normal")

# Planner -> critic, and the re-plan loop back. The arc stays inside the gap between the
# planner row and the editor row so it never crosses the Plan Editor box.
arrow((X2 + BW, Y_PLAN + 0.72), (X3, Y_PLAN + 0.72))
arrow((X3 + BW / 2, Y_PLAN), (X2 + BW / 2, Y_PLAN),
      color="#D79A2B", rad=-0.28, dashed=True)
label(X3 + BW / 2 + 0.55, 8.42, "re-plan", "#9C6E12", size=8.5, style="italic")

# Both delivery paths converge on the formatter.
arrow((X3 + BW, Y_PLAN + 0.30), (X4 + 0.5, Y_FMT + BH), rad=-0.18)
arrow((X2 + BW, Y_EDIT + 0.72), (X4, Y_FMT + 0.72), color="#C2436A")

# ---- Tools (full width, so nothing runs off the canvas) ----
ax.text(0.4, 4.00, "Tools available to the ReAct Planner", fontsize=10,
        fontweight="bold", color=INK, ha="left")
tools = ["maps_tool", "booking_tool", "flights_tool", "search_tool",
         "reviews_tool", "weather_tool", "calendar_tool"]
tgap, th = 0.13, 0.55
tw = (15.2 - tgap * (len(tools) - 1)) / len(tools)
for i, name in enumerate(tools):
    tx = 0.4 + i * (tw + tgap)
    ax.add_patch(FancyBboxPatch((tx, 3.30), tw, th,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.2, edgecolor="#138A7A", facecolor="#EAF7F4"))
    ax.text(tx + tw / 2, 3.30 + th / 2, name, ha="center", va="center",
            fontsize=7.4, color="#0C6457", fontfamily="monospace")

# ---- Deterministic layer (no LLM call -> never a step) ----
ax.add_patch(FancyBboxPatch((0.4, 2.30), 15.2, 0.72,
                            boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.5,
                            edgecolor="#7A8698", facecolor="#EDF1F7", hatch="///"))
# Knock the hatch out from behind the labels so they stay readable.
_plate = dict(boxstyle="round,pad=0.35", facecolor="#EDF1F7", edgecolor="none")
ax.text(0.7, 2.66, "Validation & Coercion  (schemas.py)", fontsize=9.5,
        fontweight="bold", color=INK, ha="left", va="center", bbox=_plate)
ax.text(6.5, 2.66, "profile · draft plan · patch merge · verdict · budget ceiling   —   "
                   "deterministic, no LLM call, never a step",
        fontsize=8, color=MUTED, ha="left", va="center", bbox=_plate)

# ---- Persistence ----
ax.add_patch(FancyBboxPatch((0.4, 0.30), 7.3, 1.72,
                            boxstyle="round,pad=0.02,rounding_size=0.12",
                            linewidth=1.6, edgecolor="#2E7FC2", facecolor="#E8F1FA"))
ax.text(0.7, 1.76, "Supabase — state & budget ledger", fontsize=9.5,
        fontweight="bold", color="#1F5E94", ha="left")
ax.text(0.7, 1.36, "conversations    current profile + plan, by conversation_id",
        fontsize=7.6, color=MUTED, ha="left", fontfamily="monospace")
ax.text(0.7, 1.04, "traveller_prefs  durable preferences per anonymous device",
        fontsize=7.6, color=MUTED, ha="left", fontfamily="monospace")
ax.text(0.7, 0.72, "runs             tokens + cost per turn (the $9 ledger)",
        fontsize=7.6, color=MUTED, ha="left", fontfamily="monospace")
ax.text(0.7, 0.44, "Optional — unconfigured, the agent runs exactly as before.",
        fontsize=7.4, color=MUTED, ha="left", fontstyle="italic")

# ---- Human-in-the-loop tiers ----
ax.add_patch(FancyBboxPatch((8.0, 1.15), 3.6, 0.87,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            linewidth=1.4, edgecolor="#3E9B53", facecolor="#EAF6EC"))
ax.text(8.25, 1.79, "Tier 1 — Autonomous (read-only)", fontsize=8.4,
        fontweight="bold", color="#2C6E3B", ha="left")
ax.text(8.25, 1.42, "maps · search · reviews · weather · flights_search",
        fontsize=7.2, color=MUTED, ha="left")

ax.add_patch(FancyBboxPatch((12.0, 1.15), 3.6, 0.87,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            linewidth=1.4, edgecolor="#C2434C", facecolor="#FBECEC"))
ax.text(12.25, 1.79, "Tier 2 — Gated (needs approval)", fontsize=8.4,
        fontweight="bold", color="#9C2A33", ha="left")
ax.text(12.25, 1.42, "booking_confirm_tool · flight_book_tool",
        fontsize=7.2, color=MUTED, ha="left")

ax.text(8.0, 0.68, "Every LLM call is time-gated and metered: the run stops before it can "
                   "exceed the function\nbudget or the cost ceiling, and degrades to a "
                   "deterministic renderer rather than failing.",
        fontsize=7.6, color=MUTED, ha="left", va="center")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "architecture.png")

if __name__ == "__main__":
    fig.savefig(_OUT, bbox_inches="tight", facecolor="#F4F8FE")
    print("wrote", _OUT)
