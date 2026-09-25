"""
CNN VISION PIPELINE - fancy edition

Watch one handwritten digit travel through your trained network.
Every tile animates what its layer really does, using the real
activations and weights of the model:

  1 INPUT    pixels are read in row by row
  2 CONV 1   a 3x3 kernel slides over the image, feature map is written
             value by value, then all 16 filters are shown
  3 POOL     a 2x2 window keeps the maximum of every patch
  4 CONV 2   same idea, but each kernel looks through all 16 maps
  5 POOL     2x2 max again
  6 OUTPUT   flatten -> fully connected -> 10 scores -> prediction
"""

import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgb, to_rgba

from model import MNISTCNN
from data import test_loader


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_INDEX = 0            # which image of the first test batch to use
FRAME_INTERVAL_MS = 50      # speed of the animation (lower = faster)
SAVE_PATH = None            # e.g. "cnn_pipeline.gif" to save instead of showing

BG = "#06080f"
PANEL = "#0f1424"
PANEL_DARK = "#0a0d17"
EDGE = "#262e45"
MUTED = "#8993a8"
DIM = "#3a4560"
GOOD = "#6ee7a1"
BAD = "#ff6b6b"

# one consistent accent colour used for every stage
ACCENT = "#7dd3fc"   # sky blue
ACC = [ACCENT] * 6


def mix(c1, c2, f):
    """blend two colours (f=0 -> c1, f=1 -> c2)"""
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a * (1 - f) + b * f)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def fmt_pct(fraction, decimals=1):
    """
    Format a 0..1 fraction as a percentage string.

    Softmax outputs are almost never exactly 1.0, but rounding a value
    like 0.9996 to one decimal place prints "100.0%", which reads as
    absolute certainty the model doesn't actually have. This clamps the
    *displayed* value just below 100 unless the fraction truly is 1.0.
    """
    pct = fraction * 100
    cap = 100 - 10 ** (-decimals)

    if pct >= 100 and fraction < 1.0:
        pct = cap

    return f"{pct:.{decimals}f}%"


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

model = MNISTCNN()

model.load_state_dict(
    torch.load(
        "mnist_cnn.pth",
        weights_only=True
    )
)

model.eval()


# ============================================================
# 2. GET ONE MNIST IMAGE
# ============================================================

images, labels = next(iter(test_loader))

image = images[SAMPLE_INDEX:SAMPLE_INDEX + 1]
true_label = labels[SAMPLE_INDEX].item()


# ============================================================
# 3. CAPTURE INTERMEDIATE ACTIVATIONS
# ============================================================

activations = {}


def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()

    return hook


model.network[0].register_forward_hook(save_activation("conv1"))
model.network[2].register_forward_hook(save_activation("pool1"))
model.network[3].register_forward_hook(save_activation("conv2"))
model.network[5].register_forward_hook(save_activation("pool2"))


# ============================================================
# 4. FORWARD PASS
# ============================================================

with torch.no_grad():
    output = model(image)

predicted_label = output.argmax(dim=1).item()
probabilities = torch.softmax(output[0], dim=0).numpy()
confidence = float(probabilities[predicted_label])
is_correct = predicted_label == true_label


# ============================================================
# 5. EVERYTHING AS NUMPY (arrays used by the animation)
# ============================================================

img = image[0, 0].numpy().astype(np.float32)          # (28, 28)

conv1_raw = activations["conv1"][0].numpy()           # (16, 28, 28)  before ReLU
pool1_np = activations["pool1"][0].numpy()            # (16, 14, 14)
conv2_raw = activations["conv2"][0].numpy()           # (32, 14, 14)  before ReLU
pool2_np = activations["pool2"][0].numpy()            # (32, 7, 7)

# learned kernels
w1 = model.network[0].weight.detach().numpy()[:, 0]           # (16, 3, 3)
w2 = model.network[3].weight.detach().numpy().mean(axis=1)    # (32, 3, 3) avg over 16 input maps
w1_lim = float(np.abs(w1).max())
w2_lim = float(np.abs(w2).max())


def n_params(module):
    return int(sum(p.numel() for p in module.parameters()))


params_conv1 = n_params(model.network[0])
params_conv2 = n_params(model.network[3])


def max_pool_2x2(x):
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


# The hooks capture the conv output *before* the ReLU.  If the model uses
# ReLU between conv and pool (the usual case) we show the post-ReLU maps,
# because that is what the pooling layer actually sees.
USE_RELU = bool(
    np.allclose(max_pool_2x2(np.maximum(conv1_raw, 0)), pool1_np, atol=1e-5)
)

conv1_show = np.maximum(conv1_raw, 0) if USE_RELU else conv1_raw
conv2_show = np.maximum(conv2_raw, 0) if USE_RELU else conv2_raw

# the feature maps with the most structure get the slow "sweeping" animation
best1 = int(conv1_show.std(axis=(1, 2)).argmax())
best2 = int(conv2_show.std(axis=(1, 2)).argmax())


# ============================================================
# 6. TIMELINE  (what happens in which frame)
# ============================================================

PHASES = [
    ("input",         30),
    ("conv1_scan",    50),
    ("conv1_filters", 32),
    ("pool1",         40),
    ("conv2_scan",    50),
    ("conv2_filters", 40),
    ("pool2",         30),
    ("output",        45),
    ("result",        40),
]

PHASE_LABELS = [
    "READ PIXELS", "SLIDE KERNEL", "ALL 16 FILTERS", "2×2 MAX", "SLIDE KERNEL",
    "ALL 32 FILTERS", "2×2 MAX", "FLATTEN + SCORE", "RESULT",
]

PHASE_START = [int(v) for v in np.cumsum([0] + [d for _, d in PHASES])]
TOTAL_FRAMES = PHASE_START[-1]

# which tile is "working" during each phase
ACTIVE_TILE = [0, 1, 1, 2, 3, 3, 4, 5, 5]

GROUP_START, GROUP_END = {}, {}

for i in range(len(PHASES)):
    g = ACTIVE_TILE[i]
    GROUP_START.setdefault(g, PHASE_START[i])
    GROUP_END[g] = PHASE_START[i + 1]

STATUS = [
    "Reading pixels: 784 brightness values, from dark to bright",
    "A 3×3 kernel slides over the image, one dot product per position",
    "16 different kernels → 16 feature maps (edges, strokes, curves)",
    "Keep only the strongest value of every 2×2 patch → half the size",
    "Each kernel looks through all 16 maps at once (one map shown)",
    "32 kernels combine simple strokes into larger shapes",
    "2×2 max again → 7 × 7 × 32 = 1568 numbers",
    "Flatten the 1568 values → fully connected layer → 10 scores",
    f"The network says: {predicted_label}   ({fmt_pct(confidence)} confident)",
]


def locate(frame):
    """frame -> (phase index, progress inside the phase in (0, 1])"""
    for i, (_, duration) in enumerate(PHASES):
        if frame < PHASE_START[i + 1]:
            return i, (frame - PHASE_START[i] + 1) / duration
    return len(PHASES) - 1, 1.0


# ============================================================
# 7. FIGURE, BACKGROUND, TITLE
# ============================================================

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(BG)


def make_background():
    h, w = 180, 320
    y, x = np.mgrid[0:h, 0:w]
    x = x / (w - 1)
    y = y / (h - 1)                                   # y = 0 is the bottom

    top_blue = np.exp(-(((x - 0.5) * 1.4) ** 2 + (y - 1.0) ** 2 * 1.2) * 3.0)
    right_pink = np.exp(-(((x - 0.92) * 1.6) ** 2 + (y - 0.0) ** 2 * 1.5) * 3.5)
    left_teal = np.exp(-(((x - 0.05) * 1.6) ** 2 + (y - 0.1) ** 2 * 1.5) * 3.5)

    rgb = (
        np.array(to_rgb(BG))
        + top_blue[..., None] * np.array([0.05, 0.09, 0.22])
        + right_pink[..., None] * np.array([0.17, 0.04, 0.14])
        + left_teal[..., None] * np.array([0.02, 0.10, 0.12])
    )

    return np.clip(rgb, 0, 1)


# full-figure axes: its data coordinates == figure fractions (0..1)
bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
bg.imshow(make_background(), extent=(0, 1, 0, 1), origin="lower",
          aspect="auto", interpolation="bilinear")
bg.set_xlim(0, 1)
bg.set_ylim(0, 1)
bg.set_autoscale_on(False)
bg.axis("off")

dots_x, dots_y = np.meshgrid(np.linspace(0.02, 0.98, 49), np.linspace(0.03, 0.97, 28))
bg.scatter(dots_x.ravel(), dots_y.ravel(), s=1.3, c="white", alpha=0.05, linewidths=0)

# gradient title with a soft glow (monospace so every letter can be placed exactly)
TITLE = "CNN VISION PIPELINE"
TITLE_FS = 30
title_cmap = LinearSegmentedColormap.from_list("title", [ACCENT, ACCENT, ACCENT])
pitch = 0.602 * TITLE_FS / 72 / 16 * 1.22
x_start = 0.5 - pitch * len(TITLE) / 2

for i, letter in enumerate(TITLE):
    if letter == " ":
        continue

    color = title_cmap(i / (len(TITLE) - 1))

    fig.text(
        x_start + (i + 0.5) * pitch, 0.925, letter,
        ha="center", va="center", color=color, fontsize=TITLE_FS,
        family="DejaVu Sans Mono", weight="bold",
        path_effects=[
            pe.Stroke(linewidth=10, foreground=color, alpha=0.10),
            pe.Stroke(linewidth=5, foreground=color, alpha=0.20),
            pe.Normal(),
        ],
    )

fig.text(0.5, 0.878, "Watch one handwritten digit travel through a convolutional neural network",
         ha="center", va="center", color=MUTED, fontsize=11.5)


# ============================================================
# 8. TILES
# ============================================================

def rounded_box(x, y, w, h, fc, ec, lw=1.5, z=-1, pad=0.006, alpha=1.0):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size={0.012 + pad - 0.006}",
        mutation_aspect=16 / 9,               # keeps corners round on a 16:9 figure
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc if fc is not None else "none",
        alpha=alpha,
        transform=fig.transFigure,
        zorder=z,
    )
    fig.add_artist(box)
    return box


TILE_Y, TILE_H = 0.33, 0.50
TILE_TOP = TILE_Y + TILE_H
TILE_W, OUT_W, GAP, X0 = 0.122, 0.19, 0.028, 0.03

TILES = [
    dict(title="INPUT",  sub="28 × 28 × 1",  caption="pixels",          first=0,
         stat="784 pixel values"),
    dict(title="CONV 1", sub="28 × 28 × 16", caption="kernel scanning", first=1,
         stat=f"{params_conv1:,} learned weights"),
    dict(title="POOL",   sub="14 × 14 × 16", caption="keep strongest",  first=3,
         stat="no weights · just max()"),
    dict(title="CONV 2", sub="14 × 14 × 32", caption="deeper features", first=4,
         stat=f"{params_conv2:,} learned weights"),
    dict(title="POOL",   sub="7 × 7 × 32",   caption="compress",        first=6,
         stat="1568 values leave here"),
    dict(title="OUTPUT", sub="10 classes",   caption="scores",          first=7,
         stat="1568 → 10 scores"),
]

for i, tile in enumerate(TILES):
    tile["x"] = X0 + i * (TILE_W + GAP)
    tile["w"] = OUT_W if i == 5 else TILE_W
    tile["cx"] = tile["x"] + tile["w"] / 2
    tile["accent"] = ACC[i]

GLOW_SPECS = [(0.016, 9, 0.07), (0.011, 5, 0.13), (0.007, 2.5, 0.28)]

tile_boxes = []
tile_glows = []
tile_header = []          # artists that get dimmed until their tile is reached
tile_captions = []
placeholders = []

for i, tile in enumerate(TILES):
    x, w, accent = tile["x"], tile["w"], tile["accent"]

    tile_boxes.append(rounded_box(x, TILE_Y, w, TILE_H, PANEL, EDGE, z=-1))

    tile_glows.append([
        rounded_box(x, TILE_Y, w, TILE_H, None, accent, lw=lw, z=-2, pad=p, alpha=a)
        for p, lw, a in GLOW_SPECS
    ])

    badge = fig.text(
        x + 0.017, TILE_TOP - 0.036, str(i + 1),
        ha="center", va="center", color=BG, fontsize=9, weight="bold",
        bbox=dict(boxstyle="circle,pad=0.35", fc=accent, ec="none"),
    )
    title = fig.text(x + 0.036, TILE_TOP - 0.031, tile["title"],
                     va="center", color="white", fontsize=13, weight="bold")
    sub = fig.text(x + 0.036, TILE_TOP - 0.059, tile["sub"],
                   va="center", color=MUTED, fontsize=7.5)
    rule = Line2D([x + 0.012, x + w - 0.012], [TILE_TOP - 0.078] * 2,
                  transform=fig.transFigure, color=accent, linewidth=1.5, alpha=0.7)
    fig.add_artist(rule)

    tile_header.append([badge, title, sub, rule])

    tile_captions.append(
        fig.text(tile["cx"], 0.372, tile["caption"], ha="center", va="center",
                 color=MUTED, fontsize=8.5, weight="bold")
    )
    fig.text(tile["cx"], 0.347, tile["stat"], ha="center", va="center",
             color=DIM, fontsize=7.5)

    if i > 0:
        placeholders.append((
            i,
            fig.text(tile["cx"], 0.60, "waiting…", ha="center", va="center",
                     color="#232b40", fontsize=10),
        ))


# ============================================================
# 9. HELPERS: axes, reveal, glow rectangles, insets, mosaics
# ============================================================

MAP_W, MAP_H, MAP_Y = 0.10, 0.178, 0.555


def make_cmap(name):
    cmap = copy.copy(plt.get_cmap(name))
    cmap.set_bad(PANEL_DARK)              # not-yet-computed cells are drawn dark
    return cmap


def frame_axes(ax, accent):
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_color(mix(accent, PANEL, 0.45))
        spine.set_linewidth(1.3)

    ax.set_facecolor(PANEL_DARK)


def make_map_axes(tile_index, data, cmap_name, size):
    tile = TILES[tile_index]
    ax = fig.add_axes([tile["cx"] - MAP_W / 2, MAP_Y, MAP_W, MAP_H])

    im = ax.imshow(data, cmap=make_cmap(cmap_name), interpolation="nearest")

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_autoscale_on(False)
    frame_axes(ax, tile["accent"])

    return ax, im


class GlowRect:
    """Rectangle with a soft halo; slides over a source map or marks one cell."""

    def __init__(self, ax, size, color, fill=False, halos=((7, 0.14), (4, 0.28))):
        self.parts = []

        for lw, alpha in halos:
            part = Rectangle((0, 0), size, size, fill=False, edgecolor=color,
                             linewidth=lw, alpha=alpha, visible=False)
            ax.add_patch(part)
            self.parts.append(part)

        if fill:
            core = Rectangle((0, 0), size, size, facecolor=color,
                             edgecolor="none", visible=False)
        else:
            core = Rectangle((0, 0), size, size, fill=False, edgecolor=color,
                             linewidth=2, visible=False)

        ax.add_patch(core)
        self.parts.append(core)

    def set_xy(self, xy):
        for part in self.parts:
            part.set_xy(xy)

    def set_visible(self, visible):
        for part in self.parts:
            part.set_visible(visible)


def reveal(arr, n):
    """Copy of arr where only the first n cells (row-major) are filled in."""
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    out.reshape(-1)[:n] = arr.reshape(-1)[:n]
    return out


def show(im, full, n=None):
    lo, hi = float(full.min()), float(full.max())

    if hi - lo < 1e-6:
        hi = lo + 1e-6

    im.set_clim(lo, hi)
    im.set_data(full if n is None else reveal(full, n))


ROW_Y = 0.40            # bottom of the inset / mosaic row


class GridInset:
    """Tiny n x n grid with the numbers written in each cell."""

    def __init__(self, tile_index, n, cmap_name, title):
        tile = TILES[tile_index]

        self.n = n
        self.cmap = plt.get_cmap(cmap_name)
        self.ax = fig.add_axes([tile["x"] + 0.010, ROW_Y, 0.042, 0.0747])

        self.im = self.ax.imshow(np.zeros((n, n)), cmap=self.cmap,
                                 vmin=0, vmax=1, interpolation="nearest")
        self.ax.set_xlim(-0.5, n - 0.5)
        self.ax.set_ylim(n - 0.5, -0.5)
        self.ax.set_autoscale_on(False)
        frame_axes(self.ax, tile["accent"])
        self.ax.set_title(title, color=MUTED, fontsize=6.5, pad=3)

        self.texts = [
            [self.ax.text(c, r, "", ha="center", va="center", fontsize=6.5)
             for c in range(n)]
            for r in range(n)
        ]

        self.highlight = Rectangle((-0.5, -0.5), 1, 1, fill=False,
                                   edgecolor="white", linewidth=2, visible=False)
        self.ax.add_patch(self.highlight)

    def update(self, values, vmin, vmax, mark_max=False):
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6

        self.im.set_data(values)
        self.im.set_clim(vmin, vmax)

        norm = Normalize(vmin, vmax)

        for r in range(self.n):
            for c in range(self.n):
                v = float(values[r, c])
                red, green, blue, _ = self.cmap(norm(v))
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue

                label = self.texts[r][c]
                label.set_text(f"{v:.1f}")
                label.set_color("black" if luminance > 0.55 else "white")

        if mark_max:
            r, c = np.unravel_index(values.argmax(), values.shape)
            self.highlight.set_xy((c - 0.5, r - 0.5))
            self.highlight.set_visible(True)


class Mosaic:
    """Thumbnails of ALL feature maps of a layer, filled in while it is computed."""

    def __init__(self, tile_index, maps, cmap_name, ncols, title):
        tile = TILES[tile_index]
        n, h, w = maps.shape
        gap = 1
        nrows = int(np.ceil(n / ncols))
        H = nrows * (h + gap) + gap
        W = ncols * (w + gap) + gap

        big = 10 ** 6
        self.full = np.full((H, W), np.nan, dtype=np.float32)
        self.owner = np.full((H, W), big, dtype=np.int32)     # which channel a pixel belongs to
        self.order = np.full((H, W), big, dtype=np.int32)     # row-major index inside its thumbnail
        self.pos = []

        cell_order = np.arange(h * w).reshape(h, w)

        for ch in range(n):
            r, c = divmod(ch, ncols)
            y0 = gap + r * (h + gap)
            x0 = gap + c * (w + gap)

            m = maps[ch]
            self.full[y0:y0 + h, x0:x0 + w] = (m - m.min()) / (m.max() - m.min() + 1e-6)
            self.owner[y0:y0 + h, x0:x0 + w] = ch
            self.order[y0:y0 + h, x0:x0 + w] = cell_order
            self.pos.append((x0, y0, w, h))

        self.ax = fig.add_axes([tile["x"] + 0.062, ROW_Y - 0.005, 0.048, 0.0853])
        self.im = self.ax.imshow(np.full((H, W), np.nan), cmap=make_cmap(cmap_name),
                                 vmin=0, vmax=1, interpolation="nearest")
        self.ax.set_xlim(-0.5, W - 0.5)
        self.ax.set_ylim(H - 0.5, -0.5)
        self.ax.set_autoscale_on(False)
        frame_axes(self.ax, tile["accent"])
        self.ax.set_title(title, color=MUTED, fontsize=6.5, pad=3)

        self.mark = Rectangle((0, 0), 1, 1, fill=False, edgecolor="white",
                              linewidth=1.3, visible=False)
        self.ax.add_patch(self.mark)

    def _mark(self, ch):
        if ch is None:
            self.mark.set_visible(False)
            return

        x0, y0, w, h = self.pos[ch]
        self.mark.set_xy((x0 - 1, y0 - 1))
        self.mark.set_width(w + 1)
        self.mark.set_height(h + 1)
        self.mark.set_visible(True)

    def show_channels(self, count, current=None):
        """channels 0..count-1 are complete"""
        self.im.set_data(np.where(self.owner < count, self.full, np.nan))
        self._mark(current)

    def show_cells(self, n, only=None, mark=None):
        """first n cells of every thumbnail (or of one channel only)"""
        mask = self.order < n

        if only is not None:
            mask &= self.owner == only

        self.im.set_data(np.where(mask, self.full, np.nan))
        self._mark(mark if mark is not None else only)


# ============================================================
# 10. THE SIX MINI-VISUALISATIONS
# ============================================================

# ---- 1 INPUT -------------------------------------------------
ax_input, input_im = make_map_axes(0, img, "gray", 28)

scan_lines = [
    ax_input.plot([-0.5, 27.5], [-0.5, -0.5], color=ACC[0], linewidth=lw,
                  alpha=alpha, visible=False)[0]
    for lw, alpha in ((7, 0.18), (3, 0.9))
]

win_conv1 = GlowRect(ax_input, 3, ACC[1])                     # kernel 1 slides over the input

# colour legend under the digit
tile0 = TILES[0]
legend_ax = fig.add_axes([tile0["x"] + 0.015, 0.445, tile0["w"] - 0.03, 0.014])
legend_ax.imshow(np.linspace(0, 1, 128)[None, :], cmap="gray", aspect="auto")
frame_axes(legend_ax, ACC[0])
fig.text(tile0["cx"], 0.482, "pixel brightness", ha="center", va="center",
         color=MUTED, fontsize=7)
fig.text(tile0["x"] + 0.015, 0.425, "dark", ha="left", va="center", color=MUTED, fontsize=7)
fig.text(tile0["x"] + tile0["w"] - 0.015, 0.425, "bright", ha="right", va="center",
         color=MUTED, fontsize=7)

# ---- 2 CONV 1 ------------------------------------------------
ax_conv1, conv1_im = make_map_axes(1, conv1_show[best1], "magma", 28)

cur_conv1 = GlowRect(ax_conv1, 1, ACC[1], fill=True, halos=((3, 0.3),))
win_pool1 = GlowRect(ax_conv1, 2, ACC[2])                     # pool 1 window slides over conv 1

kernel_inset1 = GridInset(1, 3, "coolwarm", "kernel weights")
mosaic_conv1 = Mosaic(1, conv1_show, "magma", 4, "all 16 maps")

# ---- 3 POOL 1 ------------------------------------------------
ax_pool1, pool1_im = make_map_axes(2, pool1_np[best1], "viridis", 14)

cur_pool1 = GlowRect(ax_pool1, 1, ACC[2], fill=True, halos=((3, 0.3),))
win_conv2 = GlowRect(ax_pool1, 3, ACC[3])                     # kernel 2 slides over pool 1

patch_inset1 = GridInset(2, 2, "viridis", "current 2×2 patch")
mosaic_pool1 = Mosaic(2, pool1_np, "viridis", 4, "all 16 maps")

# ---- 4 CONV 2 ------------------------------------------------
ax_conv2, conv2_im = make_map_axes(3, conv2_show[best2], "plasma", 14)

cur_conv2 = GlowRect(ax_conv2, 1, ACC[3], fill=True, halos=((3, 0.3),))
win_pool2 = GlowRect(ax_conv2, 2, ACC[4])                     # pool 2 window slides over conv 2

kernel_inset2 = GridInset(3, 3, "coolwarm", "kernel (avg of 16)")
mosaic_conv2 = Mosaic(3, conv2_show, "plasma", 8, "all 32 maps")

# ---- 5 POOL 2 ------------------------------------------------
ax_pool2, pool2_im = make_map_axes(4, pool2_np[best2], "viridis", 7)

cur_pool2 = GlowRect(ax_pool2, 1, ACC[4], fill=True, halos=((3, 0.3),))

patch_inset2 = GridInset(4, 2, "viridis", "current 2×2 patch")
mosaic_pool2 = Mosaic(4, pool2_np, "viridis", 8, "all 32 maps")

# ---- 6 OUTPUT ------------------------------------------------
tile5 = TILES[5]
BAR_IDLE = "#4a5f8f"

ax_output = fig.add_axes([tile5["x"] + 0.03, 0.40, 0.13, 0.333])

ax_output.barh(range(10), [1] * 10, height=0.68, color="#151b2e")          # empty tracks
bars = ax_output.barh(range(10), [0] * 10, height=0.68, color=BAR_IDLE)

ax_output.set_xlim(0, 1.32)
ax_output.set_ylim(9.5, -0.5)                                              # digit 0 on top
ax_output.set_yticks(range(10))
ax_output.set_yticklabels([str(i) for i in range(10)], color="white",
                          fontsize=9, weight="bold")
ax_output.set_xticks([])
ax_output.tick_params(axis="y", length=0)
ax_output.set_facecolor("none")

for name, spine in ax_output.spines.items():
    spine.set_visible(name == "left")
    spine.set_color(EDGE)

percent_texts = [
    ax_output.text(0, i, "", va="center", ha="left", color="white", fontsize=7.5)
    for i in range(10)
]

# things that only appear once their tile has been "computed"
VISIBLE_FROM_PHASE = [
    (ax_conv1, 1), (kernel_inset1.ax, 1), (mosaic_conv1.ax, 1),
    (ax_pool1, 3), (patch_inset1.ax, 3), (mosaic_pool1.ax, 3),
    (ax_conv2, 4), (kernel_inset2.ax, 4), (mosaic_conv2.ax, 4),
    (ax_pool2, 6), (patch_inset2.ax, 6), (mosaic_pool2.ax, 6),
    (ax_output, 7),
]

TRANSIENT = [
    *scan_lines,
    win_conv1, cur_conv1,
    win_pool1, cur_pool1,
    win_conv2, cur_conv2,
    win_pool2, cur_pool2,
]


# ============================================================
# 11. DATA-FLOW RAIL (a glowing packet travels from tile to tile)
# ============================================================

RAIL_Y = 0.2975
NODE_X = [t["cx"] for t in TILES]

bg.plot([NODE_X[0], NODE_X[-1]], [RAIL_Y, RAIL_Y], color=EDGE, linewidth=2, zorder=1)

for tile in TILES:                      # short stems from the tiles down to the rail
    bg.plot([tile["cx"]] * 2, [RAIL_Y, TILE_Y - 0.011], color=EDGE, linewidth=1.5, zorder=1)

rail_fill, = bg.plot([NODE_X[0], NODE_X[0]], [RAIL_Y, RAIL_Y], color=ACC[0],
                     linewidth=3, solid_capstyle="round", zorder=2)

MID_X = [(NODE_X[i] + NODE_X[i + 1]) / 2 for i in range(5)]
rail_arrows = bg.scatter(MID_X, [RAIL_Y] * 5, marker=">", s=55, c=[DIM] * 5,
                         linewidths=0, zorder=3)
rail_nodes = bg.scatter(NODE_X, [RAIL_Y] * 6, s=80, c=[DIM] * 6,
                        edgecolors=BG, linewidths=1.5, zorder=4)

TRAIL = 8
TRAIL_SIZES = np.array([130, 90, 68, 50, 38, 28, 20, 13])
TRAIL_ALPHAS = np.array([1.0, 0.8, 0.62, 0.46, 0.34, 0.24, 0.16, 0.10])

comet_halo = bg.scatter([0], [RAIL_Y], s=900, c=[to_rgba("white", 0.15)],
                        linewidths=0, zorder=5, visible=False)
comet = bg.scatter(np.zeros(TRAIL), np.full(TRAIL, RAIL_Y), s=TRAIL_SIZES,
                   linewidths=0, zorder=6, visible=False)


# ============================================================
# 12. BOTTOM DASHBOARD
# ============================================================

PANEL_Y, PANEL_H = 0.06, 0.205

rounded_box(0.03, PANEL_Y, 0.65, PANEL_H, "#0c111f", EDGE, z=-1)

stage_text = fig.text(0.05, 0.238, "", va="center", fontsize=10.5, weight="bold")
status_text = fig.text(0.05, 0.19, "", va="center", color="white", fontsize=13.5)

rounded_box(0.045, 0.085, 0.62, 0.062, "#070a12", "#1d2438", lw=1, z=-1, pad=0.003)
detail_text = fig.text(0.058, 0.116, "", va="center", color="#a5f3fc", fontsize=10.5,
                       family="DejaVu Sans Mono")

# two result cards: ground truth and prediction
CARD_X = [0.705, 0.845]
CARD_W = 0.125

rounded_box(CARD_X[0], PANEL_Y, CARD_W, PANEL_H, PANEL, EDGE, z=-1)
card_pred_box = rounded_box(CARD_X[1], PANEL_Y, CARD_W, PANEL_H, PANEL, EDGE, z=-1)

card_pred_glow = [
    rounded_box(CARD_X[1], PANEL_Y, CARD_W, PANEL_H, None, GOOD, lw=lw, z=-2, pad=p, alpha=a)
    for p, lw, a in GLOW_SPECS
]

cx_true = CARD_X[0] + CARD_W / 2
cx_pred = CARD_X[1] + CARD_W / 2

fig.text(cx_true, 0.232, "TRUE DIGIT", ha="center", va="center", color=MUTED,
         fontsize=9, weight="bold")
fig.text(cx_true, 0.148, str(true_label), ha="center", va="center", color="white",
         fontsize=46, weight="bold")
fig.text(cx_true, 0.093, "ground truth", ha="center", va="center", color=DIM, fontsize=8)

fig.text(cx_pred, 0.232, "PREDICTION", ha="center", va="center", color=MUTED,
         fontsize=9, weight="bold")
pred_digit = fig.text(cx_pred, 0.148, "?", ha="center", va="center", color=DIM,
                      fontsize=46, weight="bold")
pred_sub = fig.text(cx_pred, 0.093, "thinking…", ha="center", va="center",
                    color=DIM, fontsize=8, weight="bold")

# segmented progress bar (one segment per stage, aligned with its tile)
SEG_Y, SEG_H = 0.028, 0.006
seg_fill = []

for i, tile in enumerate(TILES):
    fig.add_artist(Rectangle((tile["x"], SEG_Y), tile["w"], SEG_H,
                             transform=fig.transFigure, facecolor="#1b2233",
                             edgecolor="none"))
    seg = Rectangle((tile["x"], SEG_Y), 0, SEG_H, transform=fig.transFigure,
                    facecolor=tile["accent"], edgecolor="none")
    fig.add_artist(seg)
    seg_fill.append(seg)


# ============================================================
# 13. ANIMATION FUNCTION
# ============================================================

def conv_detail(raw, ch, r, c, all_maps):
    value = float(raw[ch, r, c])
    where = "3×3 patch × kernel, all 16 maps" if all_maps else "3×3 patch × kernel"
    text = f"output[{r:>2},{c:>2}] = Σ({where}) + bias = {value:+.2f}"

    if USE_RELU:
        text += f"  →  ReLU  →  {max(value, 0):.2f}"

    return text


def pool_detail(source, r, c):
    patch = source[2 * r:2 * r + 2, 2 * c:2 * c + 2]
    a, b, cc, d = (float(v) for v in patch.reshape(-1))

    return f"max({a:.2f}, {b:.2f}, {cc:.2f}, {d:.2f}) = {patch.max():.2f}"


def animate(frame):
    idx, t = locate(frame)
    active = ACTIVE_TILE[idx]
    pulse = 0.5 + 0.5 * np.sin(frame * 0.45)

    # ---- reset everything that only shows up temporarily -------
    for artist in TRANSIENT:
        artist.set_visible(False)

    for ax, first_phase in VISIBLE_FROM_PHASE:
        ax.set_visible(idx >= first_phase)

    detail = ""
    live_caption = {}                      # tile index -> live caption text

    # ---- 1 INPUT: pixels are read row by row -------------------
    if idx == 0:
        rows = int(round(28 * t))
        show(input_im, img, rows * 28)

        for line in scan_lines:
            line.set_ydata([rows - 0.5, rows - 0.5])
            line.set_visible(True)

        live_caption[0] = f"row {rows} / 28"
    else:
        show(input_im, img)

    # ---- 2 CONV 1: kernel slides over the input ----------------
    ch1 = best1

    if idx == 1:
        n = max(1, int(np.ceil(t * 784)))
        r, c = divmod(n - 1, 28)

        show(conv1_im, conv1_show[ch1], n)
        mosaic_conv1.show_cells(n, only=ch1)

        win_conv1.set_xy((c - 1.5, r - 1.5))       # 3×3 window, padding 1
        win_conv1.set_visible(True)
        cur_conv1.set_xy((c - 0.5, r - 0.5))
        cur_conv1.set_visible(True)

        detail = conv_detail(conv1_raw, ch1, r, c, all_maps=False)
        live_caption[1] = f"filter {ch1 + 1} · pixel {n}/784"

    elif idx == 2:
        ch1 = min(15, int(t * 16))
        show(conv1_im, conv1_show[ch1])
        mosaic_conv1.show_channels(ch1 + 1, current=ch1)

        detail = "every kernel responds to a different pattern"
        live_caption[1] = f"filter {ch1 + 1} / 16"

    elif idx >= 3:
        show(conv1_im, conv1_show[ch1])
        mosaic_conv1.show_channels(16, current=best1)

    if idx >= 1:
        kernel_inset1.update(w1[ch1], -w1_lim, w1_lim)

    # ---- 3 POOL 1: 2×2 max over conv 1 -------------------------
    if idx >= 3:
        n = max(1, int(np.ceil(t * 196))) if idx == 3 else 196
        r, c = divmod(n - 1, 14)

        show(pool1_im, pool1_np[best1], n if idx == 3 else None)

        source = conv1_show[best1]
        patch = source[2 * r:2 * r + 2, 2 * c:2 * c + 2]
        patch_inset1.update(patch, float(source.min()), float(source.max()),
                            mark_max=True)

        if idx == 3:
            mosaic_pool1.show_cells(n, mark=best1)

            win_pool1.set_xy((2 * c - 0.5, 2 * r - 0.5))
            win_pool1.set_visible(True)
            cur_pool1.set_xy((c - 0.5, r - 0.5))
            cur_pool1.set_visible(True)

            detail = pool_detail(source, r, c)
            live_caption[2] = f"patch {n} / 196"
        else:
            mosaic_pool1.show_channels(16, current=best1)

    # ---- 4 CONV 2: kernel slides over pool 1 -------------------
    ch2 = best2

    if idx == 4:
        n = max(1, int(np.ceil(t * 196)))
        r, c = divmod(n - 1, 14)

        show(conv2_im, conv2_show[ch2], n)
        mosaic_conv2.show_cells(n, only=ch2)

        win_conv2.set_xy((c - 1.5, r - 1.5))
        win_conv2.set_visible(True)
        cur_conv2.set_xy((c - 0.5, r - 0.5))
        cur_conv2.set_visible(True)

        detail = conv_detail(conv2_raw, ch2, r, c, all_maps=True)
        live_caption[3] = f"filter {ch2 + 1} · pos {n}/196"

    elif idx == 5:
        ch2 = min(31, int(t * 32))
        show(conv2_im, conv2_show[ch2])
        mosaic_conv2.show_channels(ch2 + 1, current=ch2)

        detail = "every kernel responds to a different combination of strokes"
        live_caption[3] = f"filter {ch2 + 1} / 32"

    elif idx >= 6:
        show(conv2_im, conv2_show[ch2])
        mosaic_conv2.show_channels(32, current=best2)

    if idx >= 4:
        kernel_inset2.update(w2[ch2], -w2_lim, w2_lim)

    # ---- 5 POOL 2: 2×2 max over conv 2 -------------------------
    if idx >= 6:
        n = max(1, int(np.ceil(t * 49))) if idx == 6 else 49
        r, c = divmod(n - 1, 7)

        show(pool2_im, pool2_np[best2], n if idx == 6 else None)

        source = conv2_show[best2]
        patch = source[2 * r:2 * r + 2, 2 * c:2 * c + 2]
        patch_inset2.update(patch, float(source.min()), float(source.max()),
                            mark_max=True)

        if idx == 6:
            mosaic_pool2.show_cells(n, mark=best2)

            win_pool2.set_xy((2 * c - 0.5, 2 * r - 0.5))
            win_pool2.set_visible(True)
            cur_pool2.set_xy((c - 0.5, r - 0.5))
            cur_pool2.set_visible(True)

            detail = pool_detail(source, r, c)
            live_caption[4] = f"patch {n} / 49"
        else:
            mosaic_pool2.show_channels(32, current=best2)

    # ---- 6 OUTPUT: bars grow one class after the other ---------
    if idx >= 7:
        for i, bar in enumerate(bars):
            fill = float(np.clip(t * 1.25 - 0.25 * i / 9, 0, 1)) if idx == 7 else 1.0
            width = float(probabilities[i]) * fill
            winner = idx >= 8 and i == predicted_label

            bar.set_width(width)
            bar.set_color(ACC[5] if winner else BAR_IDLE)

            label = percent_texts[i]
            label.set_position((width + 0.03, i))
            label.set_text(fmt_pct(width, decimals=0) if width > 0.005 else "")
            label.set_color(ACC[5] if winner else "white")
            label.set_weight("bold" if winner else "normal")

        if idx == 7:
            live_caption[5] = "computing scores…"
        else:
            live_caption[5] = f"{fmt_pct(confidence, decimals=1)} sure"

    # ---- tiles: glow, edge colour, dimming ---------------------
    for i, tile in enumerate(TILES):
        reached = idx >= tile["first"]
        is_active = i == active

        for glow, (_, _, base_alpha) in zip(tile_glows[i], GLOW_SPECS):
            glow.set_visible(is_active)
            glow.set_alpha(min(1.0, base_alpha * (0.6 + 0.8 * pulse)))

        if is_active:
            edge = tile["accent"]
        elif reached:
            edge = mix(tile["accent"], EDGE, 0.55)
        else:
            edge = EDGE

        tile_boxes[i].set_edgecolor(edge)
        tile_boxes[i].set_linewidth(2.5 if is_active else 1.5)

        dim = 1.0 if reached else 0.3

        for artist in tile_header[i]:
            artist.set_alpha(dim)

        tile_header[i][0].get_bbox_patch().set_alpha(dim)

        caption = tile_captions[i]

        if i in live_caption:
            caption.set_text(live_caption[i])
            caption.set_color("white")
        else:
            caption.set_text(tile["caption"])
            caption.set_color(MUTED)

    for tile_index, placeholder in placeholders:
        placeholder.set_visible(idx < TILES[tile_index]["first"])

    # ---- data-flow rail ----------------------------------------
    g = active
    u = (frame + 1 - GROUP_START[g]) / (GROUP_END[g] - GROUP_START[g])

    if g < 5:
        travel = float(smoothstep((u - 0.6) / 0.4))          # packet leaves in the last 40 %
        head = NODE_X[g] + (NODE_X[g + 1] - NODE_X[g]) * travel
        packet_color = mix(ACC[g], ACC[g + 1], travel)
        moving = 0 < travel < 1
    else:
        head = NODE_X[5]
        packet_color = ACC[5]
        moving = False

    rail_fill.set_data([NODE_X[0], head], [RAIL_Y, RAIL_Y])
    rail_fill.set_color(ACC[g])

    node_colors, node_sizes = [], []

    for i, tile in enumerate(TILES):
        reached = idx >= tile["first"]
        node_colors.append(tile["accent"] if reached else DIM)
        node_sizes.append(80 + (90 * pulse if i == active else 0))

    rail_nodes.set_facecolor(node_colors)
    rail_nodes.set_sizes(node_sizes)

    rail_arrows.set_facecolor([
        ACC[i + 1] if idx >= TILES[i + 1]["first"] else DIM for i in range(5)
    ])

    comet.set_visible(moving)
    comet_halo.set_visible(moving)

    if moving:
        spacing = 0.008
        xs = np.maximum(head - np.arange(TRAIL) * spacing, NODE_X[g])

        comet.set_offsets(np.column_stack([xs, np.full(TRAIL, RAIL_Y)]))
        comet.set_facecolor([to_rgba(packet_color, a) for a in TRAIL_ALPHAS])
        comet_halo.set_offsets([[head, RAIL_Y]])
        comet_halo.set_facecolor([to_rgba(packet_color, 0.16)])

    # ---- dashboard ---------------------------------------------
    stage_text.set_text(
        f"STAGE {active + 1} / 6   {TILES[active]['title']}   ›   {PHASE_LABELS[idx]}"
    )
    stage_text.set_color(ACC[active])
    status_text.set_text(STATUS[idx])
    detail_text.set_text("›  " + (detail if detail else "…"))

    if idx >= 8:
        result_color = GOOD if is_correct else BAD

        pred_digit.set_text(str(predicted_label))
        pred_digit.set_color(result_color)
        pred_sub.set_text(("CORRECT" if is_correct else "WRONG") + f"  ·  {fmt_pct(confidence, decimals=0)}")
        pred_sub.set_color(result_color)

        card_pred_box.set_edgecolor(result_color)
        card_pred_box.set_linewidth(2.5)

        for glow, (_, _, base_alpha) in zip(card_pred_glow, GLOW_SPECS):
            glow.set_edgecolor(result_color)
            glow.set_visible(True)
            glow.set_alpha(min(1.0, base_alpha * (0.6 + 0.8 * pulse)))
    else:
        pred_digit.set_text("?")
        pred_digit.set_color(DIM)
        pred_sub.set_text("thinking…")
        pred_sub.set_color(DIM)

        card_pred_box.set_edgecolor(EDGE)
        card_pred_box.set_linewidth(1.5)

        for glow in card_pred_glow:
            glow.set_visible(False)

    # ---- segmented progress bar --------------------------------
    for i, seg in enumerate(seg_fill):
        if frame + 1 >= GROUP_END[i]:
            done = 1.0
        elif frame < GROUP_START[i]:
            done = 0.0
        else:
            done = (frame + 1 - GROUP_START[i]) / (GROUP_END[i] - GROUP_START[i])

        seg.set_width(TILES[i]["w"] * done)


# ============================================================
# 14. START ANIMATION
# ============================================================

animation = FuncAnimation(
    fig,
    animate,
    frames=TOTAL_FRAMES,
    interval=FRAME_INTERVAL_MS,
    blit=False,
    repeat=True,
)

if SAVE_PATH:
    animation.save(SAVE_PATH, writer="pillow", fps=1000 // FRAME_INTERVAL_MS, dpi=80)
else:
    plt.show()