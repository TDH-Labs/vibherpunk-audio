"""Generate the PWA icons (dark rounded square, amber voice bars)."""
from PIL import Image, ImageDraw

BG = (16, 20, 24)
BAR = (232, 178, 74)
BAR2 = (78, 175, 125)


def make(size, path):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(size * 0.22)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=BG + (255,))
    # five voice bars, symmetric heights
    heights = [0.30, 0.52, 0.78, 0.52, 0.30]
    bw = size * 0.085
    gap = size * 0.055
    total = 5 * bw + 4 * gap
    x = (size - total) / 2
    cy = size / 2
    for i, h in enumerate(heights):
        bh = size * h
        color = BAR if i == 2 else BAR2
        d.rounded_rectangle(
            [x, cy - bh / 2, x + bw, cy + bh / 2], radius=bw / 2, fill=color + (255,)
        )
        x += bw + gap
    img.save(path)
    print("wrote", path)


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "client"
    make(192, f"{out}/icon-192.png")
    make(512, f"{out}/icon-512.png")
