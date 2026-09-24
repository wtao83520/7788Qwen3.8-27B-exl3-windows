#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
生成控制面板的图标（favicon / 手机主屏图标）。

为什么要这个脚本，而不是只在 index.html 里写个 SVG 就完事：
    · 浏览器标签页多数支持 SVG favicon，但 Windows 的 .ico 被不少地方用到
      （书签、任务栏固定、资源管理器预览），需要一个真正的多尺寸 ICO；
    · 手机浏览器「添加到主屏幕」只认 PNG（apple-touch-icon），不认 SVG；
    · 而这些 PNG/ICO 没法从 SVG 直接生成 —— Pillow 不能读 SVG，
      装 cairosvg 又多一个重依赖。所以这里用 PIL 把**同一套几何形状**重画一遍。

    ⇒ 改了 web/icon.svg 的配色/比例，记得把下面的常量同步改掉再重跑本脚本，
      并在浏览器里肉眼确认两个版本长得一样。两处是刻意重复的：宁可重复，
      也不为了「单一数据源」去引入一个只为构建图标存在的依赖。

用法：
    .\.venv\Scripts\python.exe make_icon.py            # 写进 web/
    .\.venv\Scripts\python.exe make_icon.py --preview  # 顺便拼一张放大预览图
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    print("需要 pillow：.venv\\Scripts\\python.exe -m pip install pillow")
    sys.exit(1)

BASE = Path(__file__).resolve().parent
WEB_DIR = BASE / "web"

# ── 设计参数（必须与 web/icon.svg 保持一致）───────────────────────────────
# 所有坐标都在 64×64 的设计画布上，下面统一乘 SS（超采样倍数）再画。
CANVAS = 64.0

BG_TOP = (0x1C, 0x23, 0x30)      # 面板 --bg-soft2
BG_BOTTOM = (0x0D, 0x11, 0x17)   # 面板 --bg
BORDER = (0x2B, 0x34, 0x41)      # 面板 --border
INK_TOP = (0x79, 0xB8, 0xFF)     # Q 的渐变起色
INK_BOTTOM = (0x58, 0xA6, 0xFF)  # Q 的渐变终色 / 面板 --accent
GREEN = (0x3F, 0xB9, 0x50)       # 面板 --green（「运行中」指示灯）

RECT_XY = (2.5, 2.5, 61.5, 61.5)
RECT_RADIUS = 13.5
BORDER_W = 1.5

RING_CX, RING_CY, RING_R = 31.0, 28.0, 12.8
RING_W = 6.4
TAIL_FROM = (40.9, 37.9)
TAIL_TO = (49.0, 48.0)

DOT_CX, DOT_CY, DOT_HALO_R, DOT_R = 50.5, 13.5, 7.2, 4.6

# 超采样倍数：先在 N*SS 的分辨率上画、最后 LANCZOS 缩回去。
# 圆环这种细线条如果不超采样，16px 下边缘会明显锯齿。
SS = 8
N = int(CANVAS)

ICO_SIZES = (16, 32, 48, 64)
PNG_SIZES = (192, 512)          # Android / PWA
TOUCH_SIZE = 180                # apple-touch-icon


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _vertical_gradient(size: int, top: tuple[int, int, int],
                       bottom: tuple[int, int, int]) -> Image.Image:
    """竖直渐变。逐行画横线，比逐像素快得多。"""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / max(1, size - 1)
        draw.line([(0, y), (size, y)], fill=_lerp(top, bottom, t))
    return img


def _diagonal_gradient(c0: tuple[int, int, int], c1: tuple[int, int, int],
                       steps: int = 64) -> Image.Image:
    """45° 线性渐变（左上 → 右下），对应 SVG 里的 x1=0,y1=0 → x2=1,y2=1。

    只用 steps×steps 算，用的时候再放大：渐变本身是连续的，放大不会掉质量，
    而在 512×512 上逐像素跑 Python 循环反而慢得多（且毫无意义）。
    """
    img = Image.new("RGB", (steps, steps))
    d = ImageDraw.Draw(img)
    span = max(1, steps - 1)
    for y in range(steps):
        for x in range(steps):
            t = (x / span + y / span) / 2.0
            d.point((x, y), fill=_lerp(c0, c1, t))
    return img


def _ink_mask(size: int) -> Image.Image:
    """Q 的形状遮罩（环 + 尾巴），纯白色 L 图。

    ★ 这里刻意把「形状」和「颜色」拆开做：
      之前想用一个「渐变色圆盘」直接当环，结果圆盘圆心为了造渐变而向下偏移，
      覆盖不到环的顶部 —— 画出来的 Q 是缺口的（顶部细、底部粗）。
      正确做法：先用几何形状得到精确的遮罩，再把渐变透过遮罩叠上去，
      形状和颜色互不干扰。
    """
    s = size / CANVAS

    def px(v: float) -> float:
        return v * s

    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)

    # 环 = 大实心圆 − 小实心圆
    cx, cy, r, w = px(RING_CX), px(RING_CY), px(RING_R), px(RING_W)
    outer_r = r + w / 2
    inner_r = max(0.0, r - w / 2)
    d.ellipse([cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r], fill=255)
    d.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r], fill=0)

    # 尾巴：粗线 + 两端补圆（PIL 的 line 只有平头，没有 round cap）。
    # 必须画在「挖完环孔」之后 —— 尾巴起点落在环带里，顺序反了会被挖掉一块。
    hw = w / 2
    fr = (px(TAIL_FROM[0]), px(TAIL_FROM[1]))
    to = (px(TAIL_TO[0]), px(TAIL_TO[1]))
    d.line([fr, to], fill=255, width=max(1, round(w)))
    for p in (fr, to):
        d.ellipse([p[0] - hw, p[1] - hw, p[0] + hw, p[1] + hw], fill=255)

    return mask


def draw_icon(size: int = N * SS) -> Image.Image:
    """画一张 size×size 的图标（RGBA）。"""
    s = size / CANVAS          # 设计坐标 → 像素坐标的比例

    def px(v: float) -> float:
        return v * s

    # 1) 渐变背景 + 圆角遮罩
    grad = _vertical_gradient(size, BG_TOP, BG_BOTTOM)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [px(RECT_XY[0]), px(RECT_XY[1]), px(RECT_XY[2]), px(RECT_XY[3])],
        radius=px(RECT_RADIUS), fill=255,
    )
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    img.paste(grad, (0, 0), mask)

    draw = ImageDraw.Draw(img)

    # 2) 边框（画在圆角矩形轮廓上，正好贴合）
    draw.rounded_rectangle(
        [px(RECT_XY[0]), px(RECT_XY[1]), px(RECT_XY[2]), px(RECT_XY[3])],
        radius=px(RECT_RADIUS), outline=BORDER, width=max(1, round(px(BORDER_W))),
    )

    # 3) Q：45° 渐变透过形状遮罩贴上去
    ink = _diagonal_gradient(INK_TOP, INK_BOTTOM).resize((size, size), Image.BICUBIC)
    img.paste(ink, (0, 0), _ink_mask(size))

    # 4) 右上角状态点：先垫一圈背景色（避免贴到边框上显脏），再画绿点
    dcx, dcy = px(DOT_CX), px(DOT_CY)
    for rr, col in ((px(DOT_HALO_R), BG_BOTTOM), (px(DOT_R), GREEN)):
        draw.ellipse([dcx - rr, dcy - rr, dcx + rr, dcy + rr], fill=col)

    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="生成控制面板图标")
    ap.add_argument("--preview", action="store_true", help="额外输出 _icon_preview.png（放大拼图，便于肉眼看）")
    args = ap.parse_args()

    WEB_DIR.mkdir(parents=True, exist_ok=True)

    # 一次超采样绘制，之后全部从它 LANCZOS 缩小 —— 只画一次，省时间也不会有版本差异
    master = draw_icon(N * SS)
    written: list[tuple[str, tuple[int, int], int]] = []

    ico = WEB_DIR / "favicon.ico"
    # 给 ICO 的母图用 64×64：PIL 内部会从这张缩到 16/32/48，
    # 从 512 直接缩到 16 会糊掉细节（细环）。
    master.resize((64, 64), Image.LANCZOS).save(
        ico, format="ICO", sizes=[(s, s) for s in ICO_SIZES]
    )
    written.append((ico.name, (64, 64), ico.stat().st_size))

    for s in PNG_SIZES:
        p = WEB_DIR / f"icon-{s}.png"
        master.resize((s, s), Image.LANCZOS).save(p, format="PNG", optimize=True)
        written.append((p.name, (s, s), p.stat().st_size))

    touch = WEB_DIR / "apple-touch-icon.png"
    # iOS 不接受透明背景（会自己填黑），所以 apple-touch 铺一层实底
    flat = Image.new("RGB", (TOUCH_SIZE, TOUCH_SIZE), BG_BOTTOM)
    flat.paste(master.resize((TOUCH_SIZE, TOUCH_SIZE), Image.LANCZOS), (0, 0),
               master.resize((TOUCH_SIZE, TOUCH_SIZE), Image.LANCZOS))
    flat.save(touch, format="PNG", optimize=True)
    written.append((touch.name, (TOUCH_SIZE, TOUCH_SIZE), touch.stat().st_size))

    print("=" * 58)
    for name, size, nbytes in written:
        print(f"  {name:<22} {size[0]:>4}×{size[1]:<4} {nbytes / 1024:>7.1f} KB")
    print("=" * 58)

    if args.preview:
        # 拼一张预览：真实尺寸 + 放大到 256 看细节
        shots = [master.resize((s, s), Image.LANCZOS) for s in (16, 32, 64, 128)]
        big = master.resize((256, 256), Image.LANCZOS)
        pad, gap = 16, 14
        w = pad * 2 + sum(i.width for i in shots) + gap * (len(shots) - 1) + gap * 2 + big.width
        h = pad * 2 + max(big.height, 128)
        canvas = Image.new("RGB", (w, h), (0x0D, 0x11, 0x17))
        x = pad
        for im in shots:
            canvas.paste(im, (x, pad + (128 - im.height) // 2), im)
            x += im.width + gap
        x += gap
        canvas.paste(big, (x, pad), big)
        prev = WEB_DIR / "_icon_preview.png"
        canvas.save(prev)
        print(f"预览图：{prev}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
