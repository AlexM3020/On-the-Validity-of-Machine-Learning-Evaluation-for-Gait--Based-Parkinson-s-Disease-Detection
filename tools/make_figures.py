"""Draw the documented architecture and copy original loss pixels without metadata.

This utility generates no experiment values. Optional loss-image arguments must
point to the original raster exports; curves are not smoothed or redigitized.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageChops


INK = '#17334A'
TEAL = '#006F73'
PALE = '#E8F4F2'
LIGHT = '#F1F4F7'
GRAY = '#526674'


def copy_pixels(source, destination):
    """Preserve decoded pixels while omitting PNG text/EXIF metadata."""
    with Image.open(source) as image:
        original = image.convert('RGBA')
        clean = Image.frombytes('RGBA', original.size, original.tobytes())
        clean.save(destination, format='PNG')
    with Image.open(destination) as result:
        if ImageChops.difference(original, result.convert('RGBA')).getbbox() is not None:
            raise RuntimeError('pixel-preservation check failed')


def architecture(destination):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7.4, 2.85))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 7.4), ylim=(0, 2.85))
    ax.axis('off')
    ax.text(.16, 2.60, 'COMPACT SELECTIVE STATE-SPACE CLASSIFIER',
            color=INK, fontsize=11, weight='bold', va='center')
    ax.text(.16, 2.33, '2 blocks  |  model width 128  |  state size 32  |  275,330 parameters',
            color=GRAY, fontsize=8.5, va='center')

    boxes = [
        (.14, .88, 'Input', '18 channels\nChannel-wise\nz-score', True),
        (1.22, 1.10, 'Embedding', 'Linear 18 → 128\nLayerNorm', False),
        (2.53, 2.05, 'Mamba stack × 2', 'Width 128 · inner width 256\nState 32 · local conv 32\nTime-step rank 8 · LayerNorm', False),
        (4.80, 1.10, 'Time pooling', 'Temporal maximum\n128 features', True),
        (6.12, 1.13, 'Prediction', 'Linear 128 → 2\nControl / PD', False),
    ]
    y, h = 1.00, 1.02
    for x, w, heading, body, documented in boxes:
        patch = FancyBboxPatch((x, y), w, h,
                              boxstyle='round,pad=0.015,rounding_size=0.06',
                              facecolor=PALE if 'stack' in heading else LIGHT,
                              edgecolor=TEAL if 'stack' in heading else INK,
                              linewidth=1.1, linestyle=(0, (3, 2)) if documented else '-')
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - .21, heading, ha='center', va='center',
                fontsize=8.4, color=INK, weight='bold')
        ax.text(x + w / 2, y + .37, body, ha='center', va='center',
                fontsize=7.1, color=INK, linespacing=1.5)
    # Draw connectors after panels, so arrowheads cannot be hidden by backgrounds.
    for first, second in zip(boxes, boxes[1:]):
        right = first[0] + first[1]
        left = second[0]
        assert left > right
        ax.add_patch(FancyArrowPatch((right + .035, y + h / 2),
                                    (left - .035, y + h / 2), arrowstyle='-|>',
                                    mutation_scale=10, linewidth=1.1, color=INK))

    ax.plot([.17, .50], [.67, .67], color=INK, linewidth=1.2)
    ax.text(.57, .67, 'Checkpoint-supported modules', fontsize=7.7, color=GRAY, va='center')
    ax.plot([3.62, 3.95], [.67, .67], color=INK, linewidth=1.2, linestyle=(0, (3, 2)))
    ax.text(4.02, .67, 'Documented processing', fontsize=7.7, color=GRAY, va='center')
    ax.text(.16, .35, 'Design schematic, not an exact reconstruction of the historical forward pass.',
            fontsize=7.7, color=GRAY, va='center')
    ax.text(.16, .13, 'Window length: 2,000 in the experimental description; 1,000 in the separate run note.',
            fontsize=7.4, color=GRAY, va='center')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bounds = fig.bbox
    for text in ax.texts:
        extent = text.get_window_extent(renderer)
        if not (figure_bounds.x0 <= extent.x0 and extent.x1 <= figure_bounds.x1
                and figure_bounds.y0 <= extent.y0 and extent.y1 <= figure_bounds.y1):
            raise RuntimeError(f'text extends beyond figure: {text.get_text()}')
    fig.savefig(destination, metadata={'Title': 'Documented Mamba classifier',
                                      'Author': '', 'Creator': 'Anonymous research artifact',
                                      'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'paper' / 'figures')
    parser.add_argument('--mamba-loss', type=Path)
    parser.add_argument('--lstm-loss', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    architecture(args.output / 'architecture.pdf')
    for source, name in [(args.mamba_loss, 'mamba_loss.png'),
                         (args.lstm_loss, 'lstm_loss.png')]:
        if source is not None:
            copy_pixels(source, args.output / name)
    print('Architecture generated; supplied loss-image pixels verified unchanged.')


if __name__ == '__main__':
    main()