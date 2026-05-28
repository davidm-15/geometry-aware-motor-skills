"""
Render all OBJ meshes in a dataset directory to cropped PNG images.

Auto-discovers samples: any subdirectory of --samples-dir that contains
an OBJ file whose name matches the subdirectory name.

Usage:
    python -m scripts.batch_render --samples-dir datasets/L_shape
    python -m scripts.batch_render --samples-dir datasets/window_cross --out-dir outputs/renders/window_cross
    python -m scripts.batch_render --samples-dir datasets/I_shape --limit 50
"""

import argparse
from pathlib import Path

import pyvista as pv
from PIL import Image


def render_and_crop(obj_path: Path, out_path: Path) -> None:
    mesh = pv.read(str(obj_path)).triangulate()

    plotter = pv.Plotter(off_screen=True, window_size=[1500, 1500])
    plotter.add_mesh(mesh, color="lightgrey", smooth_shading=False)
    plotter.set_background("white")
    plotter.camera_position = "iso"

    img = Image.fromarray(plotter.screenshot(return_img=True))
    plotter.close()

    mask = img.convert("L").point(lambda p: 0 if p == 255 else 255)
    bbox = mask.getbbox()
    if bbox:
        img = img.crop(bbox)

    img.save(str(out_path))


def discover_samples(samples_dir: Path) -> list[tuple[str, Path]]:
    """Return (name, obj_path) for every subdirectory that contains a matching OBJ."""
    found = []
    for subdir in sorted(samples_dir.iterdir()):
        if not subdir.is_dir():
            continue
        obj_path = subdir / f"{subdir.name}.obj"
        if obj_path.exists():
            found.append((subdir.name, obj_path))
    return found


def main(samples_dir: Path, out_dir: Path, limit: int | None) -> None:
    samples = discover_samples(samples_dir)
    if not samples:
        print(f"No samples found in {samples_dir}")
        return

    if limit is not None:
        samples = samples[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(samples)} samples → {out_dir}/")

    ok = failed = skipped = 0
    for name, obj_path in samples:
        out_path = out_dir / f"{name}.png"
        if out_path.exists():
            skipped += 1
            continue
        try:
            render_and_crop(obj_path, out_path)
            ok += 1
            if ok % 25 == 0:
                print(f"  {ok + skipped}/{len(samples)}")
        except Exception as e:
            print(f"  [!] {name}: {e}")
            failed += 1

    print(f"Done — {ok} rendered, {skipped} skipped (already exist), {failed} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-render dataset OBJ meshes to PNG.")
    parser.add_argument(
        "--samples-dir", type=Path, default=Path("datasets/L_shape"),
        help="Dataset directory containing sample subdirectories (default: datasets/L_shape)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for PNGs (default: <samples-dir>/00_renders)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Render only the first N samples (default: all)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.samples_dir / "00_renders"
    main(args.samples_dir, out_dir, args.limit)
