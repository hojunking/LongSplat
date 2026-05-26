#!/usr/bin/env python3
"""Normalize LongSplat data layout with physical moves.

Default mode is a dry run. Add --apply to move directories.

Target layout:
  data/compress-x/{dataset}/{scene}/{variant}
  data/compress-o/{dataset}/{scene}/{codec}/{qp}/{variant}

Notes:
  - QP names stay unpadded: qp22, qp32, ...
  - Codec names are normalized to h264, h265, vvc.
  - 2D refinement outputs are stored as variants, e.g. 2d_refine_codiff.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


FREE_SCENES = ("grass", "hydrant", "lab", "pillar", "road", "sky", "stair")
TNT_SCENES = (
    "Ballroom",
    "Barn",
    "Church",
    "Family",
    "Francis",
    "Horse",
    "Ignatius",
    "Museum",
)
HIKE_SCENES = (
    "forest1",
    "forest2",
    "forest3",
    "garden1",
    "garden2",
    "garden3",
    "indoor",
    "playground",
    "university1",
    "university2",
    "university3",
    "university4",
)
REALWORLD_SCENES = ("IMG_0405", "IMG_0406")


@dataclass(frozen=True)
class Move:
    src: Path
    dst: Path
    dataset: str
    scene: str
    kind: str
    codec: str = ""
    qp: str = ""
    variant: str = "full"


def qp_name(name: str) -> str:
    """Return unpadded qp name from qp-like directory names."""
    base = name.lower()
    if not base.startswith("qp"):
        raise ValueError(f"not a qp name: {name}")
    digits = ""
    for ch in base[2:]:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        raise ValueError(f"qp has no number: {name}")
    return f"qp{int(digits)}"


def add_if_exists(moves: list[Move], move: Move) -> None:
    if move.src.exists():
        moves.append(move)


def build_moves(root: Path) -> list[Move]:
    moves: list[Move] = []
    cx = root / "data" / "compress-x"
    co = root / "data" / "compress-o"

    # Original data. Keep compress-x, normalize dataset names and put profile at
    # scene level only where the current tree already encodes it.
    for scene in HIKE_SCENES:
        add_if_exists(
            moves,
            Move(
                cx / "hike_half" / scene,
                cx / "hike" / scene / "half",
                "hike",
                scene,
                "original",
                variant="half",
            ),
        )
    for scene in REALWORLD_SCENES:
        add_if_exists(
            moves,
            Move(
                cx / "realworld" / f"{scene}_frames",
                cx / "realworld" / scene / "default",
                "realworld",
                scene,
                "original",
                variant="default",
            ),
        )
        add_if_exists(
            moves,
            Move(
                cx / "realworld" / f"{scene}_frames_full",
                cx / "realworld" / scene / "full",
                "realworld",
                scene,
                "original",
                variant="full",
            ),
        )

    # Existing codec-less qp outputs are treated as h265. JM is treated as h264.
    for dataset, scenes, qps in (
        ("free", FREE_SCENES, ("qp22", "qp27", "qp32", "qp37", "qp42", "qp47")),
        ("tnt", TNT_SCENES, ("qp27", "qp32", "qp37", "qp42", "qp47")),
    ):
        for qp in qps:
            for scene in scenes:
                add_if_exists(
                    moves,
                    Move(
                        co / dataset / qp / scene,
                        co / dataset / scene / "h265" / qp_name(qp) / "full",
                        dataset,
                        scene,
                        "compressed",
                        "h265",
                        qp_name(qp),
                        "full",
                    ),
                )
        for qp in ("qp22", "qp27", "qp32", "qp37", "qp42", "qp47"):
            for scene in scenes:
                add_if_exists(
                    moves,
                    Move(
                        co / dataset / "jm" / qp / scene,
                        co / dataset / scene / "h264" / qp_name(qp) / "full",
                        dataset,
                        scene,
                        "compressed",
                        "h264",
                        qp_name(qp),
                        "full",
                    ),
                )

    # Free has a second JM export tree. It contains codec artifacts
    # (frames/bitstream/yuv/log/csv), not the training-ready images/sparse tree.
    for qp in ("qp22", "qp27", "qp32", "qp37", "qp42", "qp47"):
        for scene in FREE_SCENES:
            add_if_exists(
                moves,
                Move(
                    co / "jm_out_free" / f"jm_{qp}" / scene,
                    co / "free" / scene / "h264" / qp_name(qp) / "codec_artifacts_jm",
                    "free",
                    scene,
                    "artifact",
                    "h264",
                    qp_name(qp),
                    "codec_artifacts_jm",
                ),
            )

    # VVC outputs currently live under vtm_output_free_qp27_47/jpg_fixed.
    for qp in ("qp27", "qp47"):
        for scene in FREE_SCENES:
            add_if_exists(
                moves,
                Move(
                    co / "free" / "vtm_output_free_qp27_47" / "jpg_fixed" / "free_dataset" / qp / scene,
                    co / "free" / scene / "vvc" / qp_name(qp) / "full",
                    "free",
                    scene,
                    "compressed",
                    "vvc",
                    qp_name(qp),
                    "full",
                ),
            )

    # 2D refinement outputs are variants under the normalized codec/qp path.
    for scene in FREE_SCENES:
        add_if_exists(
            moves,
            Move(
                co / "free" / "codiff" / "qp37" / scene,
                co / "free" / scene / "h265" / "qp37" / "2d_refine_codiff",
                "free",
                scene,
                "compressed",
                "h265",
                "qp37",
                "2d_refine_codiff",
            ),
        )
    for scene in TNT_SCENES:
        add_if_exists(
            moves,
            Move(
                co / "tnt" / "codiff" / "qp37" / scene,
                co / "tnt" / scene / "h265" / "qp37" / "2d_refine_codiff",
                "tnt",
                scene,
                "compressed",
                "h265",
                "qp37",
                "2d_refine_codiff",
            ),
        )
        add_if_exists(
            moves,
            Move(
                co / "tnt" / "pisaSR" / "qp37" / scene,
                co / "tnt" / scene / "h265" / "qp37" / "2d_refine_pisar",
                "tnt",
                scene,
                "compressed",
                "h265",
                "qp37",
                "2d_refine_pisar",
            ),
        )

    # Hike: codec-less tree is h265; *_jm and *_vvc carry codec in directory name.
    for qp_dir in ("qp32", "qp37", "qp32_half", "qp37_half"):
        qp = qp_name(qp_dir)
        variant = "half" if qp_dir.endswith("_half") else "full"
        for scene in HIKE_SCENES:
            add_if_exists(
                moves,
                Move(
                    co / "hike" / qp_dir / scene,
                    co / "hike" / scene / "h265" / qp / variant,
                    "hike",
                    scene,
                    "compressed",
                    "h265",
                    qp,
                    variant,
                ),
            )
    for codec_dir, codec in (("hike_half_jm", "h264"), ("hike_half_vvc", "vvc")):
        for qp_dir in ("qp27", "qp37", "qp47"):
            qp = qp_name(qp_dir)
            for scene in HIKE_SCENES:
                add_if_exists(
                    moves,
                    Move(
                        co / codec_dir / qp_dir / scene,
                        co / "hike" / scene / codec / qp / "half",
                        "hike",
                        scene,
                        "compressed",
                        codec,
                        qp,
                        "half",
                    ),
                )

    # Realworld compressed tree stores QP as an extra uppercase directory.
    for scene in REALWORLD_SCENES:
        for qp in ("QP32", "QP37"):
            add_if_exists(
                moves,
                Move(
                    co / "realworld" / "qp37" / scene / qp,
                    co / "realworld" / scene / "h265" / qp_name(qp) / "full",
                    "realworld",
                    scene,
                    "compressed",
                    "h265",
                    qp_name(qp),
                    "full",
                ),
            )

    return moves


def validate_moves(moves: list[Move]) -> list[str]:
    errors: list[str] = []
    seen_dst: dict[Path, Path] = {}
    for move in moves:
        if move.dst in seen_dst:
            errors.append(f"duplicate target: {move.dst} from {seen_dst[move.dst]} and {move.src}")
        seen_dst[move.dst] = move.src
        if move.dst.exists():
            errors.append(f"target already exists: {move.dst} (source: {move.src})")
        try:
            move.dst.relative_to(move.src)
        except ValueError:
            pass
        else:
            errors.append(f"target is inside source: {move.src} -> {move.dst}")
    return errors


def write_manifest(path: Path, moves: list[Move]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("dataset", "scene", "kind", "codec", "qp", "variant", "src", "dst"))
        for move in moves:
            writer.writerow(
                (
                    move.dataset,
                    move.scene,
                    move.kind,
                    move.codec,
                    move.qp,
                    move.variant,
                    move.src,
                    move.dst,
                )
            )


def prune_empty_dirs(root: Path) -> None:
    for current, dirs, files in os.walk(root, topdown=False):
        path = Path(current)
        if path == root:
            continue
        if not dirs and not files:
            path.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", choices=("free", "tnt", "hike", "realworld"), help="limit moves to one dataset")
    parser.add_argument("--tree", choices=("compress-o", "compress-x", "all"), default="compress-o", help="limit moves to one data tree; default: compress-o")
    parser.add_argument("--apply", action="store_true", help="perform physical moves")
    parser.add_argument("--prune-empty", action="store_true", help="remove empty legacy directories after apply")
    parser.add_argument("--manifest", type=Path, default=Path("comp_log/data_layout_manifest.csv"))
    args = parser.parse_args()

    root = args.root.resolve()
    moves = build_moves(root)
    if args.tree != "all":
        tree_root = root / "data" / args.tree
        moves = [move for move in moves if move.src.is_relative_to(tree_root)]
    if args.dataset:
        moves = [move for move in moves if move.dataset == args.dataset]
    errors = validate_moves(moves)
    write_manifest(root / args.manifest, moves)

    print(f"planned moves: {len(moves)}")
    print(f"manifest: {root / args.manifest}")
    if errors:
        print("errors:")
        for error in errors:
            print(f"  - {error}")
        return 2

    for move in moves:
        rel_src = move.src.relative_to(root)
        rel_dst = move.dst.relative_to(root)
        print(f"{rel_src} -> {rel_dst}")
        if args.apply:
            move.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(move.src), str(move.dst))

    if args.apply and args.prune_empty:
        prune_empty_dirs(root / "data" / "compress-o")
        prune_empty_dirs(root / "data" / "compress-x")

    if not args.apply:
        print("dry-run only; add --apply to move directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
