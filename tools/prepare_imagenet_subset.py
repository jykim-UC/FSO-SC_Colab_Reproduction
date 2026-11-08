"""Select original ImageNet JPEGs from nested TARs and package them for Colab.

Run manually. This script never downloads or changes the source archive.
The class-balanced selection and seed are implementation choices, not settings
disclosed by the FSO-SC paper. Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile


SOURCE_BYTES = 147_897_477_120
SOURCE_MD5 = "1d675b47d978889d74fa0da5fadfb00e"


def prepare_subset(source, output, per_class=10, seed=42,
                   expected_classes=1000, required_source_bytes=SOURCE_BYTES):
    source, output = Path(source), Path(output)
    partial = output.with_name(output.name + ".partial")
    if not source.is_file():
        raise FileNotFoundError(f"Source archive not found: {source}")
    if required_source_bytes is not None and source.stat().st_size != required_source_bytes:
        raise ValueError("Source size differs from the verified ILSVRC2012 train archive.")
    if per_class < 1:
        raise ValueError("per_class must be positive")
    if source.resolve() in (output.resolve(), partial.resolve()):
        raise ValueError("The output must not replace the source archive.")
    if output.exists() or partial.exists():
        raise FileExistsError(
            f"Output or partial file already exists: {output}\n"
            "Nothing was overwritten. For a new attempt use --output with a new ZIP name."
        )

    with tarfile.open(source, mode="r:") as outer:
        class_members = {}
        for member in outer.getmembers():
            if not member.isfile():
                continue
            name = PurePosixPath(member.name).name
            if not re.fullmatch(r"n[0-9]{8}\.tar", name):
                continue
            class_id = name[:-4]
            if class_id in class_members:
                raise ValueError(f"Duplicate class archive: {class_id}")
            class_members[class_id] = member
        if len(class_members) != expected_classes:
            raise ValueError(
                f"Expected {expected_classes} class TARs, found {len(class_members)}"
            )

        required_scratch = max(member.size for member in class_members.values())
        if shutil.disk_usage(tempfile.gettempdir()).free < required_scratch + 128 * 1024**2:
            raise OSError("Insufficient local temporary space for one class TAR.")
        # Preserve source order so the NAS is read sequentially, one class at a time.
        ordered_members = sorted(class_members.items(), key=lambda item: item[1].offset)
        image_count = expected_classes * per_class
        print(f"Source: {source}", flush=True)
        print(f"Output: {output}", flush=True)
        print(f"Selection: {expected_classes} classes x {per_class} = {image_count} images", flush=True)
        print(f"Seed {seed}; class-balanced sampling is NOT specified by the paper.", flush=True)
        print("The source is read once. Only ONE class TAR uses local temporary space.", flush=True)
        print("Keep this window open until COMPLETE. No source file will be changed.", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        records = []
        started = time.monotonic()
        with zipfile.ZipFile(partial, mode="x", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as destination:
            for number, (class_id, member) in enumerate(ordered_members, start=1):
                raw_class = outer.extractfile(member)
                if raw_class is None:
                    raise ValueError(f"Cannot read class TAR: {class_id}")
                with raw_class, tempfile.TemporaryFile() as local_class:
                    shutil.copyfileobj(raw_class, local_class, length=4 * 1024**2)
                    if local_class.tell() != member.size:
                        raise ValueError(f"Truncated class TAR: {class_id}")
                    local_class.seek(0)
                    with tarfile.open(fileobj=local_class, mode="r:") as nested:
                        candidates = {}
                        for image_member in nested.getmembers():
                            if not image_member.isfile():
                                continue
                            basename = PurePosixPath(image_member.name).name
                            if PurePosixPath(basename).suffix.lower() not in {".jpg", ".jpeg"}:
                                continue
                            if not re.fullmatch(class_id + r"_[0-9]+\.(?:JPEG|JPG|jpeg|jpg)", basename):
                                raise ValueError(f"Unexpected ImageNet filename: {basename}")
                            if basename in candidates:
                                raise ValueError(f"Duplicate JPEG in {class_id}: {basename}")
                            candidates[basename] = image_member
                        if len(candidates) < per_class:
                            raise ValueError(f"Not enough JPEGs in {class_id}")
                        # Per-class RNG makes the sample independent of TAR member ordering.
                        rng = random.Random(f"{seed}:{class_id}")
                        selected = sorted(rng.sample(sorted(candidates), per_class))
                        for basename in selected:
                            image_member = candidates[basename]
                            image_stream = nested.extractfile(image_member)
                            if image_stream is None:
                                raise ValueError(f"Cannot read JPEG: {basename}")
                            with image_stream:
                                original_bytes = image_stream.read()
                            if len(original_bytes) != image_member.size:
                                raise ValueError(f"Truncated JPEG: {basename}")
                            zip_path = f"ILSVRC2012/{class_id}/{basename}"
                            destination.writestr(zip_path, original_bytes)
                            records.append({
                                "path": zip_path,
                                "class_id": class_id,
                                "source_class_tar": member.name,
                                "source_member": image_member.name,
                                "bytes": len(original_bytes),
                                "sha256": hashlib.sha256(original_bytes).hexdigest(),
                            })
                elapsed = time.monotonic() - started
                remaining = elapsed / number * (expected_classes - number)
                print(
                    f"[{number:4d}/{expected_classes}] {100 * number / expected_classes:6.2f}%"
                    f" | {len(records):5d}/{image_count} images"
                    f" | elapsed {elapsed / 60:.1f} min | ETA {remaining / 60:.1f} min",
                    flush=True,
                )

            if len(records) != image_count:
                raise ValueError("Final image count mismatch")
            metadata = {
                "dataset": "ILSVRC2012 training subset for FSO-SC reproduction",
                "source_filename": source.name,
                "source_bytes": source.stat().st_size,
                "official_source_md5": SOURCE_MD5,
                "source_md5_recomputed_by_this_script": False,
                "selection": "class-balanced random sample without replacement",
                "classes": expected_classes,
                "images_per_class": per_class,
                "images": image_count,
                "seed": seed,
                "selection_specified_by_paper": False,
                "preprocessing": "none; original image bytes preserved",
                "colab_extract_directory": "/content/dataset",
                "colab_training_directory": "/content/dataset/ILSVRC2012",
            }
            destination.writestr("manifests/imagenet_selection.json", json.dumps(metadata, indent=2))
            destination.writestr(
                "manifests/imagenet_files.jsonl",
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            )

    print("Verifying ZIP CRC and image count...", flush=True)
    with zipfile.ZipFile(partial) as verification:
        image_paths = [entry.filename for entry in verification.infolist()
                       if entry.filename.startswith("ILSVRC2012/")
                       and PurePosixPath(entry.filename).suffix.lower() in {".jpg", ".jpeg"}]
        if len(image_paths) != image_count or len(set(image_paths)) != image_count:
            raise ValueError("ZIP image count/uniqueness verification failed")
        bad_file = verification.testzip()
        if bad_file is not None:
            raise ValueError(f"ZIP CRC verification failed: {bad_file}")
    if output.exists():
        raise FileExistsError(f"A completed output already exists: {output}")
    partial.rename(output)
    print(f"COMPLETE: {output}", flush=True)
    print(f"Images: {image_count}; ZIP size: {output.stat().st_size / 1024**3:.2f} GiB", flush=True)
    print("Upload this ZIP only to Google Drive. Keep the source TAR on your NAS.", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        prepare_subset(args.source, args.output, seed=args.seed)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        print("The source TAR is unchanged. An incomplete output may remain with .partial suffix.",
              file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("\nStopped by user. The source TAR is unchanged.", file=sys.stderr, flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
