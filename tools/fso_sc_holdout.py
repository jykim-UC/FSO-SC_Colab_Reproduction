"""Result-blind, exact-file-disjoint final holdout selection and verification.

This module reads identities only: no images are decoded, evaluated or written.
Exact SHA-256 separation does not establish geographic or near-image independence.
Freeze the returned manifest once; evaluation must verify it, never select anew.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


SELECTION_VERSION = "fso_sc_holdout_sha256_rank_v1"
MANIFEST_VERSION = "fso_sc_final_holdout_manifest_v1"
DEFAULT_SEED = 20261029
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def _require(ok, message):
    if not ok:
        raise ValueError("Holdout verification failed: " + message)


def canonical_sha256(value):
    """Hash canonical UTF-8 JSON; this is not the hash of pretty-printed JSON."""
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def _names(values, label):
    _require(not isinstance(values, (str, bytes)), f"{label} must be a sequence")
    try:
        result = list(values)
    except TypeError as error:
        raise ValueError(f"Holdout verification failed: invalid {label}") from error
    _require(all(isinstance(name, str) and name for name in result), f"invalid {label}")
    _require(len(result) == len(set(result)), f"duplicate {label}")
    return result


def _inventory(hash_by_name, label):
    _require(isinstance(hash_by_name, dict) and hash_by_name, f"empty/invalid {label}")
    names = _names(hash_by_name, label + " names")
    _require(all(_sha(hash_by_name[name]) for name in names), f"invalid SHA-256 in {label}")
    _require(len(set(hash_by_name.values())) == len(names), f"duplicate exact files in {label}")
    return sorted(names)


def zip_image_inventory(path, expected_images=None):
    """Return all PNG/JPEG member file hashes, rejecting duplicate names/content.

    Streams uncompressed member bytes without extraction or raster decoding.
    Nonimage metadata (for example a collection manifest) is not selected.
    """
    result = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or PurePosixPath(info.filename).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            name = info.filename
            parts = name.split("/")
            _require("\\" not in name and not name.startswith("/")
                     and all(part not in {"", ".", ".."} for part in parts)
                     and ":" not in parts[0], f"unsafe image member name: {name!r}")
            _require(name not in result, f"duplicate image member name: {name!r}")
            digest = hashlib.sha256()
            with archive.open(info) as source:
                for chunk in iter(lambda: source.read(8 * 1024**2), b""):
                    digest.update(chunk)
            result[name] = digest.hexdigest()
    names = _inventory(result, "ZIP image inventory")
    if expected_images is not None:
        _require(type(expected_images) is int and expected_images > 0,
                 "expected_images must be a positive integer")
        _require(len(names) == expected_images, "ZIP image count differs from frozen expectation")
    return {name: result[name] for name in names}


def select_holdout_names(names, development_names, development_file_hashes,
                         hash_by_name, count=200, seed=DEFAULT_SEED):
    """Return sorted, result-blind names excluding prior names AND exact files.

    The old development list and its aligned hashes must match this same pool.
    All pool files must be distinct: a renamed duplicate fails closed rather
    than silently altering the population or enabling a partition-leakage claim.
    Content-hash ranking fixes selection across Python versions, input order,
    and file renaming. No filename/coordinate is needed in a public manifest.
    """
    names = _names(names, "pool names")
    pool_names = _inventory(hash_by_name, "test pool")
    _require(set(names) == set(pool_names), "pool names/hash coverage differs")
    development_names = _names(development_names, "development names")
    _require(development_names, "development list must not be empty")
    development_file_hashes = list(development_file_hashes)
    _require(len(development_names) == len(development_file_hashes)
             and all(_sha(value) for value in development_file_hashes),
             "development names/hash coverage differs")
    _require(len(set(development_file_hashes)) == len(development_file_hashes),
             "duplicate exact files in development list")
    for name, digest in zip(development_names, development_file_hashes):
        _require(name in hash_by_name and hash_by_name[name] == digest,
                 f"development identity differs: {name!r}")
    _require(type(count) is int and count > 0, "count must be a positive integer")
    _require(type(seed) is int, "seed must be an integer")
    old_names, old_hashes = set(development_names), set(development_file_hashes)
    eligible = [name for name in pool_names if name not in old_names
                and hash_by_name[name] not in old_hashes]
    _require(len(eligible) >= count, "insufficient exact-file-disjoint candidates")
    ranked = sorted(eligible, key=lambda name: (
        hashlib.sha256(f"{seed}\0{hash_by_name[name]}".encode("ascii")).hexdigest(),
        hash_by_name[name]))
    return sorted(ranked[:count])


def build_holdout_manifest(hash_by_name, development_names, development_file_hashes, *,
                           test_zip_sha256, previous_summary_sha256, count=200,
                           seed=DEFAULT_SEED, expected_total=1000,
                           training_hash_by_name=None, training_zip_sha256=None):
    """Construct metadata to freeze once before viewing any new holdout result.

    An optional training inventory provides a pinned, local exact-file audit;
    without it the manifest explicitly says ``not_verified``. The training ZIP
    is not needed again at evaluation time. This audit says nothing about nearby
    patches or the provenance of a checkpoint's actual training inputs.
    """
    all_names = _inventory(hash_by_name, "test pool")
    _require(type(expected_total) is int and expected_total > 0
             and len(all_names) == expected_total, "test pool size differs from expectation")
    development_names = list(development_names)
    development_file_hashes = list(development_file_hashes)
    selected = select_holdout_names(all_names, development_names, development_file_hashes,
                                   hash_by_name, count=count, seed=seed)
    _require(_sha(test_zip_sha256) and _sha(previous_summary_sha256),
             "missing/invalid input file SHA-256")
    development_hashes = sorted(development_file_hashes)
    training_audit = {"status": "not_verified", "scope": "exact file SHA-256 only"}
    if training_hash_by_name is None:
        _require(training_zip_sha256 is None, "training ZIP hash without training inventory")
    else:
        training_names = _inventory(training_hash_by_name, "training inventory")
        _require(_sha(training_zip_sha256), "missing/invalid training ZIP SHA-256")
        shared_names = set(all_names).intersection(training_names)
        shared_hashes = set(hash_by_name.values()).intersection(training_hash_by_name.values())
        _require(not shared_names and not shared_hashes, "test/training exact identity overlap")
        training_audit.update({
            "status": "verified_no_overlap", "training_images": len(training_names),
            "training_zip_sha256": training_zip_sha256,
            "training_inventory_sha256": canonical_sha256(training_hash_by_name),
            "all_test_name_overlap_count": 0, "all_test_file_sha256_overlap_count": 0,
            "selected_file_sha256_overlap_count": 0,
            "checkpoint_training_provenance": "not_established_by_this_file_audit",
        })
    manifest = {
        "version": MANIFEST_VERSION, "selection_version": SELECTION_VERSION,
        "identifiers": "sha256_file_content",
        "seed": seed, "count": count, "expected_total_images": expected_total,
        "eligible_images": len(all_names) - len(development_names),
        "test_zip_sha256": test_zip_sha256,
        "previous_summary_sha256": previous_summary_sha256,
        "previous_list_sha256": canonical_sha256(development_hashes),
        "all_file_sha256": sorted(hash_by_name.values()),
        "development_file_sha256": development_hashes,
        "selected_file_sha256": sorted(hash_by_name[name] for name in selected),
        "training_exact_file_audit": training_audit,
        "independence_scope": "exact files only; geographic/near-image independence not established",
        "selection_policy": "one frozen result-blind selection; no result-based reselection",
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def verify_holdout_manifest(manifest, hash_by_name, *, test_zip_sha256=None,
                            expected_manifest_sha256=None):
    """Verify frozen membership and complete pool before evaluation; never write.

    Training audit metadata describes the frozen local audit, not a fresh scan.
    Pass an independently pinned manifest hash to detect wholesale substitution.
    """
    _require(isinstance(manifest, dict), "manifest must be an object")
    actual_digest = canonical_sha256({key: value for key, value in manifest.items()
                                      if key != "manifest_sha256"})
    _require(manifest.get("manifest_sha256") == actual_digest, "manifest digest differs")
    if expected_manifest_sha256 is not None:
        _require(_sha(expected_manifest_sha256) and actual_digest == expected_manifest_sha256,
                 "manifest differs from independent pin")
    _require(manifest.get("version") == MANIFEST_VERSION
             and manifest.get("selection_version") == SELECTION_VERSION
             and manifest.get("identifiers") == "sha256_file_content",
             "unsupported frozen manifest/selection version")
    all_names = _inventory(hash_by_name, "test pool")
    _require(type(manifest.get("expected_total_images")) is int
             and len(all_names) == manifest["expected_total_images"], "test pool size differs")
    _require(manifest.get("all_file_sha256") == sorted(hash_by_name.values()),
             "complete frozen pool identity differs")
    _require(_sha(manifest.get("test_zip_sha256"))
             and _sha(manifest.get("previous_summary_sha256")), "invalid provenance file SHA-256")
    if test_zip_sha256 is not None:
        _require(test_zip_sha256 == manifest["test_zip_sha256"], "test ZIP SHA-256 differs")
    development_hashes = manifest.get("development_file_sha256")
    _require(isinstance(development_hashes, list) and development_hashes
             and all(_sha(value) for value in development_hashes)
             and development_hashes == sorted(set(development_hashes)),
             "invalid frozen development hashes")
    _require(manifest.get("previous_list_sha256") == canonical_sha256(development_hashes),
             "previous development-list digest differs")
    name_by_hash = {digest: name for name, digest in hash_by_name.items()}
    _require(all(value in name_by_hash for value in development_hashes),
             "frozen development identities missing from pool")
    development_names = [name_by_hash[value] for value in development_hashes]
    selected = select_holdout_names(all_names, development_names,
        development_hashes, hash_by_name,
        count=manifest.get("count"), seed=manifest.get("seed"))
    _require(manifest.get("eligible_images") == len(all_names) - len(development_hashes),
             "eligible count differs")
    _require(manifest.get("selected_file_sha256") == sorted(hash_by_name[name] for name in selected),
             "frozen holdout membership differs from result-blind rule")
    audit = manifest.get("training_exact_file_audit")
    _require(isinstance(audit, dict) and audit.get("status") in {"not_verified", "verified_no_overlap"}
             and audit.get("scope") == "exact file SHA-256 only", "invalid training audit")
    if audit["status"] == "verified_no_overlap":
        _require(type(audit.get("training_images")) is int and audit["training_images"] > 0
                 and _sha(audit.get("training_zip_sha256"))
                 and _sha(audit.get("training_inventory_sha256")), "invalid training audit provenance")
        _require(all(type(audit.get(key)) is int and audit[key] == 0 for key in (
            "all_test_name_overlap_count", "all_test_file_sha256_overlap_count",
            "selected_file_sha256_overlap_count")), "nonzero/invalid training overlap audit")
    return selected
