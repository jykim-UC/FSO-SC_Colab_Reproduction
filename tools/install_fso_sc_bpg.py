"""Pinned official Bellard libbpg installer for Colab/Linux or local Win64 tests.

No API credentials, import-time work or third-party binary republishing.
Checksums identify the retrieved official distribution, not publisher signatures.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile


BPG_RELEASE = "0.9.8"
BPG_SOURCE_URL = "https://bellard.org/bpg/libbpg-0.9.8.tar.gz"
# Independently matched to FreeBSD graphics/libbpg/distinfo (2029021 bytes).
BPG_SOURCE_SHA256 = "c0788e23bdf1a7d36cb4424ccb2fae4c7789ac94949563c4ad0e2569d3bf0095"
BPG_SOURCE_CHECKSUM_REFERENCE = "https://raw.githubusercontent.com/freebsd/freebsd-ports/main/graphics/libbpg/distinfo"
BPG_WIN64_URL = "https://bellard.org/bpg/bpg-0.9.8-win64.zip"
BPG_WIN64_SHA256 = "44034757c7dbf44498bfdb4c28b8ed3fbc498b32e7d18de4b98a6995371d457c"
BPG_CMAKE_VERSION = "3.28.3"
BPG_BUILD_RECIPE = "libbpg098-cmake3283-v2"


def _validated_cmake_version(output):
    found = re.search(r"cmake version (\d+\.\d+\.\d+)",output)
    if not found or found[1] != BPG_CMAKE_VERSION:
        raise RuntimeError("This verified libbpg build requires cmake==3.28.3; "
                           "rerun the notebook installation cell before codec setup. "
                           "CMake3.31+ changes the legacy YASM linker modules.")
    return found[1]


def _get_verified_archive(url,digest,path):
    if re.fullmatch(r"[0-9a-f]{64}",digest) is None:
        raise ValueError("Expected a pinned lowercase SHA256 digest")
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return
    partial = path.with_suffix(path.suffix+".partial")
    with urllib.request.urlopen(url,timeout=60) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response,handle)
    observed = hashlib.sha256(partial.read_bytes()).hexdigest()
    if observed != digest:
        raise RuntimeError("BPG archive SHA256 mismatch; do not execute it. "
                           f"URL={url}; bytes={partial.stat().st_size}; "
                           f"expected={digest}; observed={observed}")
    partial.replace(path)


def _safe_member(root,name):
    target = (root/name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Unsafe codec archive path")
    return target


def install_official_bpg(cache_dir):
    """Return (encoder,decoder,provenance); Linux build requires apt dependencies.

    On Colab install build-essential/cmake/yasm/libpng-dev/libjpeg-dev/zlib1g-dev
    first. Uses official bundled x265; GUI and optional JCTVC are disabled.
    Never installs system packages or changes PATH on Windows.
    """
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    if platform.system() == "Windows":
        url,digest = BPG_WIN64_URL,BPG_WIN64_SHA256
        archive = root/"bpg-0.9.8-win64.zip"
        target = root/"win64"
        _get_verified_archive(url,digest,archive)
        encoder,decoder = target/"bpgenc.exe",target/"bpgdec.exe"
        if not encoder.is_file() or not decoder.is_file():
            with zipfile.ZipFile(archive) as source:
                for item in source.infolist():
                    path = _safe_member(target,item.filename)
                    if item.is_dir():
                        path.mkdir(parents=True,exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True,exist_ok=True)
                        with source.open(item) as src,path.open("wb") as dest:
                            shutil.copyfileobj(src,dest)
        build_command = "official win64 portable distribution; no build"
    elif platform.system() == "Linux":
        url,digest = BPG_SOURCE_URL,BPG_SOURCE_SHA256
        archive = root/"libbpg-0.9.8.tar.gz"
        _get_verified_archive(url,digest,archive)
        # Never reuse the failed 3.31 build tree from earlier notebook versions.
        extract_root = root/BPG_BUILD_RECIPE
        target = extract_root/"libbpg-0.9.8"
        encoder,decoder = target/"bpgenc",target/"bpgdec"
        manifest_path = extract_root/"completed_build.json"
        expected = {"recipe":BPG_BUILD_RECIPE,"archive_sha256":digest,"cmake_version":BPG_CMAKE_VERSION}
        valid_cached_build = False
        if manifest_path.is_file() and encoder.is_file() and decoder.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                valid_cached_build = (all(manifest.get(k)==v for k,v in expected.items())
                    and manifest.get("encoder_sha256")==hashlib.sha256(encoder.read_bytes()).hexdigest()
                    and manifest.get("decoder_sha256")==hashlib.sha256(decoder.read_bytes()).hexdigest())
            except (OSError,ValueError):
                pass
        if not valid_cached_build:
            with tarfile.open(archive,"r:gz") as source:
                for item in source.getmembers():
                    path = _safe_member(extract_root,item.name)
                    if not path.is_relative_to(target):
                        raise ValueError("Unexpected top-level BPG source directory")
                    if item.isdir():
                        path.mkdir(parents=True,exist_ok=True)
                    elif item.isfile():
                        path.parent.mkdir(parents=True,exist_ok=True)
                        with source.extractfile(item) as src,path.open("wb") as dest:
                            shutil.copyfileobj(src,dest)
                        path.chmod(item.mode & 0o777)
                    else:
                        raise ValueError("Unexpected link/device in source archive")
            # Keep upstream source unchanged and use the tested CMake version.
            environment = dict(os.environ)
            try:
                import cmake
                environment["PATH"] = str(cmake.CMAKE_BIN_DIR)+os.pathsep+environment.get("PATH","")
            except ImportError:
                pass
            version = subprocess.run(["cmake","--version"],env=environment,check=True,
                                     capture_output=True,text=True,timeout=15).stdout
            cmake_version = _validated_cmake_version(version)
            command = ["make","-j2","bpgenc","bpgdec","USE_BPGVIEW=","USE_JCTVC=",
                       "CMAKE_OPTS=-DENABLE_LIBNUMA=OFF"]
            log = root/"libbpg_build.log"
            print("Building official BPG0.9.8 once; compiler log:",log,flush=True)
            try:
                with log.open("wb") as handle:
                    # Official Makefile configures only if x265.out does not
                    # exist. Configure explicitly so a failed earlier configure
                    # cannot trap the next attempt in an incomplete directory.
                    common = ["-DENABLE_LIBNUMA=OFF","-DENABLE_SHARED=OFF","-DENABLE_CLI=OFF"]
                    variants = (
                        ("12bit",["-DHIGH_BIT_DEPTH=ON","-DEXPORT_C_API=OFF","-DMAIN12=ON"]),
                        ("10bit",["-DHIGH_BIT_DEPTH=ON","-DEXPORT_C_API=OFF","-DMAIN10=ON"]),
                        ("8bit",["-DLINKED_10BIT=ON","-DLINKED_12BIT=ON"]))
                    for name,flags in variants:
                        build = target/"x265.out"/name
                        build.mkdir(parents=True,exist_ok=True)
                        subprocess.run(["cmake","-S",str(target/"x265/source"),"-B",str(build),*common,*flags],
                                       env=environment,check=True,timeout=120,stdout=handle,stderr=subprocess.STDOUT)
                    subprocess.run(command,cwd=target,env=environment,check=True,timeout=900,
                                   stdout=handle,stderr=subprocess.STDOUT)
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                full_log = log.read_text(encoding="utf-8",errors="replace")
                errors = [line for line in full_log.splitlines() if any(token in line.lower()
                          for token in ("fatal error:"," error:","cmake error","undefined reference","***"))]
                detail = "Error lines:\n"+"\n".join(errors[-20:])+"\nCompiler tail:\n"+full_log[-4000:]
                raise RuntimeError("Official BPG source build failed. "+detail) from error
            if not encoder.is_file() or not decoder.is_file():
                raise RuntimeError("Build exited successfully without both BPG executables")
            manifest = {**expected,"encoder_sha256":hashlib.sha256(encoder.read_bytes()).hexdigest(),
                        "decoder_sha256":hashlib.sha256(decoder.read_bytes()).hexdigest()}
            manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        build_command = "CMake3.28.3 explicit8/10/12bit configure; make -j2 bpgenc bpgdec USE_BPGVIEW= USE_JCTVC= CMAKE_OPTS=-DENABLE_LIBNUMA=OFF"
    else:
        raise RuntimeError("This notebook supports Colab Linux; local smoke testing also supports Win64")
    return str(encoder),str(decoder),{
        "release":BPG_RELEASE,"official_distribution_url":url,"archive_sha256":digest,
        "publisher_signature_verified":False,"platform":platform.platform(),
        "build_command":build_command,
        "build_recipe":BPG_BUILD_RECIPE if platform.system()=="Linux" else "official-win64-098",
        "encoder_binary_sha256":hashlib.sha256(encoder.read_bytes()).hexdigest(),
        "decoder_binary_sha256":hashlib.sha256(decoder.read_bytes()).hexdigest(),
        "redistribution":"Do not commit codecs/DLLs to GitHub. Upstream encoder GPLv2, decoder mixed BSD/LGPL2.1; see bundled licenses."}
