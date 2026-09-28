"""Build a serial native reference using already available dependency headers.

This optional developer utility avoids modifying the normal FLARE build. Run it
with the Python interpreter that will load the resulting extension. It requires
a C++ compiler and existing Eigen, nlohmann/json, and pybind11 include trees.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sysconfig


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eigen-include", type=Path, required=True)
    parser.add_argument("--json-include", type=Path, required=True)
    parser.add_argument("--pybind11-include", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--revision", default="199273867d48f3a91415fa44da6c0dfc5ead759f")
    args = parser.parse_args()
    if platform.system() not in ("Darwin", "Linux"):
        parser.error("This reference-only utility supports macOS and Linux.")
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.revision + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "diff", revision, "--", "src", "CMakeLists.txt"], cwd=ROOT
    )
    if changed:
        parser.error("Native reference sources must match the requested reference revision.")
    cmake = (ROOT / "CMakeLists.txt").read_text()
    sources = []
    for variable in ("FLARE_SOURCES", "PYBIND_SOURCES"):
        block = re.search(r"set\(" + variable + r"\s+(.*?)\)", cmake, re.S)
        if block is None:
            parser.error("Cannot find native source list " + variable)
        sources.extend(re.findall(r"src/[\w/]+\.cpp", block.group(1)))
    includes = [
        ROOT / "src/flare_pp", ROOT / "src/flare_pp/descriptors",
        ROOT / "src/flare_pp/kernels", ROOT / "src/flare_pp/bffs",
        args.eigen_include, args.json_include, args.pybind11_include,
        Path(sysconfig.get_path("include")),
    ]
    for include in includes:
        if not include.is_dir():
            parser.error("Missing include directory: " + str(include))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve() / ("_C_flare" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [args.compiler, "-O2", "-std=c++11", "-fPIC"]
    if platform.system() == "Darwin":
        command += ["-dynamiclib", "-undefined", "dynamic_lookup", "-arch", platform.machine()]
    else:
        command += ["-shared"]
    command += ["-I" + str(path.resolve()) for path in includes]
    command += sources + ["-o", str(output)]
    subprocess.run(command, cwd=ROOT, check=True)
    tracked_sources = [ROOT / "CMakeLists.txt"] + sorted(
        path for path in (ROOT / "src").rglob("*")
        if path.suffix in (".cpp", ".h", ".hpp")
    )
    manifest = {
        "baseline_revision": revision,
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in tracked_sources},
        "extension_sha256": sha256(output),
        "compiler_command": command,
        "compiler_version": subprocess.check_output(
            [args.compiler, "--version"], text=True
        ).splitlines()[0],
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "openmp": False,
        "lapack": False,
    }
    manifest_path = output.parent / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output)
    print(manifest_path)


if __name__ == "__main__":
    main()
