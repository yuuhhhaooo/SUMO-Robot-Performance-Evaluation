#!/usr/bin/env python3
"""Build the published RVO2 library (ORCA) for this benchmark.

Upstream:
    RVO2            UNC GAMMA group -- reference implementation of
                    van den Berg et al., "Reciprocal n-Body Collision
                    Avoidance", ISRR 2009.
    Python-RVO2     https://github.com/sybrenstuvel/Python-RVO2 (Cython
                    bindings; not published on PyPI, hence this script).

    python sim/third_party/build_rvo2.py [--dest DIR] [--ref COMMIT]

The build needs a C++ toolchain (MSVC on Windows, gcc/clang elsewhere), CMake
and Cython:  pip install cython cmake

Two mechanical patches are applied to the upstream setup.py. Neither touches
the algorithm -- they only make a 2020-era build work with current CMake and
MSVC. Both are printed when applied:

  1. CMAKE_POLICY_VERSION_MINIMUM=3.5. Upstream's CMakeLists declares a
     cmake_minimum_required below 3.5, which CMake 4.x refuses outright.
  2. --config Release plus the config subdirectory on the library search path.
     MSVC generators are multi-config: without an explicit config they emit a
     Debug (/MDd) static library, which cannot be linked into a Release (/MD)
     Python extension, and they place it in build/RVO2/src/Release rather than
     build/RVO2/src.

On Windows the destination MUST be a short path. MSBuild's FileTracker fails
with FTK1011 when the intermediate directory exceeds MAX_PATH, which a deep
temp/scratch directory does easily; the default below is deliberately shallow.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

UPSTREAM = "https://github.com/sybrenstuvel/Python-RVO2.git"
DEFAULT_REF = "c2c46ba8d59556aa10faf03479293236efea154d"   # 2020-08-07


def _patch_setup_py(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    if "CMAKE_POLICY_VERSION_MINIMUM" in src:
        return
    path.with_suffix(".py.orig").write_text(src, encoding="utf-8")
    src = src.replace(
        "subprocess.check_call(['cmake', '../..', '-DCMAKE_CXX_FLAGS=-fPIC'],\n"
        "                                  cwd=build_dir)",
        "cmake_args = ['cmake', '../..',\n"
        "                          '-DCMAKE_POLICY_VERSION_MINIMUM=3.5']\n"
        "            if os.name != 'nt':\n"
        "                cmake_args.append('-DCMAKE_CXX_FLAGS=-fPIC')\n"
        "            subprocess.check_call(cmake_args, cwd=build_dir)")
    src = src.replace(
        "subprocess.check_call(['cmake', '--build', '.'], cwd=build_dir)",
        "subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'],\n"
        "                              cwd=build_dir)")
    src = src.replace(
        "library_dirs=['build/RVO2/src'],",
        "library_dirs=['build/RVO2/src', 'build/RVO2/src/Release'],")
    src = src.replace("extra_compile_args=['-fPIC']",
                      "extra_compile_args=([] if os.name == 'nt' else ['-fPIC'])")
    if "import os" not in src.split("\n\n")[0]:
        src = "import os\n" + src
    path.write_text(src, encoding="utf-8")
    print("patched upstream setup.py (cmake policy, Release config, lib dir)")


def main() -> int:
    ap = argparse.ArgumentParser()
    default_dest = (Path("C:/Users") / os.environ.get("USERNAME", "user") / "rvo2"
                    if os.name == "nt" else Path.home() / ".cache" / "rvo2")
    ap.add_argument("--dest", default=str(default_dest),
                    help="build directory (keep it SHORT on Windows)")
    ap.add_argument("--ref", default=DEFAULT_REF,
                    help="upstream commit to build (default: pinned)")
    ap.add_argument("--force", action="store_true",
                    help="delete and re-clone an existing destination")
    args = ap.parse_args()

    dest = Path(args.dest)
    if dest.exists() and args.force:
        shutil.rmtree(dest, ignore_errors=True)
    if not dest.exists():
        print(f"cloning {UPSTREAM} -> {dest}")
        subprocess.check_call(["git", "clone", "--quiet", UPSTREAM, str(dest)])
        subprocess.check_call(["git", "checkout", "--quiet", args.ref],
                              cwd=str(dest))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(dest),
                          capture_output=True, text=True).stdout.strip()
    print(f"upstream commit: {head}")

    _patch_setup_py(dest / "setup.py")

    env = dict(os.environ, CMAKE_POLICY_VERSION_MINIMUM="3.5")
    subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"],
                          cwd=str(dest), env=env)

    built = list(dest.glob("rvo2*.pyd")) + list(dest.glob("rvo2*.so"))
    if not built:
        print("ERROR: no rvo2 extension produced", file=sys.stderr)
        return 1
    print(f"\nbuilt: {built[0]}")
    print("\nAdd it to the interpreter path, e.g.:")
    print(f'    export RVO2_PATH="{dest}"        # bash')
    print(f'    $env:RVO2_PATH = "{dest}"        # PowerShell')
    print("\nsim/planners/orca_rvo2_planner.py reads RVO2_PATH automatically.")
    r = subprocess.run([sys.executable, "-c",
                        "import rvo2; s=rvo2.PyRVOSimulator(0.5,10,10,5,2,0.3,1.0);"
                        "a=s.addAgent((0,0)); s.setAgentPrefVelocity(a,(1,0));"
                        "s.doStep(); print('self-test OK, velocity',"
                        "s.getAgentVelocity(a))"],
                       cwd=str(dest), capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
