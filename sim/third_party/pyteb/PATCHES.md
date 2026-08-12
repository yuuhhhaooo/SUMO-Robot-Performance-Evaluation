# pyteb -- provenance and patches

## What this is

A **thin pybind11 bridge** to the published Timed-Elastic-Band local planner:

* Upstream: `rst-tu-dortmund/teb_local_planner`, C. Roesmann, W. Feiten, T. Woesch,
  F. Hoffmann, T. Bertram — *Trajectory modification considering dynamic constraints
  of autonomous robots* (ROBOTIK 2012) and *Efficient trajectory optimization using a
  sparse model* (ECMR 2013).
* Version adopted: **0.9.1** (ROS1 Noetic release; upstream tag `v0.9.1` on
  branch `noetic-devel`).
* Obtained as the prebuilt RoboStack conda package
  `ros-noetic-teb-local-planner 0.9.1` build `np2py312h8dce203_24`
  (channel `robostack-noetic`, platform `win-64`), which ships
  `Library/bin/teb_local_planner.dll`, `Library/lib/teb_local_planner.lib`
  and `Library/include/teb_local_planner/*.h`, plus `libg2o` (2020.5.3).
* Upstream license: BSD (see `LICENSE.teb_local_planner`, `upstream_package.xml`).
  g2o is BSD; its `csparse_extension` is LGPL-3.0+.

## Patches to upstream code

**NONE.** No upstream source file was modified, recompiled or vendored. The
teb_local_planner binary is used exactly as published by RoboStack.

`src/pyteb.cpp` is *new* glue code, not a modified copy of anything upstream.
It contains no planning mathematics: it fills a `teb_local_planner::TebConfig`,
builds a `teb_local_planner::ObstContainer` of `PointObstacle`/`LineObstacle`,
calls `TebOptimalPlanner::plan()` / `::getVelocityCommand()` /
`HomotopyClassPlanner`, and copies the resulting band out. All Timed-Elastic-Band
maths — the g2o hyper-graph, the pose **and time-difference** vertices, the
velocity/acceleration/obstacle/via-point/time-optimality edges and the sparse
Levenberg–Marquardt solve — runs inside `teb_local_planner.dll`.

## Build workarounds (build system only, not code)

1. `-DROS_BUILD_SHARED_LIBS`.
   `ros/console.h` selects `__declspec(dllimport)` for exported *data* symbols
   only when this macro is defined. RoboStack ships the ROS libraries as DLLs but
   the macro is not set by any installed CMake config, so without it the link
   fails with `LNK2001: unresolved external symbol "bool ros::console::g_initialized"`.
2. Two prebuilt extensions are shipped: `cp312` (for the ROS conda env itself)
   and `cp313` (for the benchmark's Python). `teb_local_planner.dll`,
   `rosconsole.dll` and `roscpp.dll` have **no** Python import dependency
   (verified by reading their PE import tables), so the cp313 build links against
   the same C++ DLLs while embedding the CPython 3.13 ABI.

## Runtime requirement

`teb_local_planner.dll` and its ROS/g2o/boost dependencies must be reachable.
`sim/planners/teb_upstream_planner.py` calls `os.add_dll_directory()` on:

    $TEB_LOCAL_PLANNER_DLL_DIR   (semicolon-separated; overrides the default)
    C:/Users/Mark/tebenv/Library/bin
    C:/Users/Mark/tebenv

Recreate that environment with:

    conda create -y -p C:/Users/Mark/tebenv --override-channels \
        -c robostack-noetic -c conda-forge \
        python=3.12 ros-noetic-teb-local-planner pybind11 numpy

## Rebuilding the bridge

    cmake -S . -B build -G "Visual Studio 18 2026" -A x64 \
          -DPython_EXECUTABLE=<python> -Dpybind11_DIR=<pybind11.get_cmake_dir()> \
          -DTEB_PREFIX=C:/Users/Mark/tebenv/Library
    cmake --build build --config Release
