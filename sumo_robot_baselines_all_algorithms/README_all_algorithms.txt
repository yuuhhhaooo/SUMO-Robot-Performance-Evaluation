SUMO sidewalk robot baseline package: DWA + A* + ORCA-style + RRT
=================================================================

Put all files in the same folder and open CMD in this folder.
This package keeps the same SUMO map but randomly generates pedestrian scenarios for each seed.
The packaged BasicView.xml has <delay value="0.00"/> so GUI playback is faster.

Algorithms included:
- DWA: dwa_sidewalk_robot_random_stop_collision.py
- A*: astar_sidewalk_robot_random_stop_collision.py
- ORCA-style / velocity-obstacle baseline: orca_sidewalk_robot_random_stop_collision.py
- RRT: rrt_sidewalk_robot_random_stop_collision.py
- Shared module for A*, ORCA-style and RRT: sidewalk_robot_common.py
- Unified batch runner: run_random_batch_overlay_all.py

Common behavior:
- The robot is inserted as a SUMO person and moved by TraCI moveToXY().
- The robot is constrained to the north sidewalk.
- With --random-scenario and --seed, pedestrian speed, flow density, and static pedestrian positions are randomized.
- If robot-pedestrian distance is smaller than robot_radius + pedestrian_radius, the current seed stops immediately.
- Each seed outputs trace CSV and metrics JSON.
- The batch runner creates one combined path plot for all seeds.

Test one seed with GUI:

DWA:
python dwa_sidewalk_robot_random_stop_collision.py --cfg BasicConfig.sumocfg --random-scenario --seed 1 --flow-mode personsPerHour --sumo-gui

A*:
python astar_sidewalk_robot_random_stop_collision.py --cfg BasicConfig.sumocfg --random-scenario --seed 1 --flow-mode personsPerHour --sumo-gui

ORCA-style:
python orca_sidewalk_robot_random_stop_collision.py --cfg BasicConfig.sumocfg --random-scenario --seed 1 --flow-mode personsPerHour --sumo-gui

RRT:
python rrt_sidewalk_robot_random_stop_collision.py --cfg BasicConfig.sumocfg --random-scenario --seed 1 --flow-mode personsPerHour --sumo-gui

Run 100 seeds with GUI:

DWA:
python run_random_batch_overlay_all.py --algorithm dwa --num-seeds 100 --flow-mode personsPerHour --sumo-gui --output-dir batch_dwa_gui_100

A*:
python run_random_batch_overlay_all.py --algorithm astar --num-seeds 100 --flow-mode personsPerHour --sumo-gui --output-dir batch_astar_gui_100

ORCA-style:
python run_random_batch_overlay_all.py --algorithm orca --num-seeds 100 --flow-mode personsPerHour --sumo-gui --output-dir batch_orca_gui_100

RRT:
python run_random_batch_overlay_all.py --algorithm rrt --num-seeds 100 --flow-mode personsPerHour --sumo-gui --output-dir batch_rrt_gui_100

Run 100 seeds faster without GUI:

DWA:
python run_random_batch_overlay_all.py --algorithm dwa --num-seeds 100 --flow-mode personsPerHour --output-dir batch_dwa_100

A*:
python run_random_batch_overlay_all.py --algorithm astar --num-seeds 100 --flow-mode personsPerHour --output-dir batch_astar_100

ORCA-style:
python run_random_batch_overlay_all.py --algorithm orca --num-seeds 100 --flow-mode personsPerHour --output-dir batch_orca_100

RRT:
python run_random_batch_overlay_all.py --algorithm rrt --num-seeds 100 --flow-mode personsPerHour --output-dir batch_rrt_100

Main outputs inside each batch folder:
- batch_per_seed_metrics.csv      每个 seed 的单独 evaluation 结果
- batch_summary.json              所有 seed 的平均值和标准差
- batch_all_seed_paths.png        所有 seed 的机器人路径叠加图
- seed_*/...trace.csv             每个 seed 的轨迹
- seed_*/...metrics.json          每个 seed 的指标

Notes:
- For large experiments, headless mode is faster and more stable than GUI.
- ORCA here is an ORCA-style / velocity-obstacle baseline, not a strict RVO2 library implementation.
- To make GUI slower for visual observation, edit BasicView.xml and change <delay value="0.00"/> to something like 20.00 or 200.00.
