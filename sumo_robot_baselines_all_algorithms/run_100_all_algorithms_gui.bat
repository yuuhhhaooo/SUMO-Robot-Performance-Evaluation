@echo off
python run_random_batch_overlay_all.py --algorithm dwa --num-seeds 100 --flow-mode personsPerHour --sumo-gui --output-dir batch_dwa_gui_100
python run_random_batch_overlay_all.py --algorithm astar --num-seeds 100 --flow-mode personsPerHour --sumo-gui --output-dir batch_astar_gui_100
python run_random_batch_overlay_all.py --algorithm orca --num-seeds 100 --flow-mode personsPerHour --sumo-gui --output-dir batch_orca_gui_100
python run_random_batch_overlay_all.py --algorithm rrt --num-seeds 100 --flow-mode personsPerHour --sumo-gui --output-dir batch_rrt_gui_100
pause
