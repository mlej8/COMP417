#! /bin/bash

for step_size in 5 8 11 14 17 20 23 26 29 32
do
    echo "Step_size $step_size"
for i in `seq 1 10`
do
    echo "Iteration $i"
    python rrt_planner_point_robot.py \
        --start_pos_x 10 \
        --start_pos_y 270 \
        --target_pos_x 900 \
        --target_pos_y 30 \
        --rrt_sampling_policy gaussian \
        --seed $i \
        --step_size $step_size >> point.log
done
done