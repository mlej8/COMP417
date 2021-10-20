#! /bin/bash

for step_size in 2 4 6 8 10 12 14 16 18 20
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