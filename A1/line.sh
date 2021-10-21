#! /bin/bash

for robot_length in 5 10 15 20 25 30 35 40 45 50
do
    echo "Robot length $robot_length"
for i in `seq 1 10`
do
    echo "seed $i"
    python rrt_planner_line_robot.py \
        --start_pos_x 100 \
        --start_pos_y 630 \
        --target_pos_x 800 \
        --target_pos_y 150 \
        --robot_length $robot_length \
        --seed $i \
        --rrt_sampling_policy uniform >> line.log
done
done