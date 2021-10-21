import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
filename = "line_robot_logs"
columns = ["seed","step_size","rrt_nsteps","robot_length","totaldist","path_num_nodes","graph_num_nodes"]
df = pd.read_csv(filename, names=columns)

rrt_iterations_data = []
avg_rrt_iterations_data = []
for robot_length in df.robot_length.unique():
    df_robot_length = df.loc[df.robot_length == robot_length]
    rrt_iterations_data.append([robot_length] + (df_robot_length.rrt_nsteps.tolist()))
    avg_rrt_iterations_data.append([robot_length] + [df_robot_length.rrt_nsteps.mean()])

columns = ["robot_length", *[f"trial_{i}" for i in df.seed.unique()]]
rrt_niterations_df = pd.DataFrame(rrt_iterations_data, columns=columns)

rrt_niterations_plot = rrt_niterations_df.plot(title="Num of RRT iterations vs robot length", x="robot_length", rot=0, kind="bar", stacked=False, legend=True, xlabel="Robot Length", ylabel="Number of RRT iterations", figsize=(24,16))
rrt_niterations_plot.figure.savefig("rrt_niterations_plot_line.png")

avg_rrt_iterations_df = pd.DataFrame(avg_rrt_iterations_data, columns=["robot_length", "avg_nsteps"])

avg_rrt_niterations_plot = avg_rrt_iterations_df.plot(title="Average number of RRT iterations vs robot length", x="robot_length", rot=0, kind="bar", stacked=False, legend=True, xlabel="Robot Length", ylabel="Average number of RRT iterations", figsize=(24,16))
avg_rrt_niterations_plot.figure.savefig("avg_rrt_niterations_plot_line.png")