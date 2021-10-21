import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
filename = "point_robot_logs"
columns = ["seed","step_size","rrt_nsteps","totaldist","path_num_nodes","graph_num_nodes"]
df = pd.read_csv(filename, names=columns)
a = df.groupby("step_size")

labels = df.step_size.unique()

rrt_iterations_data = []
rrt_path_lengths = []
for unique_step_size in df.step_size.unique():
    # bystepsize = [str(unique_step_size)]
    df_step_size = df.loc[df.step_size == unique_step_size]
    rrt_iterations_data.append([unique_step_size] + (df_step_size.rrt_nsteps.tolist()))
    rrt_path_lengths.append([unique_step_size] + (df_step_size.totaldist.tolist()))

columns = ["step_size", *[f"trial_{i}" for i in df.seed.unique()]]
rrt_niterations_df = pd.DataFrame(rrt_iterations_data, columns=columns)
path_lengths_df = pd.DataFrame(rrt_path_lengths, columns=columns)

rrt_niterations_plot = rrt_niterations_df.plot(title="Num of RRT iterations vs step size", x="step_size", rot=0, kind="bar", stacked=False, legend=True, xlabel="Step Size", ylabel="Number of RRT iterations")
rrt_niterations_plot.figure.savefig("rrt_niterations_plot.png")

path_lengths_plot = path_lengths_df.plot(title="Path length vs step size",x="step_size", rot=0, kind="bar", stacked=False, legend=True, figsize=(24,16), xlabel="Step Size", ylabel="RRT path length") 
path_lengths_plot.figure.savefig("path_lengths.png")