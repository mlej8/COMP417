import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
filename = "point_robot_logs"
columns = ["seed","step_size","rrt_nsteps","totaldist","path_num_nodes","graph_num_nodes"]
df = pd.read_csv(filename, names=columns)
a = df.groupby("step_size")

labels = df.step_size.unique()
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

num_rrt_iterations = []
for unique_seed in df.seed.unique():
    num_rrt_iterations.append(df.loc[df.seed == unique_seed].rrt_nsteps.tolist())

rrt_path_length = []
for unique_seed in df.seed.unique():
    rrt_path_length.append(df.loc[df.seed == unique_seed].totaldist.tolist())

rrt_iterations = []
for unique_step_size in df.step_size.unique():
    bystepsize = [str(unique_step_size)]
    rrt_steps = (df.loc[df.step_size == unique_step_size].rrt_nsteps.tolist())
    for step in rrt_steps:
        bystepsize.append(step)
    rrt_iterations.append(bystepsize)

columns = ["step_size"]
for unique_seed in df.seed.unique():
    columns.append(str(unique_seed))
new_df = pd.DataFrame(rrt_iterations, columns=columns)
new_df.plot(x="step_size", kind="bar", stacked=False)
# x = np.arange(len(labels))  # the label locations
# width = 0.05  # the width of the bars

# fig, ax = plt.subplots()
# bars =[]
# for i, rrt_it in enumerate(num_rrt_iterations, 1):
#     bars.append(ax.bar(x + (width/len(num_rrt_iterations)) * (i-1), rrt_it, width, label=f'{i}'))

# for bar in bars:
#     ax.bar_label(bar, padding=3)
# # rects1 = ax.bar(x - width/2, men_means, width, label='Men')
# # rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Scores')
# # ax.set_title('Scores by group and gender')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# # ax.legend()


# # ax.bar_label(rects2, padding=3)

# fig.tight_layout()

# plt.show()