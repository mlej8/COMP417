import matplotlib.pyplot as plt

accuracy = [0.63,0.53,0.77,0.67,0.63,0.73,0.53,0.77,0.50,0.63,0.47,0.67,0.70,0.70,0.67,0.60]
n_components = list(range(3,18 + 1))

plt.plot(n_components, accuracy)
plt.ylabel('accuracy')
plt.xlabel('number of components')
plt.title('Accuracy vs Number of Components')
plt.savefig("plots/accuracy.png")
plt.clf()

# arable 
title = "arable"
recall = [0.14,0.14,0.57,0.14,0.29,0.43,0.29,0.57,0.14,0.00,0.29,0.43,0.29,0.29,0.43,0.14]
precision = [0.33,0.14,0.80,1.00,0.50,1.00,0.29,1.00,0.25,0.00,0.40,0.60,1.00,1.00,0.75,1.00]
f1_scores = [0.20,0.14,0.67,0.25,0.36,0.60,0.29,0.73,0.18,0.00,0.33,0.50,0.44,0.44,0.55,0.25]
plt.plot(n_components, recall)
plt.ylabel('recall')
plt.xlabel('number of components')
plt.title(f'Recall vs Number of Components for {title}')
plt.savefig(f"plots/recall_{title}.png")
plt.clf()
plt.plot(n_components, precision)
plt.ylabel('precision')
plt.xlabel('number of components')
plt.title(f'Precision vs Number of Components for {title}')
plt.savefig(f"plots/precision_{title}.png")
plt.clf()
plt.plot(n_components, f1_scores)
plt.ylabel('f1-score')
plt.xlabel('number of components')
plt.title(f'F1 Scores vs Number of Components for {title}')
plt.savefig(f"plots/f1_scores_{title}.png")



# urban 
title = "urban"
precision = [0.61,0.65,0.75,0.70,0.74,0.73,0.65,0.70,0.63,0.82,1.00,0.78,0.92,0.73,1.00,0.76]
recall = [0.88,0.94,0.94,1.00,0.88,1.00,0.69,1.00,0.75,0.88,0.50,0.88,0.75,1.00,0.62,0.81]
f1_scores = [0.72,0.77,0.83,0.82,0.80,0.84,0.67,0.82,0.69,0.85,0.67,0.82,0.83,0.84,0.77,0.79]
plt.clf()
plt.plot(n_components, recall)
plt.ylabel('recall')
plt.xlabel('number of components')
plt.title(f'Recall vs Number of Components for {title}')
plt.savefig(f"plots/recall_{title}.png")
plt.clf()
plt.plot(n_components, precision)
plt.ylabel('precision')
plt.xlabel('number of components')
plt.title(f'Precision vs Number of Components for {title}')
plt.savefig(f"plots/precision_{title}.png")
plt.clf()
plt.plot(n_components, f1_scores)
plt.ylabel('f1-score')
plt.xlabel('number of components')
plt.title(f'F1 Scores vs Number of Components for {title}')
plt.savefig(f"plots/f1_scores_{title}.png")


# water
title = "water"
precision = [1.00,0.00,0.80,0.50,0.43,0.60,0.50,1.00,0.29,0.42,0.24,0.43,0.47,0.50,0.44,0.33]
recall = [0.57,0.00,0.57,0.43,0.43,0.43,0.43,0.43,0.29,0.71,0.57,0.43,1.00,0.43,1.00,0.57]
f1_scores = [0.73,0.00,0.67,0.46,0.43,0.50,0.46,0.60,0.29,0.53,0.33,0.43,0.64,0.46,0.61,0.42]
plt.clf()
plt.plot(n_components, recall)
plt.ylabel('recall')
plt.xlabel('number of components')
plt.title(f'Recall vs Number of Components for {title}')
plt.savefig(f"plots/recall_{title}.png")
plt.clf()
plt.plot(n_components, precision)
plt.ylabel('precision')
plt.xlabel('number of components')
plt.title(f'Precision vs Number of Components for {title}')
plt.savefig(f"plots/precision_{title}.png")
plt.clf()
plt.plot(n_components, f1_scores)
plt.ylabel('f1-score')
plt.xlabel('number of components')
plt.title(f'F1 Scores vs Number of Components for {title}')
plt.savefig(f"plots/f1_scores_{title}.png")
