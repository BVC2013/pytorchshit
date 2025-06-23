import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as pd

def plot_umap(before, after):
    reducer = umap.UMAP()
    all_data = pd.concat([before, after])
    embedding = reducer.fit_transform(all_data)
    labels = ["Raw"] * len(before) + ["Compensated"] * len(after)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, alpha=0.5)
    plt.title("UMAP: Before vs After Compensation")
    plt.show()
