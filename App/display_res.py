import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('test_score.csv', index_col=0)

with sns.plotting_context("poster"):
    sns.boxplot(data=df, whis=100)
    sns.despine(bottom=True)
    plt.ylim([0.5, 1.0])
    plt.yticks([0.5, 0.75, 1.0])
    plt.show()
