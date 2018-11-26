import pandas as pd
import numpy as np
import seaborn as sns

s_f = np.load('s_f.npy')
s_h = np.load('s_h.npy')
s_z = np.load('s_z.npy')

f = s_f['test_score']
h = s_h['test_score']
z = s_z['test_score']

d = {'f': f, 'h': h, 'z': z}
df = pd.DataFrame(data=d)

sns.set(style='whitegrid')
ax = sns.boxplot(x='C_SVS', y='Test ', data=df, linewidth=2.5)
