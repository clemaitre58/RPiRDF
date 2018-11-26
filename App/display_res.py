import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('data.csv')

s_f = np.load('s_f.npy')
s_h = np.load('s_h.npy')
s_z = np.load('s_z.npy')

f = s_f['f']
h = s_h['h']
z = s_z['z']


sns.set(style='whitegrid')
ax = sns.boxplot(x='C_SVS', y='Test ', data=df, linewidth=2.5)
