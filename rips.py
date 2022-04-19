from sample import sample
import seaborn as sns
import matplotlib.pyplot as plt
points_per_cluster = 100
from ripser import ripser
import numpy as np
import pandas as pd
from IPython.display import display
from persim import plot_diagrams

s1 = np.array(([sample(20, 18, 0, 0.0) for _ in range(points_per_cluster)]))
s2 = np.array(([sample(20, 12, 0, 0.0) for _ in range(points_per_cluster)]))
s = np.concatenate([s1, s2+10])
x, y = zip(*s)

sns.scatterplot(x, y)
plt.show()
rips = ripser(s)
display(rips)

display(plot_diagrams(rips['dgms'], show=True))

b0 = rips['dgms'][0]
b1 = rips['dgms'][1]

df = pd.DataFrame(b1, columns=['birth', 'death'])
intervals = np.arange(int(df.birth.min()), int(df.death.max()))
max_param = int(df.death.max())

# for j in intervals:
#     i = j - 1
#     if df.between(i, j)
# np.where(np.logical_and(a >= 6, a <= 10))
# def grouper(row):
#     row.birth
#     row.death

# df['group'] = df.apply(lambda x: )




