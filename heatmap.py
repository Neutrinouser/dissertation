import seaborn 
import matplotlib.pyplot as plt
import pickle

with open('parameterSweepData.p','rb') as handle:
    d = pickle.load(handle); 

features = ['dispersion','frontPosition','tumourRadius']

# In scientific notation
ylabels, y = zip(*[("{:.1e}".format(i),i) for i in d.keys() ])
xlabels, x = zip(*[("{:.1e}".format(i),i) for i in d[y[0]].keys()])

# Form new dictionary
#s = {k: [[j[k][0] for j in i.values() ] for i in d.values() ] for k in features}
s = {k: [[ d[j][i][k][0] for i in x ] for j in y] for k in d[y[0]][x[0]].keys()}

for k in features:
    ax=seaborn.heatmap(s[k])
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    plt.title(k)
    plt.show()