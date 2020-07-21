import seaborn 
import matplotlib.pyplot as plt
import pickle

with open('parameterSweepData.p','rb') as handle:
    d = pickle.load(handle) 

# In scientific notation
ylabels, y = zip(*[(round(i,1),i) for i in d.keys() ])
xlabels, x = zip(*[("{:.1e}".format(i),i) for i in d[y[0]].keys()])

# Name of features
features = d[y[0]][x[0]].keys()

# Form new dictionary
s = {k: [[ d[j][i][k][0] for i in x ] for j in y] for k in features}

for k in features:
    ax=seaborn.heatmap(s[k])
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    plt.title(k)
    plt.xlabel('Diffusivity of microbeads')
    plt.ylabel('Proliferation of cells')
    plt.show()
