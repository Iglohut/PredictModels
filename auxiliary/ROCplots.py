import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Fig path
path_figs = os.getcwd() + '/figs/'

# Load ROC metrics in one df
def get_df(skiprows=None):
    path_base = os.getcwd() + '/Data/'
    subsets = ['None', 'or', 'od', 'con']
    modelnames = ['RF', 'XGBoost']
    dfs =[]
    for subset in subsets:
        for modelname in modelnames:
            path_model = path_base + subset + '_' + modelname + '_metrics.csv'
            dfs.append(pd.read_csv(path_model, skiprows=skiprows))
    df = pd.concat(dfs, ignore_index=True)

    # 'Pre-processing'
    df = df.replace('None', 'all')
    return df



 # ------ PLOT AUROCs -------- #
df = get_df(skiprows=[1, 3])

# Draw a nested barplot f AUROCs
g = sns.catplot(x="condition", y="AUROC", hue="model", data=df, height=6, kind="bar", palette="gray", legend=False)
g.despine(left=True)
g.add_legend(title='')

# Draw on indication of Pvalues
p_values = list(df['p_value'])
p_values = np.array(p_values)
p_strings = (p_values < 0.05).astype(int) + (p_values < 0.01) + (p_values < 0.001)
x_step = 0.18 # x-axis categorical pixel step
x_locs = [list(np.array([0.16, 0.24]) + i * x_step)  for i in range(4)]
x_locs = np.array(x_locs).flatten()
for i, v in enumerate(p_strings):
    g.fig.text(x_locs[i], df.iloc[i]['AUROC'] / plt.ylim()[-1], "".join(["*"] * v), color='black', ha="center")

plt.savefig(path_figs + 'AUROCs.pdf')
plt.close()

# ---------- PLOT ROCs ------
df = get_df()
subsets = df['condition'].unique()
modelnames = df['model'].unique()

# Plotting
fig, ax= plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance') # Baseplot with red chance line


colors = ['black', 'gray', 'rosybrown', 'olivedrab']
linestyles = [None, ':']


for i, subset in enumerate(subsets):
    for j, modelname in enumerate(modelnames):
        df_plot = df[(df['condition'] == subset) & (df['model'] == modelname)]
        ax.plot(df_plot['fpr'], df_plot['tpr'], color=colors[i], linestyle= linestyles[j], label='{}, {} (AUROC: {:.4f})'.format(subset, modelname, df_plot['AUROC'].unique()[0]), lw=2)



ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right")
fig.set_size_inches(10, 6)
plt.savefig(path_figs + 'ROCs.pdf')
plt.close()
