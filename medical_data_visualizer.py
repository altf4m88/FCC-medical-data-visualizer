import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['bmi'] = df['weight'] / ( (df['height'] / 100) ** 2)

df['overweight'] = (df['bmi'] > 25).astype(int)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1 )
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1 )

# 4
def draw_cat_plot():
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    df_cat_grouped = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')

    catplot = sns.catplot(x='variable', y='count', hue='value', col='cardio', data=df_cat_grouped, kind='bar', height=5, aspect=1.5)

    fig = catplot.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[df['ap_lo'] <= df['ap_hi']]

    df_heat = df[(df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975))]
    df_heat = df[(df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr))

    plt.figure(figsize=(10, 10))

    heatmap = sns.heatmap(corr, annot=True, fmt=".1f", mask=mask, cmap="coolwarm", cbar=True, square=True)

    # 14
    fig = heatmap.fig

    # 16
    fig.savefig('heatmap.png')
    return fig
