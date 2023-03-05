import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
def is_overweight(row):
  weight = row['weight']       # weight in kg
  height = row['height'] / 100 # height in m

  if (weight/(height**2)) > 25:
    return 1
  else:
    return 0

df['overweight'] = df.apply(lambda row: is_overweight(row), axis=1)
# print(df.head())

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] != 0, 'cholesterol'] = 1
# print(df.cholesterol.unique())

df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] != 0, 'gluc'] = 1
# print(df.gluc.unique())

# Draw Categorical Plot
def draw_cat_plot():
  # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
  df_cat = df.melt(id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])
  # print(df_cat.head())
  
  # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
  
  # Draw the catplot with 'sns.catplot()'
  g = sns.catplot(data=df_cat,
                 x='variable',
                 hue='value',
                 col='cardio',
                 kind='count').set_axis_labels("variable", "total")
  
  # Get the figure for the output
  fig = g.fig
  
  
  # Do not modify the next two lines
  fig.savefig('catplot.png')
  return fig


# Draw Heat Map
def draw_heat_map():
  # Clean the data
  df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
               (df['height'] >= df['height'].quantile(0.025)) &
               (df['height'] <= df['height'].quantile(0.975)) &
               (df['weight'] >= df['weight'].quantile(0.025)) &
               (df['weight'] <= df['weight'].quantile(0.975))]
  # print(df_heat.head());
  
  # Calculate the correlation matrix
  corr = df_heat.corr(method='pearson')
  
  # Generate a mask for the upper triangle
  mask = np.triu(corr)
  
  # Set up the matplotlib figure
  fig, ax = plt.subplots()
  
  # Draw the heatmap with 'sns.heatmap()'
  sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=1, square=True, cbar_kws={'shrink':0.5})
  
  # Do not modify the next two lines
  fig.savefig('heatmap.png')
  return fig