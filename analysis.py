import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
from scipy import stats
import matplotlib.gridspec as gridspec

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Grocery_Database.csv')

# Set style for better-looking plots
plt.style.use('default')  # Using default style
sns.set_theme(style="whitegrid")  # Set seaborn theme
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Load the data
try:
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded data from {csv_path}")
except FileNotFoundError:
    print(f"Error: Could not find 'Grocery_Database.csv' at {csv_path}")
    print("Current directory contents:")
    print(os.listdir(script_dir))
    exit(1)

# Clean data - remove rows with missing nutritional data or price
nutritional_columns = ['Protein', 'Total Fat', 'Carbohydrate', 'Sugars, total', 
                       'Fiber, total dietary', 'Sodium', 'Cholesterol']
df_clean = df.dropna(subset=['price'] + nutritional_columns)

# Create price bins to group products by price range
price_bins = [0, 2, 5, 10, 50, df_clean['price'].max()]
price_labels = ['$0-2', '$2-5', '$5-10', '$10-50', '$50+']
df_clean['Price_Range'] = pd.cut(df_clean['price'], bins=price_bins, labels=price_labels)

# Remove outliers using z-score
df_filtered = df_clean.copy()
for col in nutritional_columns:
    z_scores = stats.zscore(df_filtered[col], nan_policy='omit')
    df_filtered = df_filtered[(z_scores >= -2) & (z_scores <= 2)]

# Split nutritional columns into two groups for different normalization approaches
main_nutrients = ['Protein', 'Total Fat', 'Carbohydrate', 'Sugars, total', 'Fiber, total dietary']
micro_nutrients = ['Sodium', 'Cholesterol']

# Create a copy of the filtered dataframe for creating the plot data
plot_df = df_filtered.copy()

# Normalize main nutrients normally
for col in main_nutrients:
    max_val = plot_df[col].max()
    if max_val > 0:  # Avoid division by zero
        plot_df[f'{col}_Normalized'] = plot_df[col] / max_val

# Apply special normalization to sodium and cholesterol to make them more visible
# We'll use a more aggressive normalization to bring out their patterns
for col in micro_nutrients:
    # Get 95th percentile instead of max to avoid extreme outliers affecting normalization
    p95 = np.percentile(plot_df[col], 95)
    plot_df[f'{col}_Normalized'] = plot_df[col] / p95
    # Cap at 1.0 for consistency with other nutrients
    plot_df[f'{col}_Normalized'] = plot_df[f'{col}_Normalized'].clip(upper=1.0)

# Prepare data for box plot - combine all normalized columns
nutritional_normalized = [f'{col}_Normalized' for col in main_nutrients + micro_nutrients]
plot_data = pd.melt(plot_df, 
                   id_vars=['Price_Range'], 
                   value_vars=nutritional_normalized,
                   var_name='Nutrient', 
                   value_name='Normalized_Value')

# Clean up nutrient names for display
plot_data['Nutrient'] = plot_data['Nutrient'].replace({
    'Protein_Normalized': 'Protein',
    'Total Fat_Normalized': 'Total Fat',
    'Carbohydrate_Normalized': 'Carbohydrate',
    'Sugars, total_Normalized': 'Sugars',
    'Fiber, total dietary_Normalized': 'Fiber dietary',
    'Sodium_Normalized': 'Sodium',
    'Cholesterol_Normalized': 'Cholesterol'
})

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# Define colors for the price ranges
colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']

# Create the box plot
ax = plt.subplot(111)
box_plot = sns.boxplot(
    data=plot_data,
    x='Nutrient',
    y='Normalized_Value',
    hue='Price_Range',
    palette=colors,
    width=0.8,
    fliersize=2,
    showfliers=False,  # Don't show outliers in the box plot
    ax=ax
)

# Add strip plot to show individual points with low alpha to avoid visual clutter
sns.stripplot(
    data=plot_data,
    x='Nutrient',
    y='Normalized_Value',
    hue='Price_Range',
    palette=colors,
    dodge=True,
    size=2,
    alpha=0.2,
    jitter=True,
    ax=ax
)

# Add a subtitle noting the different normalization for sodium and cholesterol
plt.figtext(0.5, 0.92, 'Nutritional Content Comparison by Price Range', 
          fontsize=12, ha='center')

# Fix overlapping titles by adjusting spacing
plt.title('Are More Nutritious Foods More Expensive?', fontsize=16, y=1.05)
plt.xlabel('Nutritional Categories', fontsize=12)
plt.ylabel('Normalized Nutritional Content (0-1 scale)', fontsize=12)

# Add annotations to highlight sodium and cholesterol
for i, nutrient in enumerate(['Sodium', 'Cholesterol']):
    idx = list(plot_data['Nutrient'].unique()).index(nutrient)
    plt.annotate('*', xy=(idx, 0.05), xytext=(idx, 0.1), 
                fontsize=16, ha='center', va='center')

# Customize legend - only show it once
handles, labels = box_plot.get_legend_handles_labels()
leg = box_plot.legend(handles[:5], labels[:5], title='Price Range', 
           bbox_to_anchor=(1.05, 1), loc='upper left')

# Add the sodium and cholesterol explanation below the legend
sodium_text = 'Note: Sodium and Cholesterol values\nusing 95th percentile normalization\nfor better visibility.'
plt.text(1.05, 0.85, sodium_text, transform=ax.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round'),
         fontsize=10, verticalalignment='top')

# Add statistics in a text box
# Calculate stats based on both normalizations separately
main_nutrients_norm = [f'{col}_Normalized' for col in main_nutrients]
micro_nutrients_norm = [f'{col}_Normalized' for col in micro_nutrients]
all_nutrients_norm = main_nutrients_norm + micro_nutrients_norm

avg_nutrition_by_price = plot_df.groupby('Price_Range')[all_nutrients_norm].mean().mean(axis=1)
corr_price_nutrition = plot_df['price'].corr(plot_df[all_nutrients_norm].mean(axis=1))

stats_text = (
    f"Correlation between price and nutrition: {corr_price_nutrition:.2f}\n\n"
    "Average Nutritional Content by Price:\n"
)
for price_range, avg in avg_nutrition_by_price.items():
    stats_text += f"{price_range}: {avg:.2f}\n"

plt.text(1.05, 0.5, stats_text, transform=ax.transAxes, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
         fontsize=10, verticalalignment='center')

# Add some space at the top for the title
plt.subplots_adjust(top=0.9, bottom=0.1)

# Adjust layout
plt.tight_layout()
plt.savefig('nutrition_price_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Enhanced visualization with better visibility for sodium and cholesterol has been generated!")
