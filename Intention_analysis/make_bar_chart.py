import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = "/Users/vscalar/project/M.I.N.D/Intention_analysis/labeled/discussion1_q1_label.csv"
df = pd.read_csv(file_path)

# 1. Calculate the frequencies of the original clusters
cluster_counts = df['Label'].value_counts()
#print(cluster_counts)

# 2. Create a new column mapping single-instance clusters to 'others'
# We convert all labels to strings so they can coexist with the word 'others'
df['cluster_viz'] = df['Label'].apply(lambda x: 'others' if cluster_counts[x] == 1 else str(x))

# 3. Get the new order of clusters sorted by frequency (descending)
cluster_order = df['cluster_viz'].value_counts().index

# 4. Initialize the plot layout
fig, ax = plt.subplots(figsize=(8, 5))

# 5. Create the bar chart using the updated column and order
sns.countplot(
    data=df, 
    x='cluster_viz', 
    order=cluster_order, 
    palette='muted', 
    ax=ax
)

# 6. Customize titles and labels for readability
ax.set_title('D1_Q1', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Cluster Label', fontsize=12, labelpad=10)
ax.set_ylabel('Frequency (Count)', fontsize=12, labelpad=10)

# Ensure the text labels don't overlap or truncate
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
plt.tight_layout()

# 7. Save the visualization
plt.savefig('D1_Q1.png', dpi=300)