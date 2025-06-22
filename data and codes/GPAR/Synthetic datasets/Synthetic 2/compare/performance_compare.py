import matplotlib.pyplot as plt
import pandas as pd

# Read data from final_metrics_summary.csv
metrics_df = pd.read_csv(output_dir + "/final_metrics_summary.csv")

# Extract data from the DataFrame
min_support = metrics_df["Min_Support"].tolist()

# Runtime data (in seconds)
gpar_runtime = metrics_df["GPAR_Runtime (s)"].tolist()
apriori_runtime = metrics_df["Apriori_Runtime (s)"].tolist()
fpgrowth_runtime = metrics_df["FP-Growth_Runtime (s)"].tolist()
eclat_runtime = metrics_df["Eclat_Runtime (s)"].tolist()

# Memory usage data (in MB)
gpar_memory = metrics_df["GPAR_Memory (MB)"].tolist()
apriori_memory = metrics_df["Apriori_Memory (MB)"].tolist()
fpgrowth_memory = metrics_df["FP-Growth_Memory (MB)"].tolist()
eclat_memory = metrics_df["Eclat_Memory (MB)"].tolist()

# Number of frequent itemsets
gpar_itemsets = metrics_df["GPAR_Frequent_Itemsets"].tolist()
apriori_itemsets = metrics_df["Apriori_Frequent_Itemsets"].tolist()
fpgrowth_itemsets = metrics_df["FP-Growth_Frequent_Itemsets"].tolist()
eclat_itemsets = metrics_df["Eclat_Frequent_Itemsets"].tolist()

# Number of rules
gpar_rules = metrics_df["GPAR_Rules"].tolist()
apriori_rules = metrics_df["Apriori_Rules"].tolist()
fpgrowth_rules = metrics_df["FP-Growth_Rules"].tolist()
eclat_rules = metrics_df["Eclat_Rules"].tolist()

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot 1: Runtime Performance (Top Left)
axs[0, 0].plot(min_support, gpar_runtime, 'r-o', label='GPAR (RBF)')
axs[0, 0].plot(min_support, apriori_runtime, 'b--s', label='Apriori')
axs[0, 0].plot(min_support, fpgrowth_runtime, 'y-^', label='FP-Growth')
axs[0, 0].plot(min_support, eclat_runtime, 'g-d', label='Eclat')
axs[0, 0].set_title('Runtime performance')
axs[0, 0].set_xlabel('Minimum support threshold')
axs[0, 0].set_ylabel('Runtime (seconds)')
axs[0, 0].grid(True)

# Plot 2: Memory Usage (Top Right)
axs[0, 1].plot(min_support, gpar_memory, 'r-o')
axs[0, 1].plot(min_support, apriori_memory, 'b--s')
axs[0, 1].plot(min_support, fpgrowth_memory, 'y-^')
axs[0, 1].plot(min_support, eclat_memory, 'g-d')
axs[0, 1].set_title('Memory usage')
axs[0, 1].set_xlabel('Minimum support threshold')
axs[0, 1].set_ylabel('Memory usage (MB)')
axs[0, 1].grid(True)

# Plot 3: Number of Frequent Itemsets (Bottom Left)
axs[1, 0].plot(min_support, gpar_itemsets, 'r-o')
axs[1, 0].plot(min_support, apriori_itemsets, 'b--s')
axs[1, 0].plot(min_support, fpgrowth_itemsets, 'y-^')
axs[1, 0].plot(min_support, eclat_itemsets, 'g-d')
axs[1, 0].set_title('Number of frequent itemsets')
axs[1, 0].set_xlabel('Minimum support threshold')
axs[1, 0].set_ylabel('Number of frequent itemsets')
axs[1, 0].grid(True)

# Plot 4: Number of Rules (Bottom Right)
axs[1, 1].plot(min_support, gpar_rules, 'r-o')
axs[1, 1].plot(min_support, apriori_rules, 'b--s')
axs[1, 1].plot(min_support, fpgrowth_rules, 'y-^')
axs[1, 1].plot(min_support, eclat_rules, 'g-d')
axs[1, 1].set_title('Number of rules generated')
axs[1, 1].set_xlabel('Minimum support threshold')
axs[1, 1].set_ylabel('Number of rules generated')
axs[1, 1].grid(True)

# Adjust layout to make space for the shared legend
plt.tight_layout(rect=[0, 0.1, 1, 1])

# Create a shared legend below the plots
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0))

# Save the figure
plt.savefig('performance_plots_2x2.png')
plt.close()
