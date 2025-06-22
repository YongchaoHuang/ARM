import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the CSV files
output_dir = "~/Synthetic datasets/compare"
csv_files = {
    'rbf': output_dir + '/final_metrics_summary_RBF.csv',
    'shifted_rbf': output_dir + '/final_metrics_summary_shifted_RBF.csv',
    'neural_net': output_dir + '/final_metrics_summary_neural_network_kernel.csv',
    'ntk': output_dir + '/final_metrics_summary_NTK.csv'
}

# Dictionary to store dataframes
dataframes = {}
for key, file in csv_files.items():
    df = pd.read_csv(file)
    dataframes[key] = df

# Extract min_support (should be the same across all files)
min_support = dataframes['rbf']['Min_Support'].tolist()

# Extract GPAR-specific metrics for each kernel type
gpar_rbf_runtime = dataframes['rbf']['GPAR_Runtime (s)'].tolist()
gpar_shifted_rbf_runtime = dataframes['shifted_rbf']['GPAR_Runtime (s)'].tolist()
gpar_neural_net_runtime = dataframes['neural_net']['GPAR_Runtime (s)'].tolist()
gpar_ntk_runtime = dataframes['ntk']['GPAR_Runtime (s)'].tolist()

gpar_rbf_memory = dataframes['rbf']['GPAR_Memory (MB)'].tolist()
gpar_shifted_rbf_memory = dataframes['shifted_rbf']['GPAR_Memory (MB)'].tolist()
gpar_neural_net_memory = dataframes['neural_net']['GPAR_Memory (MB)'].tolist()
gpar_ntk_memory = dataframes['ntk']['GPAR_Memory (MB)'].tolist()

gpar_rbf_itemsets = dataframes['rbf']['GPAR_Frequent_Itemsets'].tolist()
gpar_shifted_rbf_itemsets = dataframes['shifted_rbf']['GPAR_Frequent_Itemsets'].tolist()
gpar_neural_net_itemsets = dataframes['neural_net']['GPAR_Frequent_Itemsets'].tolist()
gpar_ntk_itemsets = dataframes['ntk']['GPAR_Frequent_Itemsets'].tolist()

gpar_rbf_rules = dataframes['rbf']['GPAR_Rules'].tolist()
gpar_shifted_rbf_rules = dataframes['shifted_rbf']['GPAR_Rules'].tolist()
gpar_neural_net_rules = dataframes['neural_net']['GPAR_Rules'].tolist()
gpar_ntk_rules = dataframes['ntk']['GPAR_Rules'].tolist()

# Compute averages for Apriori, FP-Growth, and Eclat across the four files
apriori_runtimes = [dataframes[key]['Apriori_Runtime (s)'].tolist() for key in dataframes]
apriori_avg_runtime = np.mean(apriori_runtimes, axis=0).tolist()

fpgrowth_runtimes = [dataframes[key]['FP-Growth_Runtime (s)'].tolist() for key in dataframes]
fpgrowth_avg_runtime = np.mean(fpgrowth_runtimes, axis=0).tolist()

eclat_runtimes = [dataframes[key]['Eclat_Runtime (s)'].tolist() for key in dataframes]
eclat_avg_runtime = np.mean(eclat_runtimes, axis=0).tolist()

apriori_memories = [dataframes[key]['Apriori_Memory (MB)'].tolist() for key in dataframes]
apriori_avg_memory = np.mean(apriori_memories, axis=0).tolist()

fpgrowth_memories = [dataframes[key]['FP-Growth_Memory (MB)'].tolist() for key in dataframes]
fpgrowth_avg_memory = np.mean(fpgrowth_memories, axis=0).tolist()

eclat_memories = [dataframes[key]['Eclat_Memory (MB)'].tolist() for key in dataframes]
eclat_avg_memory = np.mean(eclat_memories, axis=0).tolist()

apriori_itemsets = [dataframes[key]['Apriori_Frequent_Itemsets'].tolist() for key in dataframes]
apriori_avg_itemsets = np.mean(apriori_itemsets, axis=0).tolist()

fpgrowth_itemsets = [dataframes[key]['FP-Growth_Frequent_Itemsets'].tolist() for key in dataframes]
fpgrowth_avg_itemsets = np.mean(fpgrowth_itemsets, axis=0).tolist()

eclat_itemsets = [dataframes[key]['Eclat_Frequent_Itemsets'].tolist() for key in dataframes]
eclat_avg_itemsets = np.mean(eclat_itemsets, axis=0).tolist()

apriori_rules = [dataframes[key]['Apriori_Rules'].tolist() for key in dataframes]
apriori_avg_rules = np.mean(apriori_rules, axis=0).tolist()

fpgrowth_rules = [dataframes[key]['FP-Growth_Rules'].tolist() for key in dataframes]
fpgrowth_avg_rules = np.mean(fpgrowth_rules, axis=0).tolist()

eclat_rules = [dataframes[key]['Eclat_Rules'].tolist() for key in dataframes]
eclat_avg_rules = np.mean(eclat_rules, axis=0).tolist()

# Create a 1x4 grid of subplots
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

# Plot 1: Runtime Performance
axs[0].plot(min_support, gpar_rbf_runtime, 'k-o', label='GPAR (RBF)')
axs[0].plot(min_support, gpar_shifted_rbf_runtime, 'r-s', label='GPAR (Shifted RBF)')
axs[0].plot(min_support, gpar_neural_net_runtime, 'm-d', label='GPAR (Neural Net)')
axs[0].plot(min_support, gpar_ntk_runtime, 'r-^', label='GPAR (NTK)')
axs[0].plot(min_support, apriori_avg_runtime, 'b--x', label='Apriori (AVG)')
axs[0].plot(min_support, fpgrowth_avg_runtime, 'y--^', label='FP-Growth (AVG)')
axs[0].plot(min_support, eclat_avg_runtime, 'g--s', label='Eclat (AVG)')
axs[0].set_title('Runtime Performance')
axs[0].set_xlabel('Min Support Threshold')
axs[0].set_ylabel('Runtime (seconds)')
axs[0].grid(True)

# Plot 2: Memory Usage
axs[1].plot(min_support, gpar_rbf_memory, 'k-o')
axs[1].plot(min_support, gpar_shifted_rbf_memory, 'r-s')
axs[1].plot(min_support, gpar_neural_net_memory, 'm-d')
axs[1].plot(min_support, gpar_ntk_memory, 'r-^')
axs[1].plot(min_support, apriori_avg_memory, 'b--x')
axs[1].plot(min_support, fpgrowth_avg_memory, 'y--^')
axs[1].plot(min_support, eclat_avg_memory, 'g--s')
axs[1].set_title('Memory Usage')
axs[1].set_xlabel('Min Support Threshold')
axs[1].set_ylabel('Memory Usage (MB)')
axs[1].grid(True)

# Plot 3: Number of Frequent Itemsets
axs[2].plot(min_support, gpar_rbf_itemsets, 'k-o')
axs[2].plot(min_support, gpar_shifted_rbf_itemsets, 'r-s')
axs[2].plot(min_support, gpar_neural_net_itemsets, 'm-d')
axs[2].plot(min_support, gpar_ntk_itemsets, 'r-^')
axs[2].plot(min_support, apriori_avg_itemsets, 'b--x')
axs[2].plot(min_support, fpgrowth_avg_itemsets, 'y--^')
axs[2].plot(min_support, eclat_avg_itemsets, 'g--s')
axs[2].set_title('Number of Frequent Itemsets')
axs[2].set_xlabel('Min Support Threshold')
axs[2].grid(True)

# Plot 4: Number of Rules
axs[3].plot(min_support, gpar_rbf_rules, 'k-o')
axs[3].plot(min_support, gpar_shifted_rbf_rules, 'r-s')
axs[3].plot(min_support, gpar_neural_net_rules, 'm-d')
axs[3].plot(min_support, gpar_ntk_rules, 'r-^')
axs[3].plot(min_support, apriori_avg_rules, 'b--x')
axs[3].plot(min_support, fpgrowth_avg_rules, 'y--^')
axs[3].plot(min_support, eclat_avg_rules, 'g--s')
axs[3].set_title('Number of Rules Generated')
axs[3].set_xlabel('Min Support Threshold')
axs[3].grid(True)

# Adjust layout to make space for the shared legend
plt.tight_layout(rect=[0, 0.15, 1, 1])

# Create a shared legend below the plots in a row
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=7, bbox_to_anchor=(0.5, 0))

# Save the figure
plt.savefig('performance_plots.png')
plt.close()
