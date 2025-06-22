import matplotlib.pyplot as plt
import os

# Define output directory
output_dir = "~/RLAR/results"
os.makedirs(output_dir, exist_ok=True)

# Common data
min_support = [0.1, 0.2, 0.3, 0.4, 0.5]

# Data for Runtime Performance
rlar_runtime = [5337.17, 5803.99, 3451.94, 4046.33, 716.99]
gpar_rbf_runtime = [10.179, 8.448, 3.295, 0.215, 0.013]
gpar_shifted_rbf_runtime = [8.930, 8.697, 2.658, 0.174, 0.034]
gpar_neural_net_runtime = [0.133, 0.046, 0.017, 0.002, 0.005]
gpar_ntk_runtime = [0.449, 0.077, 0.030, 0.016, 0.003]
barm_runtime = [1.1821, 0.2314, 0.0734, 0.0481, 0.0451]
mab_arm_runtime = [1.6599, 1.6036, 1.6251, 1.6126, 0.9390]
apriori_runtime = [0.483, 0.505, 0.565, 0.570, 0.211]
fp_growth_runtime = [2.967, 2.734, 2.622, 2.700, 2.019]
eclat_runtime = [0.556, 0.555, 0.575, 0.558, 0.635]

# Data for Memory Usage
rlar_memory = [433.24, 1.25, 0.5, 1.25, 96.75]
gpar_rbf_memory = [0.0, 0.0, 0.0, 0.0, 0.0]
gpar_shifted_rbf_memory = [0.0, 0.0, 0.0, 0.0, 0.0]
gpar_neural_net_memory = [0.0, 0.0, 0.0, 0.0, 0.0]
gpar_ntk_memory = [0.0, 0.0, 0.0, 0.0, 0.0]
barm_memory = [0.0, 0.0, 0.0, 0.0, 0.0]
mab_arm_memory = [0.0, 13.625, 0.625, 6.375, 0.125]
apriori_memory = [0.0, 0.0, 0.0, 0.0, 0.0]
fp_growth_memory = [0.063, 0.0, 0.0, 0.0, 0.0]
eclat_memory = [18.149, 0.0, 1.438, 7.969, 0.156]

# Data for Number of Frequent Itemsets
rlar_items = [79, 97, 98, 148, 145]
gpar_rbf_items = [1013, 981, 533, 67, 1]
gpar_shifted_rbf_items = [1013, 982, 536, 65, 2]
gpar_neural_net_items = [140, 43, 9, 0, 0]
gpar_ntk_items = [305, 67, 11, 2, 0]
barm_items = [876, 385, 165, 45, 44]
mab_arm_items = [1000, 1000, 1000, 1000, 736]
apriori_items = [1023, 1023, 1023, 1023, 746]
fp_growth_items = [1023, 1023, 1023, 1023, 746]
eclat_items = [1033, 1033, 1033, 1033, 1033]

# Data for Number of Rules
rlar_rules = [1106, 1454, 1394, 2436, 2382]
gpar_rbf_rules = [54036, 51479, 19251, 782, 2]
gpar_shifted_rbf_rules = [54036, 51328, 16067, 530, 8]
gpar_neural_net_rules = [322, 59, 18, 0, 0]
gpar_ntk_rules = [1485, 159, 26, 4, 0]
barm_rules = [12029, 3024, 802, 90, 88]
mab_arm_rules = [50372, 50372, 50372, 50372, 20948]
apriori_rules = [57002, 57002, 57002, 57002, 20948]
fp_growth_rules = [57002, 57002, 57002, 57002, 20948]
eclat_rules = [57002, 57002, 57002, 57002, 57002]

# Create a 1x4 grid of subplots
fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=False)

# Plot 1: Runtime Performance
axs[0].plot(min_support, rlar_runtime, 'k-o', label='RLAR')
axs[0].plot(min_support, gpar_rbf_runtime, 'r-s', label='GPAR (RBF)')
axs[0].plot(min_support, gpar_shifted_rbf_runtime, 'm-d', label='GPAR (Shifted RBF)')
axs[0].plot(min_support, gpar_neural_net_runtime, 'r-^', label='GPAR (Neural Net)')
axs[0].plot(min_support, gpar_ntk_runtime, 'b--x', label='GPAR (NTK)')
axs[0].plot(min_support, barm_runtime, 'y--^', label='BARM')
axs[0].plot(min_support, mab_arm_runtime, 'g--s', label='MAB-ARM')
axs[0].plot(min_support, apriori_runtime, 'c-p', label='Apriori')
axs[0].plot(min_support, fp_growth_runtime, 'm-h', label='FP-Growth')
axs[0].plot(min_support, eclat_runtime, 'k-D', label='Eclat')
axs[0].set_title('Runtime Performance', fontsize=12)
axs[0].set_xlabel('Min Support Threshold', fontsize=10)
axs[0].set_ylabel('Runtime (seconds)', fontsize=10)
axs[0].grid(True)
axs[0].tick_params(axis='both', which='major', labelsize=8)

# Plot 2: Memory Usage
axs[1].plot(min_support, rlar_memory, 'k-o', label='RLAR')
axs[1].plot(min_support, gpar_rbf_memory, 'r-s', label='GPAR (RBF)')
axs[1].plot(min_support, gpar_shifted_rbf_memory, 'm-d', label='GPAR (Shifted RBF)')
axs[1].plot(min_support, gpar_neural_net_memory, 'r-^', label='GPAR (Neural Net)')
axs[1].plot(min_support, gpar_ntk_memory, 'b--x', label='GPAR (NTK)')
axs[1].plot(min_support, barm_memory, 'y--^', label='BARM')
axs[1].plot(min_support, mab_arm_memory, 'g--s', label='MAB-ARM')
axs[1].plot(min_support, apriori_memory, 'c-p', label='Apriori')
axs[1].plot(min_support, fp_growth_memory, 'm-h', label='FP-Growth')
axs[1].plot(min_support, eclat_memory, 'k-D', label='Eclat')
axs[1].set_title('Memory Usage', fontsize=12)
axs[1].set_xlabel('Min Support Threshold', fontsize=10)
axs[1].set_ylabel('Memory Usage (MB)', fontsize=10)
axs[1].grid(True)
axs[1].tick_params(axis='both', which='major', labelsize=8)

# Plot 3: Number of Frequent Itemsets
axs[2].plot(min_support, rlar_items, 'k-o', label='RLAR')
axs[2].plot(min_support, gpar_rbf_items, 'r-s', label='GPAR (RBF)')
axs[2].plot(min_support, gpar_shifted_rbf_items, 'm-d', label='GPAR (Shifted RBF)')
axs[2].plot(min_support, gpar_neural_net_items, 'r-^', label='GPAR (Neural Net)')
axs[2].plot(min_support, gpar_ntk_items, 'b--x', label='GPAR (NTK)')
axs[2].plot(min_support, barm_items, 'y--^', label='BARM')
axs[2].plot(min_support, mab_arm_items, 'g--s', label='MAB-ARM')
axs[2].plot(min_support, apriori_items, 'c-p', label='Apriori')
axs[2].plot(min_support, fp_growth_items, 'm-h', label='FP-Growth')
axs[2].plot(min_support, eclat_items, 'k-D', label='Eclat')
axs[2].set_title('Number of Frequent Itemsets', fontsize=12)
axs[2].set_xlabel('Min Support Threshold', fontsize=10)
axs[2].set_ylabel('Number of Frequent Itemsets', fontsize=10)
axs[2].grid(True)
axs[2].tick_params(axis='both', which='major', labelsize=8)

# Plot 4: Number of Rules
axs[3].plot(min_support, rlar_rules, 'k-o', label='RLAR')
axs[3].plot(min_support, gpar_rbf_rules, 'r-s', label='GPAR (RBF)')
axs[3].plot(min_support, gpar_shifted_rbf_rules, 'm-d', label='GPAR (Shifted RBF)')
axs[3].plot(min_support, gpar_neural_net_rules, 'r-^', label='GPAR (Neural Net)')
axs[3].plot(min_support, gpar_ntk_rules, 'b--x', label='GPAR (NTK)')
axs[3].plot(min_support, barm_rules, 'y--^', label='BARM')
axs[3].plot(min_support, mab_arm_rules, 'g--s', label='MAB-ARM')
axs[3].plot(min_support, apriori_rules, 'c-p', label='Apriori')
axs[3].plot(min_support, fp_growth_rules, 'm-h', label='FP-Growth')
axs[3].plot(min_support, eclat_rules, 'k-D', label='Eclat')
axs[3].set_title('Number of Rules Generated', fontsize=12)
axs[3].set_xlabel('Min Support Threshold', fontsize=10)
axs[3].set_ylabel('Number of Rules', fontsize=10)
axs[3].grid(True)
axs[3].tick_params(axis='both', which='major', labelsize=8)

# Adjust layout to make space for the shared legend
plt.tight_layout(rect=[0, 0.15, 1, 1])

# Create a shared legend below the plots in a row
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=10, bbox_to_anchor=(0.5, 0), frameon=False)

# Save the figure
plt.savefig(os.path.join(output_dir, 'performance_plots_row.png'))
plt.close()
