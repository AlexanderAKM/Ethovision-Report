import numpy as np
from scipy.stats import mannwhitneyu

# Values for CTR (first 8) and FTX (last 8)
ctr_values = np.array([2.73607, 4.07074, 1.06773, 0.266934, 1.5015, 4.27094, 4.5045, 9.94328])
ftx_values = np.array([0.533867, 0, 3.53687, 0, 0, 0, 0, 0])

# Perform Mann-Whitney U Test
stat, p_value = mannwhitneyu(ctr_values, ftx_values)

print(stat, p_value)
