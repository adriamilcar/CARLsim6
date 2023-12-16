import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Read the phase data
phase_data = pd.read_csv("phaseData.csv", header=None)

# Plotting
plt.figure(figsize=(12, 6))
for i, neuron in enumerate(phase_data.columns[::-1]):
    weights_hist = np.ones_like(phase_data[neuron].dropna()) / len(phase_data[neuron].dropna())
    plt.hist(phase_data[neuron].dropna(), bins=np.arange(0, 2*np.pi+0.03, 0.03), label=f"neuron {i+1}", alpha=0.7, weights=weights_hist)

plt.xticks(np.arange(0, 5*np.pi/2, np.pi/2), [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/4$", r"$2\pi$"], fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Phase (rad)", size=22)
plt.ylabel("Probability", size=22)
plt.legend(fontsize=16, loc='upper right', frameon=False)
sns.despine()
plt.show()
