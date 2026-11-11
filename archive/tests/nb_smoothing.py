import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../src')
from emagpy import Problem

datadir = '.'

k = Problem()
k.createSurvey(datadir + '/coverCrop.csv')
k.surveys[0].df = k.surveys[0].df[:30] # truncate dataset

# Define beta values to compare
beta_values = [0, 0.1, 50]

fig, axs = plt.subplots(1, len(beta_values), sharex=True, sharey=True, figsize=(12, 3))

for i, beta in enumerate(beta_values):
    ax = axs[i]
    k.invert(beta=beta)
    k.showResults(ax=ax)
    ax.set_title(f'beta = {beta:.2f}')

plt.tight_layout()

# Save the figure
plt.savefig('beta_comparison.png', dpi=300, bbox_inches='tight')
plt.show()