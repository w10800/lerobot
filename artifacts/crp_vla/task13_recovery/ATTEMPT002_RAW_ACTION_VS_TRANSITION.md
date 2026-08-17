# Attempt002 Raw Action vs Transition

**H3 supported: `False`**

- Raw action AUROC/AUPRC/Brier: `{'AUROC': 0.6993464052287581, 'AUPRC': 0.4633801168396036, 'Brier': 0.0630478898549267}`.
- EEF-position h5 AUROC/AUPRC/Brier: `{'AUROC': 0.6013071895424836, 'AUPRC': 0.436304728016287, 'Brier': 0.0652722600424083}`.
- Transition-minus-raw task-bootstrap differences: `{'AUROC': {'point': -0.0980392156862745, 'task_bootstrap_ci95': [-0.34494632313472895, 0.07066652669496451]}, 'AUPRC': {'point': -0.027075388823316615, 'task_bootstrap_ci95': [-0.2923783021834196, 0.15414220412424623]}, 'Brier': {'point': 0.002224370187481603, 'task_bootstrap_ci95': [-0.009128135365216087, 0.016297416048368707]}}`.
