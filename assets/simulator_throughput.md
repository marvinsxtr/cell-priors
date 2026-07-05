# Simulator throughput (networks / second)

End-to-end JAX priors (sample GRN + kinetics + simulate), vmapped batch of 8, 64 cells/network. Higher is better.

| simulator | platform | 20 genes | 50 genes | 100 genes |
|---|---|---|---|---|
| SERGIO (Hill SDE) | cpu | 311 | 83 | 53 |
| grn-paper (sigmoid SDE) | cpu | 6 | 4 | 3 |
| BoolODE (mRNA+protein) | cpu | 2 | 2 | 1 |
