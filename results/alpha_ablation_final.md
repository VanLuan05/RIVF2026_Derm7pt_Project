# Final Alpha Ablation on Validation Set

Alpha is selected using mean Validation Disease Macro-F1 across three seeds. Concept Macro-F1 is used as a secondary criterion. The Test set is not used for selection.

|   Alpha |   Seed 42 Disease F1 |   Seed 100 Disease F1 |   Seed 2026 Disease F1 |   Mean Disease F1 |   SD Disease F1 |   Mean Concept F1 |   SD Concept F1 |
|--------:|---------------------:|----------------------:|-----------------------:|------------------:|----------------:|------------------:|----------------:|
|  2.0000 |               0.5344 |                0.6082 |                 0.5716 |            0.5714 |          0.0369 |            0.3885 |          0.0105 |
|  3.0000 |               0.5400 |                0.5390 |                 0.6073 |            0.5621 |          0.0392 |            0.3703 |          0.0134 |

**Selected alpha: 2.0**
