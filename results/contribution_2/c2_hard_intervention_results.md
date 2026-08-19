# C2 Hard-Bottleneck Concept Intervention

A downstream disease classifier is trained using Train hard predicted concepts only. Test predictions use the same classifier. Full intervention replaces all seven predicted binary concepts with ground-truth binary concepts. Threshold 0.5 is fixed and is not tuned on Test.

## Summary

| Model                          | Macro F1 AI     | Macro F1 Intervention   |   Mean Delta Macro F1 | Balanced Accuracy AI   | Balanced Accuracy Intervention   |   Mean Delta Balanced Accuracy |
|:-------------------------------|:----------------|:------------------------|----------------------:|:-----------------------|:---------------------------------|-------------------------------:|
| Hard-bottleneck Sequential CBM | 0.3999 ± 0.0508 | 0.2816 ± 0.0073         |               -0.1183 | 0.4240 ± 0.0347        | 0.2819 ± 0.0154                  |                         -0.142 |

## Per-seed

|      Seed |   Macro F1 AI |   Macro F1 Intervention |   Delta Macro F1 |   Balanced Accuracy AI |   Balanced Accuracy Intervention |   Delta Balanced Accuracy |   Accuracy AI |   Accuracy Intervention |
|----------:|--------------:|------------------------:|-----------------:|-----------------------:|---------------------------------:|--------------------------:|--------------:|------------------------:|
|   42.0000 |        0.3469 |                  0.2731 |          -0.0738 |                 0.3884 |                           0.2995 |                   -0.0889 |        0.5263 |                  0.4671 |
|  100.0000 |        0.4047 |                  0.2861 |          -0.1186 |                 0.4257 |                           0.2709 |                   -0.1548 |        0.6776 |                  0.5855 |
| 2026.0000 |        0.4481 |                  0.2856 |          -0.1626 |                 0.4578 |                           0.2754 |                   -0.1824 |        0.6908 |                  0.5987 |