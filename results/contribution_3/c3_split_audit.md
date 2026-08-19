# Split and Data Audit

The split is audited at `case_num` group level. Do not describe it as patient-level unless Derm7pt documentation independently confirms that `case_num` uniquely identifies patients.

## Class distribution

| Split       | Class                |   Count |   Percent |
|:------------|:---------------------|--------:|----------:|
| Train       | Basal Cell Carcinoma |      28 |      4.63 |
| Train       | Melanoma             |     153 |     25.29 |
| Train       | Miscellaneous        |      49 |      8.10 |
| Train       | Nevus                |     346 |     57.19 |
| Train       | Seborrheic Keratosis |      29 |      4.79 |
| Validation  | Basal Cell Carcinoma |       7 |      4.61 |
| Validation  | Melanoma             |      34 |     22.37 |
| Validation  | Miscellaneous        |      20 |     13.16 |
| Validation  | Nevus                |      86 |     56.58 |
| Validation  | Seborrheic Keratosis |       5 |      3.29 |
| Calibration | Basal Cell Carcinoma |       2 |      1.96 |
| Calibration | Melanoma             |      30 |     29.41 |
| Calibration | Miscellaneous        |      12 |     11.76 |
| Calibration | Nevus                |      53 |     51.96 |
| Calibration | Seborrheic Keratosis |       5 |      4.90 |
| Test        | Basal Cell Carcinoma |       5 |      3.29 |
| Test        | Melanoma             |      35 |     23.03 |
| Test        | Miscellaneous        |      16 |     10.53 |
| Test        | Nevus                |      90 |     59.21 |
| Test        | Seborrheic Keratosis |       6 |      3.95 |

## case_num overlap

| Split A     | Split B     |   Overlapping case_num |
|:------------|:------------|-----------------------:|
| Train       | Validation  |                      0 |
| Train       | Calibration |                      0 |
| Train       | Test        |                      0 |
| Validation  | Calibration |                      0 |
| Validation  | Test        |                      0 |
| Calibration | Test        |                      0 |

## Group summary

| Split       |   Samples |   Unique case_num |   Unknown/missing case_num rows |
|:------------|----------:|------------------:|--------------------------------:|
| Train       |       605 |               605 |                               0 |
| Validation  |       152 |               152 |                               0 |
| Calibration |       102 |               102 |                               0 |
| Test        |       152 |               152 |                               0 |

## Image audit

Unique referenced image paths checked: 2022

No missing referenced images were detected.


## Final status

- PASS: no blocking split/data integrity issue detected.
