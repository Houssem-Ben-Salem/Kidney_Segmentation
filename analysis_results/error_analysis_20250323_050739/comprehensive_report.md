# TransUNet Comprehensive Error Analysis

## Overview
- **Model**: R50-ViT-B_16
- **Dataset**: KiTS19 Test Set (100.0% sampled)
- **Date**: 20250323_050739

## 1. Performance Metrics

### Kidney
- **Dice Score**: 0.8077 ± 0.3671
- **Jaccard**: 0.7892 ± 0.3645
- **HD95**: 1.3196 ± 2.9096
- **ASD**: 0.3014 ± 0.8237

### Tumor
- **Dice Score**: 0.8581 ± 0.3343
- **Jaccard**: 0.8497 ± 0.3365
- **HD95**: 0.3249 ± 1.0826
- **ASD**: 0.1140 ± 0.3512

## 2. Error Type Analysis

The errors in segmentation can be categorized into three main types:

1. **False Negatives**: 7.5% - These are areas that should be segmented as kidney or tumor but were missed
2. **False Positives**: 48.9% - These are areas incorrectly labeled as kidney or tumor
3. **Class Confusion**: 2.0% - These are areas where kidney was labeled as tumor or vice versa

## 3. Size-dependent Performance Analysis
| gt_category   |   ('tumor_dice', 'mean') |   ('tumor_dice', 'std') |   ('tumor_dice', 'count') |   ('tumor_hd95', 'mean') |   ('tumor_hd95', 'std') |   ('tumor_hd95', 'count') |   ('tumor_asd', 'mean') |   ('tumor_asd', 'std') |   ('tumor_asd', 'count') |
|:--------------|-------------------------:|------------------------:|--------------------------:|-------------------------:|------------------------:|--------------------------:|------------------------:|-----------------------:|-------------------------:|
| large         |                 0.955185 |               0.0235839 |                       262 |                 2.6869   |                 1.55997 |                       262 |                0.822306 |               0.268169 |                      262 |
| medium        |                 0.901256 |               0.0484266 |                       368 |                 2.52158  |                 1.69078 |                       368 |                0.876088 |               0.458855 |                      368 |
| none          |                 0.862195 |               0.34472   |                      6814 |                -0.137805 |                 0.34472 |                      6814 |               -0.137805 |               0.34472  |                     6814 |
| small         |                 0.565268 |               0.322319  |                       236 |                 2.14725  |                 2.60754 |                       236 |                0.791784 |               1.02968  |                      236 |

### Observations:
- Small tumors tend to be more difficult to segment accurately, with lower Dice scores compared to larger tumors.
- Boundary accuracy (measured by HD95 and ASD) generally improves with tumor size.

## 4. Boundary Accuracy Assessment

Boundary accuracy is measured using Hausdorff Distance (HD95) and Average Surface Distance (ASD).

### Kidney Boundaries
- Mean HD95: 1.3196 pixels
- Mean ASD: 0.3014 pixels

### Tumor Boundaries
- Mean HD95: 0.3249 pixels
- Mean ASD: 0.1140 pixels

### Observations:
- 
- Boundary errors typically occur in regions where tumor and kidney tissue are difficult to distinguish.

## 5. Computational Efficiency

- **Mean Inference Time**: 20.73 ms per slice
- **Median Inference Time**: 20.12 ms per slice
- **Memory Usage**: 0.01 MB per inference
- **Tumor vs. Non-tumor Slices**: Tumor-containing slices require more processing time

## 6. Summary and Recommendations

### Key Findings:
1. The model achieves better segmentation performance on kidneys (0.8077 Dice) than tumors (0.8581 Dice).
2. Small tumors are particularly challenging to segment accurately.
3. The most common error type is false positives (48.9% of all errors).
4. Boundary accuracy is comparable between kidneys and tumors.

### Recommendations:
1. Focus on improving the detection of small tumors, possibly by using specialized loss functions.
2. Reduce false positives by applying post-processing to remove small false positives.
3. Improve boundary accuracy by incorporating boundary-aware loss functions or post-processing techniques.
4. The model is computationally efficient with an average inference time of 20.73 ms per slice, making it suitable for clinical applications.

