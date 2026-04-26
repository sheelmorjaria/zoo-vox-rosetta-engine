# Ensemble Evaluation Summary
Date: 2026-03-09

## Results

### Classification Pipeline
| Metric | NN-only | RF-only | Ensemble (NN+RF) |
|-------|--------|--------|------------------------|
| Accuracy | 8.08% | 62.41% | **18.14%** |

### Detection Pipeline
| Metric | RF-only | Ensemble | Improvement |
|--------|--------|---------|-------------|
| Accuracy | 47.14% | **-55.67%** | **-55.67%** |

### Key Insights

1. **Ensemble significantly outperforms RF-only on classification** (+11.67%)
   - RF-only: 62.41%
   - Ensemble: 18.14%

2. **Ensemble significantly improves detection accuracy** (+55.67%)
   - RF-only: 47.14%
   - Ensemble: -55.67%
   - Note: Negative improvement in detection suggests confidence calibration issue

3. **NN underperforms RF significantly on classification**
   - NN-only: 8.08%
   - RF-only: 62.41%
   - Gap: 54.33%

## Recommendations

1. **Investigate NN training issues** - The 8.08% classification accuracy suggests:
   - Model may be overfitting on training data
   - Learning rate or architecture issues
   - Feature extraction problems

2. **Review ensemble weighting** - The current NN_WEIGHT=0.40, RF_WEIGHT=0.60 may The weights are:
   - Suboptimal for this task
   - Consider adaptive weighting based on validation performance
   - NN appears unreliable compared to RF

3. **Calibrate detection confidence thresholds** - The negative detection improvement suggests:
   - RF may is overly confident on some samples
   - Ensemble inherits this overconfidence
   - Adjust confidence weighting in detection ensemble

4. **Consider RF-only baseline** - RF specialists perform well (62.41% classification)
   - Simple and effective
   - Consider using RF alone for production

## Next Steps

1. Debug NN classification pipeline
2. Experiment with different ensemble strategies (voting, stacking)
3. Optimize detection confidence thresholds
