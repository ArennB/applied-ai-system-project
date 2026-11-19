# PawPal+ task model: evaluation

Dataset: `data/task_dataset.csv`, 219 labelled tasks. Hold-out split: 175 train / 44 test (stratified by priority). Cross-validation: 5 folds over all rows.

| Target | Hold-out accuracy | Majority baseline | 5-fold CV accuracy | CV baseline |
|--------|-------------------|-------------------|--------------------|-------------|
| priority | 64% | 50% (always "high") | 64% ± 8% | 51% |
| minutes | 59% | 39% (always "5") | 52% ± 2% | 33% |
| frequency | 82% | 75% (always "daily") | 79% ± 4% | 66% |

Duration on the hold-out set: mean absolute error 9.1 min; 75% of predictions within one bucket (5, 10, 15, 20, 30, 45, 60, 90 min).

## Confidence vs accuracy (hold-out set)

Each prediction comes with a confidence (the probability of the chosen label). If confidence is meaningful, predictions at or above 70% (the bar the app uses before the model second-guesses the agent) should be right more often than the rest.

| Target | Mean confidence | Accuracy when ≥ 70% sure | Accuracy when < 70% sure |
|--------|-----------------|------------------|------------------|
| priority | 0.73 | 72% (18/25) | 53% (10/19) |
| minutes | 0.53 | 78% (7/9) | 54% (19/35) |
| frequency | 0.81 | 85% (28/33) | 73% (8/11) |

## Per-class results (hold-out set)

### priority

```
              precision    recall  f1-score   support

        high       0.79      0.68      0.73        22
         low       0.50      0.29      0.36         7
      medium       0.52      0.73      0.61        15

    accuracy                           0.64        44
   macro avg       0.60      0.57      0.57        44
weighted avg       0.65      0.64      0.63        44
```

### minutes

```
              precision    recall  f1-score   support

          10       0.69      0.75      0.72        12
          15       0.29      0.67      0.40         3
          20       1.00      0.33      0.50         3
          30       0.29      0.67      0.40         3
          45       0.00      0.00      0.00         2
           5       0.86      0.71      0.77        17
          60       0.00      0.00      0.00         2
          90       0.00      0.00      0.00         2

    accuracy                           0.59        44
   macro avg       0.39      0.39      0.35        44
weighted avg       0.63      0.59      0.58        44
```

### frequency

```
              precision    recall  f1-score   support

       daily       0.96      0.79      0.87        33
        once       0.83      0.83      0.83         6
      weekly       0.45      1.00      0.62         5

    accuracy                           0.82        44
   macro avg       0.75      0.87      0.78        44
weighted avg       0.89      0.82      0.83        44
```

## Caveats

- The dataset is small and its labels were written with AI assistance. The model learns whatever the labels say, so review them before trusting it on new kinds of task.
- With ~44 test rows, one more or fewer correct prediction moves hold-out accuracy by about 2 points; the cross-validation figure is the steadier one.

Regenerate with `python -m task_model`.
