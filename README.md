## LOWESS with Gradient Boosting Implementation

[GradientBoostedLowess](https://github.com/XiongCynthia/GradientBoostedLowess/blob/main/GradientBoostedLowess.py) is a class for fitting and predicting data on a locally weighted scatterplot smoothing (LOWESS) model with gradient boosting.

It requires [Lowess](https://xiongcynthia.github.io/Lowess) and [RegressionTree](https://xiongcynthia.github.io/RegressionTree) to work.

### Usage

```python
from GradientBoostedLowess import GradientBoostedLowess
gb_lowess = GradientBoostedLowess()
gb_lowess.fit(x_train, y_train)
y_pred = gb_lowess.predict(x_test)
```

More example usages are included in [GradientBoostedLowess_examples.ipynb](https://github.com/XiongCynthia/GradientBoostedLowess/blob/main/GradientBoostedLowess_examples.ipynb).
