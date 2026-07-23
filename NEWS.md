# DirichletRF 0.2.0

## Breaking changes

* The class of the object returned by `DirichletRF()` has been renamed from
  `dirichlet_forest` to `DirichletRF`, and the S3 methods have been renamed
  accordingly (`predict.dirichlet_forest()` is now `predict.DirichletRF()`,
  and likewise for the `print()` method). Calls of the form
  `predict(forest, newdata)` are unaffected, but code that tests the class
  directly (for example `inherits(x, "dirichlet_forest")`) or that calls a
  method by its full name must be updated. Forest objects saved from
  version 0.1.0 will not dispatch to the new methods.

* Trees are now grown on resampled data by default. Version 0.1.0 grew every
  tree on all `n` observations; `DirichletRF()` now defaults to
  `replace = TRUE` with `sample.fraction = 0.632`. Fitted forests therefore
  differ from those produced by 0.1.0 even for the same `seed`. The previous
  behaviour is recovered with
  `replace = FALSE, sample.fraction = 1, compute.oob = FALSE`.

* `$fitted` and `$residuals` are now computed from out-of-bag predictions
  rather than from in-bag training predictions, so each observation is
  predicted only by trees it was not used to grow. These components are
  `NULL` when `compute.oob = FALSE`, in which case in-bag values can still be
  obtained with `predict(forest, X)`.

## New features

* Out-of-bag estimation is now supported, controlled by the new `replace`,
  `sample.fraction`, and `compute.oob` arguments. `$oob` contains the OOB
  prediction matrix, the OOB Dirichlet parameter estimates, and a scalar OOB
  MSE. The full prediction matrix is returned so that alternative
  compositional error measures, such as the Aitchison distance, can be
  applied directly.

* New `distributional` argument. When `TRUE`, leaf sample indices are
  retained, allowing the forest to estimate the full conditional
  distribution of the response as a weighted distribution over the training
  data, at the cost of higher memory usage.

* New function `predict_weights()` returns the proximity weights of new
  observations over the training sample. These weights can be used to
  estimate conditional means, quantiles, and other functionals. An OOB
  proximity matrix is also available via `$oob$weights` when both
  `distributional = TRUE` and `compute.oob = TRUE`.

* New function `sample_conditional()` draws compositional observations from
  the forest-weighted conditional distribution at a single covariate point.

* Impurity-based feature importance is now returned in `$importance` as raw
  gain, normalised gain, and split counts, and is summarised by the new
  `importance()` method.

* New function `permutation_importance()` computes permutation-based
  out-of-bag variable importance, with a scaled variant accounting for
  tree-to-tree variability. Aitchison distance (the default), mean squared
  error, and KL divergence are supported as loss functions.

## Documentation


* The package reference has been updated to the published book chapter
  (Masoumifard, van der Westhuizen and Gardner-Lubbe, 2026,
  ISBN:9781032903910).

## Internal

* Removed unused internal cluster-setup helpers. Parallel tree building is
  handled entirely by OpenMP within the compiled code and is controlled by
  `num.cores`.
