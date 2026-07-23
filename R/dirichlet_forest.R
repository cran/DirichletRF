#' Build a Dirichlet Random Forest for Compositional Responses
#'
#' \code{DirichletRF} fits a random forest tailored to compositional
#' responses, i.e.\ non-negative vectors that sum to one and therefore
#' reside in the unit simplex. Each tree is grown by recursively
#' partitioning the covariate space using a \strong{Dirichlet
#' log-likelihood splitting criterion}: at every internal node the
#' candidate split that maximises the gain in Dirichlet log-likelihood
#' is selected. 
#'
#' \strong{Predictions.} The fitted forest produces two complementary
#' point-prediction surfaces: a \emph{mean-based} prediction (the
#' sample mean of training responses in the matched leaf) and a
#' \emph{parameter-based} prediction. The forest is also able to produce
#' \strong{full distributional (weight-based) predictions}: Trains a Distributional Random Forest which estimates the full conditional distribution \eqn{P(Y | X)}
#' for possibly multivariate response Y and predictors X. The conditional distribution estimate is represented
#' as a weighted distribution of the training data. The weights can be conveniently used in the downstream analysis
#' to estimate any quantity of interest \eqn{\tau(P(Y | X))}.
#' 
#' \strong{Out-of-bag (OOB) evaluation.} Each observation is predicted
#' exclusively by trees for which it was held out. The OOB prediction
#' matrix and scalar OOB MSE are returned in \code{$oob}; the full
#' prediction matrix is also available so users may apply any
#' alternative compositional error measure such as the Aitchison
#' distance.
#'
#' \strong{Feature importance.} Three complementary importance
#' measures are computed automatically:
#' \describe{
#'   \item{Gain (raw and normalised)}{Total Dirichlet log-likelihood
#'     gain accumulated over every split where a feature was chosen,
#'     summed across all trees. The normalised version sums to 1,
#'     facilitating comparison across forests.}
#'   \item{Split count}{Number of times a feature was selected as the
#'     best split variable across all internal nodes and all trees.}
#'   \item{Permutation importance}{Computed post-hoc via
#'     \code{\link{permutation_importance}}: the mean increase in OOB
#'     loss when a feature's values are randomly permuted within each
#'     tree's OOB sample, with a scaled (t-statistic-like) variant
#'     that accounts for tree-to-tree variability. Supports
#'     Aitchison distance, MSE, and KL divergence as loss functions.}
#' }
#'
#' The implementation delegates all tree-building to compiled C++ code
#' and uses \strong{OpenMP} for parallel construction of trees.
#' 
#' @param X A numeric (n x p) matrix of covariates. Note that the current
#'   version only allows numeric covariates. Users may use one-hot encoding
#'   to possibly include categorical covariates.
#' @param Y A numeric (n x k) matrix of compositional responses. Each row
#'   should sum to 1. That is, data should already be normalised if needed.
#' @param num.trees Number of trees grown in the forest. Default is 100.
#' @param max.depth Maximum depth of trees. Default is 10.
#' @param min.node.size Minimum size of observations in each tree leaf.
#'   Default is 5. Note that nodes with sizes smaller than
#'   \code{min.node.size} can occur.
#' @param mtry Number of covariates randomly selected as candidates at each
#'   split. Default is \code{sqrt(p)}, indicated by \code{-1}.
#' @param seed The seed of the C++ random number generator.
#' @param est.method Parameter estimation method for the Dirichlet
#'   distribution when splitting is done. Users may either use maximum
#'   likelihood (\code{"mle"}) or method of moments (\code{"mom"}).
#'   Default is \code{"mom"}.
#' @param distributional Logical. If \code{TRUE}, the forest stores the
#'   training indices at every leaf node, enabling distributional (weight-based)
#'   predictions at test time. For each test point, the forest computes a
#'   weighted empirical distribution over training observations -- weights are
#'   proportional to co-occurrence in the same leaf across trees -- and a new
#'   Dirichlet is fitted to that weighted distribution via \code{est.method}.
#'   This produces a full predictive distribution rather than a single point
#'   estimate, at the cost of higher memory usage since leaf sample indices are
#'   retained for all trees. Default is \code{FALSE}.
#' @param num.cores Number of OpenMP threads used for parallel tree building.
#'   The default is \code{-1} which uses all the cores on the system minus 1.
#'   Users may also specify \code{1} which means that the forest will be
#'   built sequentially.
#' @param replace Logical. If \code{TRUE} (default), each tree is grown on a
#'   bootstrap sample drawn with replacement. If \code{FALSE}, each tree is
#'   grown on a subsample drawn without replacement. When \code{replace = FALSE}
#'   and \code{sample.fraction = 1} (the default), every tree sees all \code{n}
#'   observations and tree diversity comes entirely from random feature
#'   subsetting controlled by \code{mtry}. When \code{replace = FALSE} and
#'   \code{sample.fraction < 1}, each tree sees a different random subset of
#'   the data, enabling out-of-bag estimation.
#' @param sample.fraction Numeric. Fraction of observations used to grow each
#'   tree, as a proportion of \code{n}. Default is \code{0.632}. When
#'   \code{replace = FALSE}, must be in \code{(0, 1]}; values greater than 1
#'   are not allowed since you cannot draw more unique observations than
#'   available. When \code{replace = TRUE}, values greater than 1 are allowed
#'   (e.g. \code{1.5} draws \code{1.5n} bootstrap observations), though
#'   values in \code{(0, 1]} are most common. A warning is issued when
#'   \code{sample.fraction < 0.1} regardless of \code{replace}, as trees
#'   grown on very few observations tend to be unreliable.
#' @param compute.oob Logical. If \code{TRUE}, computes out-of-bag (OOB)
#'   predictions after the forest is built, using only trees for which each
#'   observation was not in training. Both the OOB prediction matrix and a
#'   scalar MSE are returned via \code{$oob$predictions} and \code{$oob$mse}.
#'   Not available when \code{replace = FALSE} and \code{sample.fraction = 1}
#'   since no held-out observations exist. Default is \code{TRUE}.
#' 
#' @details
#' \strong{Out-of-Bag (OOB) Predictions}
#'
#' When \code{compute.oob = TRUE}, each observation is predicted by averaging
#' over only the trees for which it was out-of-bag. This requires
#' \code{replace = TRUE} or \code{replace = FALSE} with
#' \code{sample.fraction < 1}. The reported \code{$oob$mse} is the MSE
#' between OOB predictions and true responses, averaged over components and
#' OOB observations. Note that MSE is not universally accepted for
#' compositional data since it ignores the simplex geometry -- the Aitchison
#' distance, which operates in log-ratio space, is an alternative. The full
#' OOB prediction matrix \code{$oob$predictions} (n x k, with \code{NA} for
#' observations never out-of-bag) is returned so users can apply any
#' alternative error measure directly.
#'
#' @return A list of class \code{DirichletRF} which contains the
#'   following elements:
#' \describe{
#'   \item{\code{type}}{Parallelisation type used: \code{"openmp"} or
#'     \code{"sequential"}.}
#'   \item{\code{num.cores}}{Number of cores used.}
#'   \item{\code{num.trees}}{Total number of trees in the forest.}
#'   \item{\code{replace}}{Logical indicating whether bootstrap sampling
#'     was used.}
#'   \item{\code{sample.fraction}}{The fraction of observations used per
#'     tree.}
#'   \item{\code{compute.oob}}{Logical indicating whether OOB prediction was
#'     computed.}
#'   \item{\code{distributional}}{Logical indicating whether the forest was
#'     built in distributional mode (leaf sample indices retained).}
#'   \item{\code{est.method}}{The estimation method used (\code{"mom"} or
#'     \code{"mle"}).}
#'   \item{\code{Y_train}}{The training compositional response matrix.}
#'   \item{\code{fitted}}{A list of fitted values on the training data:
#'     \describe{
#'       \item{\code{alpha_hat}}{Estimated Dirichlet alpha parameters
#'         (n x k matrix).}
#'       \item{\code{mean_based}}{Mean-based fitted values (n x k matrix),
#'         derived from sample means at each leaf.}
#'       \item{\code{param_based}}{Parameter-based fitted values (n x k
#'         matrix), obtained by normalising \code{alpha_hat} so rows sum
#'         to 1.}
#'     }
#'   }
#'   \item{\code{residuals}}{A list of residuals (Y - fitted values):
#'     \describe{
#'       \item{\code{mean_based}}{Residuals from mean-based predictions.}
#'       \item{\code{param_based}}{Residuals from parameter-based
#'         predictions.}
#'     }
#'   }
#'   \item{\code{importance}}{A list of feature importance measures:
#'     \describe{
#'       \item{\code{gain}}{Raw total likelihood gain per feature, summed
#'         over all trees and all splits where the feature was selected.}
#'       \item{\code{gain_normalised}}{Gain divided by total gain across
#'         all features, summing to 1. Recommended for interpretation and
#'         comparison across forests.}
#'       \item{\code{count}}{Number of times each feature was selected as
#'         the best split variable across all trees and all internal
#'         nodes.}
#'     }
#'   }
#'   \item{\code{oob}}{A list of OOB results. All elements are \code{NA}
#'     or \code{NULL} when \code{compute.oob = FALSE}:
#'     \describe{
#'       \item{\code{mse}}{Scalar OOB mean squared error, averaged over
#'         all components and all observations that appeared OOB at least
#'         once.}
#'       \item{\code{predictions}}{An (n x k) matrix of OOB predictions.
#'         Rows corresponding to observations that never appeared OOB are
#'         \code{NA}.}
#'       \item{\code{alpha_predictions}}{An (n x k) matrix of OOB
#'         Dirichlet alpha parameter estimates, averaged over OOB trees.
#'         \code{NA} for observations never out-of-bag.}
#'       \item{\code{weights}}{An (n x n) matrix of OOB proximity weights.
#'         \code{weights[i, j]} is the average fraction of OOB trees in
#'         which observations \code{i} and \code{j} landed in the same
#'         leaf. Only available when both \code{distributional = TRUE} and
#'         \code{compute.oob = TRUE}; \code{NULL} otherwise. The matrix
#'         is generally asymmetric since row \code{i} is averaged only
#'         over trees where \code{i} was out-of-bag.}
#'     }
#'   }
#' }
#'
#' @examples
#' # -- Minimal example -------------------------------------------------------
#' set.seed(42)
#' n <- 50; p <- 2
#' X <- matrix(rnorm(n * p), n, p)
#' colnames(X) <- paste0("X", 1:p)
#' G <- matrix(rgamma(n * 3, shape = rep(c(2, 3, 4), each = n)), n, 3)
#' Y <- G / rowSums(G)
#'
#' # Hold out a test set for validation
#' ids   <- sample(1:50, 10)
#' X_trn <- X[-ids, ]
#' Y_trn <- Y[-ids, ]
#' X_tst <- X[ids, ]
#' Y_tst <- Y[ids, ]
#'
#' # Fit with default settings
#' forest <- DirichletRF(X_trn, Y_trn, num.cores = 1)
#' print(forest)
#'
#' # Feature importance
#' importance(forest)
#' permutation_importance(forest, X_trn)  # default loss is the Aitchison distance
#'
#' # Prediction on the held-out test set
#' pred <- predict(forest, X_tst)
#' pred$mean_predictions  # may be compared to Y_tst for validation
#'
#' \donttest{
#' # -- Distributional forest with OOB weight matrix ---------------------------
#' forest_dist <- DirichletRF(X_trn, Y_trn, num.trees = 100, num.cores = 1,
#'                            distributional = TRUE)
#'
#' # OOB weight matrix: n x n, W[i,j] = proximity of i to j via OOB trees
#' W <- forest_dist$oob$weights
#' dim(W)
#'
#' # Symmetrise if a symmetric proximity matrix is preferred
#' W_sym <- (W + t(W)) / 2
#'
#' # Weights for new observations
#' W_new <- predict_weights(forest_dist, X_tst)  # 10 x n_train
#'
#' # Use weights to obtain mean predictions
#' W_new %*% Y_trn[, 1]
#'
#' # Can also use weights to obtain quantiles
#' weighted_quantile <- function(y, w, probs) {
#'   ord <- order(y)
#'   y   <- y[ord]
#'   w   <- w[ord]
#'   cw  <- cumsum(w)
#'   sapply(probs, function(p) y[which(cw >= p)[1]])
#' }
#'
#' # quantiles for the first part of Y
#' q <- t(apply(W_new, 1, function(w)
#'   weighted_quantile(Y_trn[, 1], w, probs = c(0.025, 0.5, 0.975))))
#'}
#' @references
#' Masoumifard, K., van der Westhuizen, S., & Gardner-Lubbe, S. (2026).
#' Dirichlet random forest for predicting compositional data.
#' In A. Bekker, P. Nagar, J. Ferreira, B. Erasmus, & A. Ramoelo (Eds.),
#' Environmental Modelling with Contemporary Statistics: Learning,
#' Directionality, and Space-Time Dynamics.
#' Chapman & Hall/CRC. ISBN: 9781032903910.
#'
#' @seealso
#' \code{\link{predict.DirichletRF}} for point predictions on new data
#'   (call as \code{predict(forest, newdata)}, documented under
#'   \code{?predict.DirichletRF}).
#' \code{\link{print.DirichletRF}} for a summary of the fitted object
#'   (call as \code{print(forest)} or just \code{forest}).
#' \code{\link{sample_conditional}} for drawing compositional samples
#'   from the conditional predictive distribution (requires
#'   \code{distributional = TRUE}).
#' \code{\link{importance.DirichletRF}} for impurity-based (gain and
#'   count) feature importance.
#' \code{\link{permutation_importance}} for permutation-based OOB feature
#'   importance (requires \code{compute.oob = TRUE}).
#' \code{\link{predict_weights}} for proximity weights for new observations
#'   (requires \code{distributional = TRUE}).
#' 
#' @importFrom stats predict
#' @export
DirichletRF <- function(X, Y, num.trees = 100, max.depth = 10,
                        min.node.size = 5, mtry = -1, seed = 123,
                        est.method = "mom", distributional = FALSE,
                        num.cores = -1, replace = TRUE,
                        sample.fraction = 0.632, compute.oob = TRUE) {

  store_samples <- distributional

  # Input validation
  if (!is.matrix(X) || !is.matrix(Y)) stop("X and Y must be matrices")
  if (nrow(X) != nrow(Y)) stop("X and Y must have the same number of rows")

  # applies to both
  if (sample.fraction <= 0)
    stop("sample.fraction must be positive")

  # only meaningful for replace = FALSE
  if (!replace && sample.fraction > 1)
    stop("sample.fraction must be in (0, 1] when replace = FALSE")

  if (sample.fraction < 0.1)
    warning("sample.fraction < 0.1: trees will be grown on very few observations ",
          "(this applies regardless of replace = ", replace, ")")

  # OOB requires held-out observations
  if (compute.oob && !replace && sample.fraction == 1.0)
    stop("OOB is not available when replace = FALSE and sample.fraction = 1")

  # Resolve num.cores
  if (num.cores == -1) {
    num.cores <- if (nzchar(Sys.getenv("_R_CHECK_LIMIT_CORES_"))) {
      2L   # CRAN check farm: never use more than two
    } else {
      max(1L, parallel::detectCores() - 1L)
    }
  }
  num.cores <- max(1L, as.integer(num.cores))
  
  par_type <- if (num.cores == 1L) "sequential" else "openmp"
  message("Building ", par_type, " forest with ", num.cores,
          " thread(s) for ", num.trees, " trees")

  # All parallelism handled inside C++ via OpenMP
  forest <- DirichletForest(X, Y,
                            B               = num.trees,
                            d_max           = max.depth,
                            n_min           = min.node.size,
                            m_try           = mtry,
                            seed            = seed,
                            method          = est.method,
                            store_samples   = store_samples,
                            num_cores       = num.cores,
                            replace         = replace,
                            sample_fraction = sample.fraction,
                            compute_oob     = compute.oob)
  # OOB weight matrix -- only when both distributional and OOB are requested
  oob_weights <- NULL
  if (distributional && compute.oob) {
    message("Computing OOB weight matrix...")
    oob_weights <- OOBWeightMatrix(forest, X)
  }
  # Extract feature importance from the C++ forest object
  imp_gain  <- forest$importance_gain   # sum of likelihood gains per feature
  imp_count <- forest$importance_count  # number of times selected for a split

  # Normalise gain importance to sum to 1 (makes forests comparable)
  imp_gain_norm <- if (sum(imp_gain) > 0) imp_gain / sum(imp_gain)
                   else rep(0, length(imp_gain))

  # Attach feature names if X has column names
  feat_names <- if (!is.null(colnames(X))) colnames(X)
                else paste0("X", seq_len(ncol(X)))
  names(imp_gain)      <- feat_names
  names(imp_gain_norm) <- feat_names
  names(imp_count)     <- feat_names

  result <- list(
    type            = par_type,
    forest          = forest,
    num.cores       = num.cores,
    num.trees       = num.trees,
    replace         = replace,
    sample.fraction = sample.fraction,
    compute.oob     = compute.oob,
    distributional  = distributional,
    est.method      = est.method,
    Y_train         = Y,
    importance = list(
      gain            = imp_gain,
      gain_normalised = imp_gain_norm,
      count           = imp_count
    ),
    oob = list(
      mse               = forest$oob_mse,
      predictions       = forest$oob_predictions,
      alpha_predictions = forest$oob_alpha_predictions,
      indices           = forest$oob_indices,
      weights           = oob_weights   
    )
  )
  class(result) <- c("DirichletRF", "list")

  # -- Fitted values and residuals ------------------------------------------
  # Only populated when compute.oob = TRUE, so each observation is predicted
  # exclusively by trees it was NOT trained on (true OOB estimates).
  # When OOB is unavailable, fitted and residuals are left NULL -- the user
  # can call predict(forest, X) themselves if in-bag values are acceptable.
  if (compute.oob) {
    message("Computing fitted values and residuals from OOB predictions...")
    oob_means  <- forest$oob_predictions        # n x k, NA where never OOB
    oob_alphas <- forest$oob_alpha_predictions  # n x k, NA where never OOB
    alpha_norm <- oob_alphas / rowSums(oob_alphas)

    result$fitted <- list(
      alpha_hat   = oob_alphas,
      mean_based  = oob_means,
      param_based = alpha_norm
    )
    result$residuals <- list(
      mean_based  = Y - oob_means,
      param_based = Y - alpha_norm
    )
  } else {
        message(
          "compute.oob = FALSE: $fitted and $residuals are NULL.\n",
          "  - To obtain OOB-based fitted values, refit with compute.oob = TRUE\n",
          "    (requires replace = TRUE or sample.fraction < 1).\n"
        )
        result$fitted    <- NULL
        result$residuals <- NULL
      }


  return(result)
}


#' Custom Print Method for DirichletRF Objects
#'
#' Suppresses the display of large data matrices (Y_train, fitted,
#' residuals) when the object is printed, while keeping them accessible
#' via \code{$}.
#'
#' @param x A \code{DirichletRF} object.
#' @param ... Further arguments passed to or from other methods.
#'
#' @return Invisibly returns \code{x}, the \code{DirichletRF} object
#'   unchanged. Called primarily for its side effect of printing a summary
#'   of the model to the console.
#' @seealso
#' \code{\link{predict.DirichletRF}} for point predictions on new data
#'   (call as \code{predict(forest, newdata)}, documented under
#'   \code{?predict.DirichletRF}).
#' \code{\link{sample_conditional}} for drawing compositional samples
#'   from the conditional predictive distribution (requires
#'   \code{distributional = TRUE}).
#' \code{\link{importance.DirichletRF}} for impurity-based (gain and
#'   count) feature importance.
#' \code{\link{permutation_importance}} for permutation-based OOB feature
#'   importance (requires \code{compute.oob = TRUE}).
#' \code{\link{predict_weights}} for proximity weights for new observations
#'   (requires \code{distributional = TRUE}).
#' 
#' @rdname print.DirichletRF
#' @aliases print
#' @export
print.DirichletRF <- function(x, ...) {

  N <- nrow(x$Y_train)
  K <- ncol(x$Y_train)
  train_info <- if (!is.null(N) && !is.null(K))
    paste0(N, " observations (n) x ", K, " components (k)")
  else
    "unknown"

  oob_str <- if (is.na(x$oob$mse)) "not computed" else round(x$oob$mse, 6)
  samp_str <- paste0(if (x$replace) "with replacement" else "without replacement",
                     ", fraction = ", x$sample.fraction)
  dist_str <- if (isTRUE(x$distributional)) "yes" else "no"

  cat(
    "============================================\n",
    "Dirichlet Forest Model\n",
    "============================================\n",
    " Type:           ", x$type, "\n",
    " Total Trees:    ", x$num.trees, "\n",
    " Cores Used:     ", x$num.cores, "\n",
    " Sampling:       ", samp_str, "\n",
    " Est. Method:    ", x$est.method, "\n",
    " Distributional: ", dist_str, "\n",
    " Training Data:  ", train_info, "\n",
    " OOB MSE:        ", oob_str, "\n",
    "--------------------------------------------\n",
    " Note: Large data structures (fitted values,\n",
    "       residuals) are suppressed.\n",
    "\n Access via:\n",
    "   $Y_train\n",
    "   $fitted$alpha_hat\n",
    "   $fitted$mean_based\n",
    "   $fitted$param_based\n",
    "   $residuals$mean_based\n",
    "   $residuals$param_based\n",
    "   $importance$gain\n",
    "   $importance$gain_normalised\n",
    "   $importance$count\n",
    "   $oob$mse\n",
    "   $oob$predictions\n",
    "   $oob$alpha_predictions\n",
    "   $oob$weights\n",
    " Note: $fitted and $residuals are NULL when\n",
    "       compute.oob = FALSE.\n",
    " Use importance(forest) for a summary table.\n",
    "============================================\n",
    sep = ""
  )

  invisible(x)
}




#' Predict with a Dirichlet Forest
#'
#' Makes predictions using a fitted \code{DirichletRF} object returned
#' by \code{\link{DirichletRF}}.
#'
#' @param object A \code{DirichletRF} object.
#' @param newdata A numeric matrix of new covariates (n_new x p).
#' @param ... Currently unused.
#'
#' @return A list with the following elements:
#' \describe{
#'   \item{\code{alpha_predictions}}{Estimated Dirichlet alpha parameters
#'     for each new observation (n_new x k matrix).}
#'   \item{\code{mean_predictions}}{Mean-based compositional predictions
#'     (n_new x k matrix).}
#' }
#'
#' @seealso
#' \code{\link{print.DirichletRF}} for a summary of the fitted object
#'   (call as \code{print(forest)} or just \code{forest}).
#' \code{\link{sample_conditional}} for drawing compositional samples
#'   from the conditional predictive distribution (requires
#'   \code{distributional = TRUE}).
#' \code{\link{importance.DirichletRF}} for impurity-based (gain and
#'   count) feature importance.
#' \code{\link{permutation_importance}} for permutation-based OOB feature
#'   importance (requires \code{compute.oob = TRUE}).
#' \code{\link{predict_weights}} for proximity weights for new observations
#'   (requires \code{distributional = TRUE}).
#' @examples
#' # Small toy example (auto-tested)
#' set.seed(42)
#' n <- 50; p <- 2
#' X <- matrix(rnorm(n * p), n, p)
#' G <- matrix(rgamma(n * 3, shape = rep(c(2, 3, 4), each = n)), n, 3)
#' Y <- G / rowSums(G)
#' forest  <- DirichletRF(X, Y, num.trees = 5, num.cores = 1)
#' Xtest   <- matrix(rnorm(5 * p), 5, p)
#' pred    <- predict(forest, Xtest)
#' pred$mean_predictions
#'
#' \donttest{
#' n <- 500; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' alpha <- c(2, 3, 4)
#' G <- matrix(rgamma(n * length(alpha), shape = rep(alpha, each = n)),
#'             n, length(alpha))
#' Y <- G / rowSums(G)
#' forest <- DirichletRF(X, Y, num.trees = 50, num.cores = 1)
#' Xtest  <- matrix(rnorm(10 * p), 10, p)
#' pred   <- predict(forest, Xtest)
#' param_pred  <- pred$alpha_predictions / rowSums(pred$alpha_predictions)
#' single_pred <- predict(forest, Xtest[1, , drop = FALSE])
#' }
#' @rdname predict.DirichletRF
#' @aliases predict
#' @export
predict.DirichletRF <- function(object, newdata, ...) {

  X_new <- newdata

  # Input validation and coercion
  if (!is.matrix(X_new)) {
    if (is.data.frame(X_new)) {
      X_new <- as.matrix(X_new)
    } else if (is.vector(X_new) || is.numeric(X_new)) {
      X_new <- matrix(X_new, nrow = 1)
      warning("Input was a vector. Converting to 1-row matrix. ",
              "Consider using newdata[i, , drop = FALSE] when ",
              "subsetting matrices.")
    } else {
      stop("newdata must be a matrix, data frame, or numeric vector")
    }
  }

  if (!is.numeric(X_new)) stop("newdata must contain numeric values")

  return(PredictDirichletForest(object$forest, X_new,
                                method               = object$est.method,
                                use_leaf_predictions = TRUE))
}

#' Draw Conditional Samples from a Dirichlet Forest
#'
#' Given a fitted \code{DirichletRF} built with
#' \code{distributional = TRUE} and a single test covariate vector, draws
#' \code{size} compositional observations from the forest-weighted empirical
#' distribution over the training responses. Each training observation
#' receives a weight proportional to how often it co-occurs with the test
#' point in the same leaf across all trees; the returned rows are a
#' weighted-bootstrap draw from those training Y rows.
#'
#' @param object A \code{DirichletRF} object built with
#'   \code{distributional = TRUE}.
#' @param x_new A numeric vector of length p (a single test covariate point).
#' @param size A positive integer giving the number of compositional
#'   observations to draw. Default is \code{100L}.
#'
#' @return A numeric matrix of dimensions \code{size x k}, where each row
#'   is one draw from the conditional distribution of Y given \code{x_new}.
#'   Row names are \code{draw_1}, \code{draw_2}, \ldots and column names
#'   are inherited from the training Y matrix if available.
#'
#' @seealso
#' \code{\link{predict.DirichletRF}} for point predictions on new data.
#' \code{\link{print.DirichletRF}} for a summary of the fitted object.
#' \code{\link{importance.DirichletRF}} for impurity-based feature importance.
#' \code{\link{permutation_importance}} for permutation-based OOB feature importance.
#' \code{\link{predict_weights}} for proximity weights for new observations.
#'
#' @examples
#' set.seed(1)
#' n <- 80; p <- 3
#' X <- matrix(rnorm(n * p), n, p)
#' G <- matrix(rgamma(n * 4, shape = rep(c(1, 2, 3, 4), each = n)), n, 4)
#' Y <- G / rowSums(G)
#' forest <- DirichletRF(X, Y, num.trees = 20, num.cores = 1,
#'                       distributional = TRUE)
#' x_test <- rnorm(p)
#' draws  <- sample_conditional(forest, x_test, size = 200L)
#' colMeans(draws)   # estimated conditional mean of Y | x_test
#'
#' @export
sample_conditional <- function(object, x_new, size = 100L) {

  if (!isTRUE(object$distributional))
    stop("sample_conditional() requires the forest to be built with distributional = TRUE")

  if (!is.numeric(x_new) || !is.vector(x_new))
    stop("x_new must be a numeric vector (one test point)")

  if (!is.numeric(size) || length(size) != 1L || size < 1L)
    stop("size must be a positive integer")

  result <- DrawConditionalSample(object$forest,
                                  as.numeric(x_new),
                                  as.integer(size))

  colnames(result) <- colnames(object$Y_train)
  rownames(result) <- paste0("draw_", seq_len(nrow(result)))
  result
}

#' Feature Importance for a Dirichlet Forest
#'
#' Returns a data frame summarising feature importance from a fitted
#' \code{DirichletRF} object. Two measures are provided:
#' \describe{
#'   \item{\code{gain}}{Total likelihood gain accumulated across all splits
#'     where this feature was selected (raw, summed over all trees).}
#'   \item{\code{gain_normalised}}{Same as \code{gain} but normalised to
#'     sum to 1 across all features, making values comparable across
#'     forests of different sizes.}
#'   \item{\code{count}}{Number of times the feature was chosen as the
#'     best split variable across all trees and all internal nodes.}
#' }
#' The data frame is sorted by \code{gain_normalised} in descending order.
#'
#' @param object A \code{DirichletRF} object returned by
#'   \code{\link{DirichletRF}}.
#' @param ... Currently unused.
#'
#' @return A data frame with columns \code{feature}, \code{gain},
#'   \code{gain_normalised}, and \code{count}, sorted by
#'   \code{gain_normalised} descending.
#' 
#' @seealso
#' \code{\link{predict.DirichletRF}} for point predictions on new data.
#' \code{\link{print.DirichletRF}} for a summary of the fitted object
#'   (call as \code{print(forest)} or just \code{forest}).
#' \code{\link{sample_conditional}} for drawing compositional samples
#'   from the conditional predictive distribution (requires
#'   \code{distributional = TRUE}).
#' \code{\link{permutation_importance}} for permutation-based OOB feature
#'   importance (requires \code{compute.oob = TRUE}).
#' \code{\link{predict_weights}} for proximity weights for new observations
#'   (requires \code{distributional = TRUE}).
#' @examples
#' set.seed(42)
#' n <- 50; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' colnames(X) <- paste0("X", 1:p)
#' G <- matrix(rgamma(n * 3, shape = rep(c(2, 3, 4), each = n)), n, 3)
#' Y <- G / rowSums(G)
#' forest <- DirichletRF(X, Y, num.trees = 10, num.cores = 1)
#' importance(forest)
#'
#' @rdname importance.DirichletRF
#' @export
importance <- function(object, ...) UseMethod("importance")

#' @rdname importance.DirichletRF
#' @export
importance.DirichletRF <- function(object, ...) {
  imp <- object$importance
  if (is.null(imp))
    stop("No importance information found. Please refit with the current version.")

  df <- data.frame(
    feature         = names(imp$gain),
    gain            = unname(imp$gain),
    gain_normalised = unname(imp$gain_normalised),
    count           = unname(imp$count),
    stringsAsFactors = FALSE
  )

  df <- df[order(df$gain_normalised, decreasing = TRUE), ]
  rownames(df) <- NULL
  df
}

#' Permutation Feature Importance for a Dirichlet Forest
#'
#' Computes permutation-based variable importance (VI) for each feature.
#' For each tree \eqn{b} and feature \eqn{j}, the OOB error is measured
#' before and after randomly permuting column \eqn{j} within the OOB
#' sample of that tree. The importance of feature \eqn{j} is:
#'
#' \deqn{
#'   \mathrm{VI}(j) = \frac{1}{B} \sum_{b=1}^{B}
#'   \left[
#'     \frac{1}{R} \sum_{r=1}^{R}
#'     L\!\left(S_b,\, \tilde{X}^{(j,r)}\right)
#'     - L\!\left(S_b,\, X\right)
#'   \right]
#' }
#'
#' where \eqn{S_b} is the OOB index set of tree \eqn{b}, \eqn{R} is
#' \code{num.permutations}, and \eqn{\tilde{X}^{(j,r)}} is the data
#' matrix with column \eqn{j} randomly permuted on replicate \eqn{r},
#' holding all other columns fixed.
#'
#' The scaled version divides by the \strong{population} standard
#' deviation of the per-tree importances (denominator \eqn{B},
#' not \eqn{B-1}):
#'
#' \deqn{
#'   \mathrm{VI}_{\mathrm{scaled}}(j) =
#'   \frac{\mathrm{VI}(j)}{\widehat{\sigma}_j}, \quad
#'   \widehat{\sigma}_j^2 =
#'   \frac{1}{B}\sum_{b=1}^{B}
#'   \mathrm{VI}_{b,j}^2 - \mathrm{VI}(j)^2
#' }
#'
#' where \eqn{\mathrm{VI}_{b,j}} denotes the bracketed quantity above
#' for a single tree.
#'
#' @param object A \code{DirichletRF} object fitted with
#'   \code{compute.oob = TRUE}.
#' @param X The training covariate matrix (n x p) passed to
#'   \code{\link{DirichletRF}}. Required because leaf predictions are
#'   recomputed on permuted copies.
#' @param loss Loss function used to measure OOB error. One of:
#'   \describe{
#'     \item{\code{"aitchison"}}{(default) Mean Aitchison distance between
#'       predicted and true compositions. Respects simplex geometry.
#'       \eqn{d_A(y,\hat{y}) = \|\mathrm{clr}(y) - \mathrm{clr}(\hat{y})\|_2}.}
#'     \item{\code{"mse"}}{Mean squared error, averaged over components.}
#'     \item{\code{"kl"}}{Mean KL divergence \eqn{\sum_k y_k \log(y_k/\hat{y}_k)}.}
#'   }
#' @param num.permutations Number of random permutations to average over
#'   per feature per tree. Higher values reduce Monte Carlo noise.
#'   Default is \code{5}.
#' @param seed Integer random seed for reproducibility of permutations.
#'   Default is \code{42L}.
#'
#' @return A data frame with one row per feature and columns:
#'   \describe{
#'     \item{\code{feature}}{Feature name.}
#'     \item{\code{importance}}{Mean increase in OOB loss when the feature
#'       is permuted (\eqn{\mathrm{VI}(j)}). Larger = more important.}
#'     \item{\code{importance_scaled}}{Importance divided by its standard
#'       deviation across trees (\eqn{\mathrm{VI}_{\mathrm{scaled}}(j)}).
#'       Analogous to a t-statistic; values \eqn{> 1} suggest a feature
#'       contributes meaningfully.}
#'     \item{\code{importance_sd}}{Standard deviation of the per-tree
#'       importance values, giving a sense of variability.}
#'   }
#'   Sorted by \code{importance} descending.
#'
#' @details
#' \strong{Loss functions for compositional data}
#'
#' MSE ignores the simplex constraint and treats the components
#' independently. The Aitchison distance operates in the log-ratio space
#' that is natural for compositions and is the recommended default. KL
#' divergence is asymmetric but common in information-theoretic contexts.
#'
#' Small predicted values near zero can cause numerical issues for
#' Aitchison and KL losses. A small constant (\code{1e-10}) is added to
#' all predictions before computing these losses.
#'
#' \strong{Interpretation}
#'
#' A feature with \code{importance} near zero (or negative, due to Monte
#' Carlo noise) does not contribute to predictive accuracy. Features with
#' large positive \code{importance_scaled} are robustly important across
#' trees.
#'
#' @seealso
#' \code{\link{predict.DirichletRF}} for point predictions on new data.
#' \code{\link{print.DirichletRF}} for a summary of the fitted object
#'   (call as \code{print(forest)} or just \code{forest}).
#' \code{\link{sample_conditional}} for drawing compositional samples
#'   from the conditional predictive distribution (requires
#'   \code{distributional = TRUE}).
#' \code{\link{importance.DirichletRF}} for impurity-based (gain and
#'   count) feature importance.
#' \code{\link{predict_weights}} for proximity weights for new observations
#'   (requires \code{distributional = TRUE}).
#' 
#' @examples
#' set.seed(42)
#' n <- 100; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' colnames(X) <- paste0("X", 1:p)
#' alpha_mat <- cbind(2 + 3 * (X[, 1] > 0), 3 + 3 * (X[, 2] > 0), rep(4, n))
#' G <- matrix(rgamma(n * 3, shape = as.vector(t(alpha_mat))), n, 3, byrow = TRUE)
#' Y <- G / rowSums(G)
#'
#' forest <- DirichletRF(X, Y, num.trees = 50, num.cores = 1,
#'                       replace = TRUE, compute.oob = TRUE)
#' permutation_importance(forest, X)
#'
#' @export
permutation_importance <- function(object, X,
                                   loss             = c("aitchison", "mse", "kl"),
                                   num.permutations = 5L,
                                   seed             = 42L) {

  if (!inherits(object, "DirichletRF"))
    stop("object must be a DirichletRF")
  if (!isTRUE(object$compute.oob))
    stop("Permutation importance requires compute.oob = TRUE.")

  loss <- match.arg(loss)
  if (!is.matrix(X)) X <- as.matrix(X)

  feat_names <- if (!is.null(colnames(X))) colnames(X)
                else paste0("X", seq_len(ncol(X)))

  df <- PermutationImportance(object$forest, X, object$Y_train,
                              loss, num.permutations, seed)

  df$feature <- feat_names
  df <- df[, c("feature", "importance", "importance_scaled", "importance_sd")]
  df <- df[order(df$importance, decreasing = TRUE), ]
  rownames(df) <- NULL
  df
}


#' Proximity Weights for New Observations
#'
#' For each row of \code{newdata}, computes a normalised weight vector
#' over all training observations, where the weight of training
#' observation \code{j} is proportional to how often it co-occurs with
#' the new point in the same leaf across all trees. These weights define
#' the forest-weighted empirical distribution over the training responses
#' and can be used to estimate conditional quantities such as means,
#' variances, or probabilities for the new covariate point.
#'
#' @param object A \code{DirichletRF} object built with
#'   \code{distributional = TRUE}.
#' @param newdata A numeric matrix of new covariates (n_test x p).
#'   Column order must match the training matrix passed to
#'   \code{\link{DirichletRF}}.
#'
#' @return A numeric matrix of dimensions n_test x n_train. Row \code{i}
#'   contains the normalised proximity weights of the \code{i}-th new
#'   observation over all \code{n_train} training observations. Each row
#'   sums to 1. Entries are zero for training observations that never
#'   shared a leaf with the new point across any tree.
#'
#' @details
#' Weights are computed using all trees in the forest (no OOB restriction
#' applies, since new observations were not part of training). This
#' contrasts with \code{$oob$weights}, which restricts each training
#' observation to its held-out trees only and is available directly on
#' the fitted object when both \code{distributional = TRUE} and
#' \code{compute.oob = TRUE}.
#'
#' @seealso
#' \code{\link{predict.DirichletRF}} for point predictions on new data.
#' \code{\link{print.DirichletRF}} for a summary of the fitted object
#'   (call as \code{print(forest)} or just \code{forest}).
#' \code{\link{sample_conditional}} for drawing compositional samples
#'   from the conditional predictive distribution (requires
#'   \code{distributional = TRUE}).
#' \code{\link{importance.DirichletRF}} for impurity-based (gain and
#'   count) feature importance.
#' \code{\link{permutation_importance}} for permutation-based OOB feature
#'   importance (requires \code{compute.oob = TRUE}).
#'
#' @examples
#' set.seed(42)
#' n <- 100; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' colnames(X) <- paste0("X", 1:p)
#' alpha_mat <- cbind(2 + 3 * (X[, 1] > 0), 3 + 3 * (X[, 2] > 0), rep(4, n))
#' G <- matrix(rgamma(n * 3, shape = as.vector(t(alpha_mat))), n, 3, byrow = TRUE)
#' Y <- G / rowSums(G)
#'
#' forest <- DirichletRF(X, Y, num.trees = 50, num.cores = 1,
#'                       distributional = TRUE)
#'
#' # Weights for 5 new observations -- matrix is 5 x 100
#' Xtest <- matrix(rnorm(5 * p), 5, p)
#' colnames(Xtest) <- paste0("X", 1:p)
#' W <- predict_weights(forest, Xtest)
#' dim(W)       # 5 x 100
#' rowSums(W)   # all 1
#'
#' # Weighted conditional mean for each new observation
#' Y_hat <- W %*% Y   # 5 x k
#'
#' @export
predict_weights <- function(object, newdata) {
  if (!inherits(object, "DirichletRF"))
    stop("object must be a DirichletRF")
  if (!isTRUE(object$distributional))
    stop("predict_weights requires distributional = TRUE")
  if (!is.matrix(newdata)) newdata <- as.matrix(newdata)
  PredictWeights(object$forest, newdata)
}
