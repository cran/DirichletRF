#' Build a Dirichlet Random Forest for Compositional Responses
#'
#' Build a Dirichlet random forest for compositional responses. In
#' compositional data analysis (CoDA), parts reside in the simplex, and this
#' random forest ensures model output abide by CoDA principles. The
#' implementation uses OpenMP for parallel tree building. Note that this implementation
#' does not support out-of-bag (OOB) error estimation.
#'
#' The forest provides two types of fitted values: mean-based predictions
#' (derived from sample means at each leaf) and parameter-based predictions
#' (derived from normalised Dirichlet alpha parameters).
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
#' @param num.cores Number of OpenMP threads used for parallel tree building.
#'   The default is \code{-1} which uses all the cores on the system minus 1.
#'   Users may also specify \code{1} which means that the forest will be
#'   built sequentially.
#'
#' @return A list of the forest which contains the following:
#' \describe{
#'   \item{\code{type}}{Parallelisation type used: \code{"openmp"} or
#'     \code{"sequential"}.}
#'   \item{\code{num.cores}}{Number of cores used.}
#'   \item{\code{num.trees}}{Total number of trees in the forest.}
#'   \item{\code{Y_train}}{Training data used (no OOB).}
#'   \item{\code{fitted}}{A list of fitted values:
#'     \describe{
#'       \item{\code{alpha_hat}}{Estimated Dirichlet alpha parameters
#'         (n x k matrix).}
#'       \item{\code{mean_based}}{Mean-based fitted values (n x k matrix).}
#'       \item{\code{param_based}}{Parameter-based fitted values obtained
#'         by normalising \code{alpha_hat} (n x k matrix).}
#'     }
#'   }
#'   \item{\code{residuals}}{A list of residuals (Y - fitted):
#'     \describe{
#'       \item{\code{mean_based}}{Residuals from mean-based predictions.}
#'       \item{\code{param_based}}{Residuals from parameter-based
#'         predictions.}
#'     }
#'   }
#' }
#'
#' @examples
#' # Small toy example (auto-tested)
#' set.seed(42)
#' n <- 50; p <- 2
#' X <- matrix(rnorm(n * p), n, p)
#' G <- matrix(rgamma(n * 3, shape = rep(c(2, 3, 4), each = n)), n, 3)
#' Y <- G / rowSums(G)
#' forest <- DirichletRF(X, Y, num.trees = 5, num.cores = 1)
#' print(forest)
#' Xtest <- matrix(rnorm(5 * p), 5, p)
#' pred  <- predict(forest, Xtest)
#'
#' \donttest{
#' # Larger example
#' n <- 500; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' alpha <- c(2, 3, 4)
#' G <- matrix(rgamma(n * length(alpha), shape = rep(alpha, each = n)),
#'             n, length(alpha))
#' Y <- G / rowSums(G)
#'
#' forest1 <- DirichletRF(X, Y, num.trees = 100, num.cores = 1)
#' forest2 <- DirichletRF(X, Y, num.trees = 100)
#'
#' alpha_hat   <- forest1$fitted$alpha_hat
#' mean_fit    <- forest1$fitted$mean_based
#' param_fit   <- forest1$fitted$param_based
#' resid_mean  <- forest1$residuals$mean_based
#' resid_param <- forest1$residuals$param_based
#'
#' Xtest <- matrix(rnorm(10 * p), 10, p)
#' pred  <- predict(forest1, Xtest)
#' param_pred <- pred$alpha_predictions / rowSums(pred$alpha_predictions)
#' }
#'
#' @references
#' Masoumifard, K., van der Westhuizen, S., & Gardner-Lubbe, S. (In press).
#' Dirichlet-random forest for predicting compositional data. In A. Bekker,
#' J. Ferreira, & P. Nagar (Eds.), \emph{Environmental Statistics: Innovative
#' Methods and Applications}. CRC Press.
#'
#' @seealso
#' \code{\link{predict.dirichlet_forest}} for making predictions on new data.
#' @importFrom stats predict
#' @export
DirichletRF <- function(X, Y, num.trees = 100, max.depth = 10,
                        min.node.size = 5, mtry = -1, seed = 123,
                        est.method = "mom", num.cores = -1) {

  # Hardcoded for this version; will be exposed in a future release.
  store_samples <- FALSE

  # Input validation
  if (!is.matrix(X) || !is.matrix(Y)) stop("X and Y must be matrices")
  if (nrow(X) != nrow(Y)) stop("X and Y must have the same number of rows")

  # Resolve num.cores
  if (num.cores == -1) {
    num.cores <- max(1L, parallel::detectCores() - 1L)
  }
  num.cores <- max(1L, as.integer(num.cores))

  par_type <- if (num.cores == 1L) "sequential" else "openmp"
  message("Building ", par_type, " forest with ", num.cores,
          " thread(s) for ", num.trees, " trees")

  # All parallelism handled inside C++ via OpenMP
  forest <- DirichletForest(X, Y,
                            B             = num.trees,
                            d_max         = max.depth,
                            n_min         = min.node.size,
                            m_try         = mtry,
                            seed          = seed,
                            method        = est.method,
                            store_samples = store_samples,
                            num_cores     = num.cores)

  result <- list(
    type      = par_type,
    forest    = forest,
    num.cores = num.cores,
    num.trees = num.trees,
    Y_train   = Y
  )
  class(result) <- c("dirichlet_forest", "list")

  message("Computing fitted values and residuals...")
  fitted_preds <- predict(result, X)
  alpha_means  <- fitted_preds$alpha_predictions /
                  rowSums(fitted_preds$alpha_predictions)

  result$fitted <- list(
    alpha_hat   = fitted_preds$alpha_predictions,
    mean_based  = fitted_preds$mean_predictions,
    param_based = alpha_means
  )
  result$residuals <- list(
    mean_based  = Y - fitted_preds$mean_predictions,
    param_based = Y - alpha_means
  )

  return(result)
}


#' Custom Print Method for dirichlet_forest Objects
#'
#' Suppresses the display of large data matrices (Y_train, fitted,
#' residuals) when the object is printed, while keeping them accessible
#' via \code{$}.
#'
#' @param x A \code{dirichlet_forest} object.
#' @param ... Further arguments passed to or from other methods.
#'
#' @return Invisibly returns \code{x}, the \code{dirichlet_forest} object
#'   unchanged. Called primarily for its side effect of printing a summary
#'   of the model to the console.
#' @export
print.dirichlet_forest <- function(x, ...) {

  N <- nrow(x$Y_train)
  K <- ncol(x$Y_train)
  train_info <- if (!is.null(N) && !is.null(K))
    paste0(N, " observations (n) x ", K, " components (k)")
  else
    "unknown"

  cat(
    "============================================\n",
    "Dirichlet Forest Model\n",
    "============================================\n",
    " Type:          ", x$type, "\n",
    " Total Trees:   ", x$num.trees, "\n",
    " Cores Used:    ", x$num.cores, "\n",
    " Training Data: ", train_info, "\n",
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
    "============================================\n",
    sep = ""
  )

  invisible(x)
}




#' Predict with a Dirichlet Forest
#'
#' Makes predictions using a fitted \code{dirichlet_forest} object returned
#' by \code{\link{DirichletRF}}.
#'
#' @param object A \code{dirichlet_forest} object.
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
#' @export
predict.dirichlet_forest <- function(object, newdata, ...) {

  # Hardcoded for this version; will be exposed in a future release.
  est.method           <- "mom"
  use_leaf_predictions <- TRUE

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
                                method               = est.method,
                                use_leaf_predictions = use_leaf_predictions))
}