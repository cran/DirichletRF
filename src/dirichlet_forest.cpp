#include <Rcpp.h>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// ============================================================================
// NODE STRUCTURE — uses STL only, no Rcpp types
// ============================================================================
struct Node {
    int    feature_index;
    double split_value;
    bool   is_leaf;
    Node*  left;
    Node*  right;

    // Pre-computed leaf predictions
    std::vector<double> mean_prediction;
    std::vector<double> alpha_prediction;

    // Distributional mode (store_samples = TRUE)
    std::vector<int> leaf_samples;

    Node() : feature_index(-1), split_value(0.0), is_leaf(false),
             left(nullptr), right(nullptr) {}

    ~Node() {
        delete left;
        delete right;
    }
};

// ============================================================================
// DIRICHLET LOG-LIKELIHOOD — pure STL
// ============================================================================
double log_likelihood_dirichlet(const std::vector<double>& Y_vec,
                                const std::vector<int>&    indices,
                                int                        n_classes,
                                const std::vector<double>& alpha) {
    int    n         = (int)indices.size();
    double loglik    = 0.0;
    double alpha_sum = 0.0;

    for (int j = 0; j < n_classes; j++) alpha_sum += alpha[j];

    double lgamma_alpha_sum = std::lgamma(alpha_sum);

    std::vector<double> lgamma_alpha(n_classes);
    for (int j = 0; j < n_classes; j++)
        lgamma_alpha[j] = std::lgamma(alpha[j]);

    for (int i = 0; i < n; i++) {
        int    row    = indices[i];
        double sum_y  = 0.0;
        double contrib = 0.0;

        for (int j = 0; j < n_classes; j++) {
            double y = Y_vec[row * n_classes + j];
            if (y <= 0.0 || y >= 1.0) return -1e18;
            sum_y   += y;
            contrib += (alpha[j] - 1.0) * std::log(y);
        }

        if (std::abs(sum_y - 1.0) > 1e-6) return -1e18;

        loglik += lgamma_alpha_sum;
        for (int j = 0; j < n_classes; j++) loglik -= lgamma_alpha[j];
        loglik += contrib;
    }

    return loglik;
}



// ============================================================================
// METHOD OF MOMENTS — pure STL
// ============================================================================
std::vector<double> estimate_mom(const std::vector<double>& Y_vec,
                                 const std::vector<int>&    indices,
                                 int                        n_classes) {
    int n = (int)indices.size();

    if (n == 0) return std::vector<double>(n_classes, 1.0);

    if (n == 1) {
        std::vector<double> result(n_classes);
        int row = indices[0];
        for (int j = 0; j < n_classes; j++)
            result[j] = std::max(0.1, std::min(1000.0,
                                 Y_vec[row * n_classes + j]));
        return result;
    }

    // Means
    std::vector<double> means(n_classes, 0.0);
    for (int i = 0; i < n; i++) {
        int row = indices[i];
        for (int j = 0; j < n_classes; j++)
            means[j] += Y_vec[row * n_classes + j];
    }
    for (int j = 0; j < n_classes; j++) means[j] /= n;

    // Variances (Bessel's correction)
    std::vector<double> variances(n_classes, 0.0);
    for (int i = 0; i < n; i++) {
        int row = indices[i];
        for (int j = 0; j < n_classes; j++) {
            double diff = Y_vec[row * n_classes + j] - means[j];
            variances[j] += diff * diff;
        }
    }
    for (int j = 0; j < n_classes; j++) {
        variances[j] /= (n - 1);
        if (variances[j] < 1e-8) variances[j] = 1e-8;
    }

    // Concentration
    double v = (means[0] * (1.0 - means[0])) / variances[0] - 1.0;
    if (v <= 0.0) v = 0.1;

    // Alpha
    std::vector<double> alpha(n_classes);
    for (int j = 0; j < n_classes; j++)
        alpha[j] = std::max(0.1, std::min(1000.0, v * means[j]));

    return alpha;
}

// ============================================================================
// LU SOLVER — pure STL
// ============================================================================
bool lu_solve(std::vector<std::vector<double>>  A,
              const std::vector<double>&         b,
              std::vector<double>&               x,
              int                                n) {
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    for (int k = 0; k < n; k++) {
        int    max_row = k;
        double max_val = std::abs(A[k][k]);
        for (int i = k + 1; i < n; i++) {
            double val = std::abs(A[i][k]);
            if (val > max_val) { max_val = val; max_row = i; }
        }
        if (max_val < 1e-12) return false;

        if (max_row != k) {
            std::swap(A[k], A[max_row]);
            std::swap(perm[k], perm[max_row]);
        }

        for (int i = k + 1; i < n; i++) {
            A[i][k] /= A[k][k];
            for (int j = k + 1; j < n; j++)
                A[i][j] -= A[i][k] * A[k][j];
        }
    }

    std::vector<double> b_perm(n);
    for (int i = 0; i < n; i++) b_perm[i] = b[perm[i]];

    std::vector<double> y(n);
    for (int i = 0; i < n; i++) {
        y[i] = -b_perm[i];
        for (int j = 0; j < i; j++) y[i] -= A[i][j] * y[j];
    }

    x.resize(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) x[i] -= A[i][j] * x[j];
        x[i] /= A[i][i];
    }

    return true;
}

// ============================================================================
// MLE (Newton-Raphson) — pure STL
// ============================================================================
std::vector<double> estimate_mle(const std::vector<double>& Y_vec,
                                 const std::vector<int>&    indices,
                                 int                        n_classes,
                                 int    max_iter = 1000,
                                 double tol      = 1e-6,
                                 double lambda   = 1e-6) {
    int n = (int)indices.size();
    if (n == 0) return std::vector<double>(n_classes, 1.0);

    std::vector<double> alpha = estimate_mom(Y_vec, indices, n_classes);

    // Pre-compute log Y
    std::vector<std::vector<double>> log_Y(n, std::vector<double>(n_classes));
    std::vector<std::vector<bool>>   valid(n,  std::vector<bool>(n_classes, false));
    for (int i = 0; i < n; i++) {
        int row = indices[i];
        for (int j = 0; j < n_classes; j++) {
            double y = Y_vec[row * n_classes + j];
            if (y > 0.0) {
                log_Y[i][j] = std::log(y);
                valid[i][j] = true;
            }
        }
    }

    for (int iter = 0; iter < max_iter; iter++) {
        double alpha_sum = 0.0;
        for (int j = 0; j < n_classes; j++) alpha_sum += alpha[j];

        double dig_sum  = R::digamma(alpha_sum);
        double trig_sum = R::trigamma(alpha_sum);

        // Gradient
        std::vector<double> grad(n_classes, 0.0);
        for (int j = 0; j < n_classes; j++) {
            grad[j] = n * (dig_sum - R::digamma(alpha[j]));
            for (int i = 0; i < n; i++)
                if (valid[i][j]) grad[j] += log_Y[i][j];
        }

        // Hessian
        std::vector<std::vector<double>> H(n_classes,
                                           std::vector<double>(n_classes));
        for (int j = 0; j < n_classes; j++)
            for (int l = 0; l < n_classes; l++)
                H[j][l] = (j == l)
                    ? n * (trig_sum - R::trigamma(alpha[j])) + lambda
                    : n * trig_sum;

        // Solve
        std::vector<double> delta(n_classes);
        if (!lu_solve(H, grad, delta, n_classes)) {
            for (int j = 0; j < n_classes; j++) {
                double d = n * (trig_sum - R::trigamma(alpha[j])) + lambda;
                delta[j] = -grad[j] / d;
            }
        }

        // Convergence
        double norm_sq = 0.0;
        for (int j = 0; j < n_classes; j++) norm_sq += delta[j] * delta[j];
        if (norm_sq < tol * tol) break;

        // Line search
        double step    = 1.0;
        bool   updated = false;
        for (int ls = 0; ls < 10; ls++) {
            bool ok = true;
            for (int j = 0; j < n_classes; j++) {
                double a = alpha[j] + step * delta[j];
                if (a < 0.1 || a > 1000.0) { ok = false; break; }
            }
            if (ok) {
                for (int j = 0; j < n_classes; j++)
                    alpha[j] += step * delta[j];
                updated = true;
                break;
            }
            step *= 0.5;
        }
        if (!updated) break;
    }

    return alpha;
}

// ============================================================================
// R EXPORT WRAPPERS (thin Rcpp wrappers around STL functions)
// ============================================================================
// [[Rcpp::export]]
NumericVector estimate_dirichlet_mom(const NumericMatrix& Y) {
    int n = Y.nrow(), k = Y.ncol();
    std::vector<double> Y_vec(n * k);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            Y_vec[i * k + j] = Y(i, j);
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    auto result = estimate_mom(Y_vec, indices, k);
    return NumericVector(result.begin(), result.end());
}

// [[Rcpp::export]]
NumericVector estimate_dirichlet_mle(const NumericMatrix& Y,
                                     int    max_iter = 10000,
                                     double tol      = 1e-6,
                                     double lambda   = 1e-6) {
    int n = Y.nrow(), k = Y.ncol();
    std::vector<double> Y_vec(n * k);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            Y_vec[i * k + j] = Y(i, j);
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    auto result = estimate_mle(Y_vec, indices, k, max_iter, tol, lambda);
    return NumericVector(result.begin(), result.end());
}

// ============================================================================
// FIT TERMINAL NODE — pure STL
// ============================================================================
void FitTerminalNode(Node*                      node,
                     const std::vector<double>& Y_vec,
                     const std::vector<int>&    indices,
                     int                        n_classes,
                     const std::string&         method,
                     bool                       store_samples) {
    node->is_leaf = true;

    if (indices.empty()) {
        node->alpha_prediction.assign(n_classes, 1.0);
        node->mean_prediction.assign(n_classes,  1.0 / n_classes);
    } else {
        // Mean
        node->mean_prediction.assign(n_classes, 0.0);
        for (int i : indices)
            for (int j = 0; j < n_classes; j++)
                node->mean_prediction[j] += Y_vec[i * n_classes + j];
        for (int j = 0; j < n_classes; j++)
            node->mean_prediction[j] /= indices.size();

        // Alpha
        node->alpha_prediction = (method == "mle")
            ? estimate_mle(Y_vec, indices, n_classes)
            : estimate_mom(Y_vec, indices, n_classes);
    }

    if (store_samples) node->leaf_samples = indices;
}

// ============================================================================
// FIND BEST SPLIT — pure STL
// ============================================================================
struct SplitResult {
    double           gain;
    int              feature;
    double           split_value;
    std::vector<int> left_indices;
    std::vector<int> right_indices;
};

SplitResult FindBestSplit(const std::vector<double>& X_vec,
                          const std::vector<double>& Y_vec,
                          const std::vector<int>&    indices,
                          const std::vector<int>&    feature_subset,
                          int                        n_features,
                          int                        n_classes,
                          const std::string&         method,
                          std::vector<double>&       importance_gain,
                          std::vector<int>&          importance_count) {
    SplitResult best;
    best.gain    = -std::numeric_limits<double>::infinity();
    best.feature = -1;
    best.split_value = 0.0;

    int n = (int)indices.size();

    // Parent log-likelihood
    std::vector<double> parent_alpha = (method == "mle")
        ? estimate_mle(Y_vec, indices, n_classes)
        : estimate_mom(Y_vec, indices, n_classes);
    double parent_ll = log_likelihood_dirichlet(Y_vec, indices,
                                                n_classes, parent_alpha);

    for (int feat : feature_subset) {
        // Unique sorted values for this feature
        std::vector<double> vals;
        vals.reserve(n);
        for (int i : indices) vals.push_back(X_vec[i * n_features + feat]);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

        if ((int)vals.size() <= 1) continue;

        for (int k = 1; k < (int)vals.size(); k++) {
            double split_val = (vals[k-1] + vals[k]) / 2.0;

            std::vector<int> left, right;
            left.reserve(n);
            right.reserve(n);
            for (int i : indices) {
                if (X_vec[i * n_features + feat] <= split_val)
                    left.push_back(i);
                else
                    right.push_back(i);
            }

            if ((int)left.size() < 2 || (int)right.size() < 2) continue;

            std::vector<double> la = (method == "mle")
                ? estimate_mle(Y_vec, left,  n_classes)
                : estimate_mom(Y_vec, left,  n_classes);
            std::vector<double> ra = (method == "mle")
                ? estimate_mle(Y_vec, right, n_classes)
                : estimate_mom(Y_vec, right, n_classes);

            double gain =
                log_likelihood_dirichlet(Y_vec, left,  n_classes, la) +
                log_likelihood_dirichlet(Y_vec, right, n_classes, ra) -
                parent_ll;

            if (gain > best.gain) {
                best.gain         = gain;
                best.feature      = feat;
                best.split_value  = split_val;
                best.left_indices  = left;
                best.right_indices = right;
            }
        }
    }

    // Accumulate importance for the winning feature
    if (best.feature != -1 && best.gain > 0.0) {
        importance_gain[best.feature]  += best.gain;
        importance_count[best.feature] += 1;
    }

    return best;
}

// ============================================================================
// GROW TREE — pure STL
// ============================================================================
Node* GrowTree(const std::vector<double>& X_vec,
               const std::vector<double>& Y_vec,
               int                        n_features,
               int                        n_classes,
               const std::vector<int>&    indices,
               int                        current_depth,
               int                        d_max,
               int                        n_min,
               int                        m_try,
               std::mt19937&              gen,
               const std::string&         method,
               bool                       store_samples,
               std::vector<double>&       importance_gain,
               std::vector<int>&          importance_count) {
    Node* node = new Node();

    if ((int)indices.size() < n_min || current_depth > d_max ||
        indices.empty()) {
        FitTerminalNode(node, Y_vec, indices, n_classes, method, store_samples);
        return node;
    }

    // Feature subset — pure STL shuffle
    std::vector<int> all_features(n_features);
    std::iota(all_features.begin(), all_features.end(), 0);
    std::shuffle(all_features.begin(), all_features.end(), gen);
    std::vector<int> feature_subset(all_features.begin(),
                                    all_features.begin() +
                                    std::min(m_try, n_features));

    SplitResult split = FindBestSplit(X_vec, Y_vec, indices, feature_subset,
                                      n_features, n_classes, method,
                                      importance_gain, importance_count);

    if (split.gain <= 0.0 || split.feature == -1) {
        FitTerminalNode(node, Y_vec, indices, n_classes, method, store_samples);
        return node;
    }

    node->feature_index = split.feature;
    node->split_value   = split.split_value;
    node->is_leaf       = false;

    node->left  = GrowTree(X_vec, Y_vec, n_features, n_classes,
                            split.left_indices,  current_depth + 1,
                            d_max, n_min, m_try, gen, method, store_samples,
                            importance_gain, importance_count);
    node->right = GrowTree(X_vec, Y_vec, n_features, n_classes,
                            split.right_indices, current_depth + 1,
                            d_max, n_min, m_try, gen, method, store_samples,
                            importance_gain, importance_count);

    return node;
}

// ============================================================================
// TREE TRAVERSAL — pure STL
// ============================================================================
Node* FindLeafNode(Node* node, const std::vector<double>& x,
                   int n_features, int row) {
    if (node->is_leaf) return node;
    return (x[row * n_features + node->feature_index] <= node->split_value)
        ? FindLeafNode(node->left,  x, n_features, row)
        : FindLeafNode(node->right, x, n_features, row);
}

// Fast single-sample traversal for prediction
Node* FindLeafNode1D(Node* node, const std::vector<double>& x) {
    if (node->is_leaf) return node;
    return (x[node->feature_index] <= node->split_value)
        ? FindLeafNode1D(node->left,  x)
        : FindLeafNode1D(node->right, x);
}

// ============================================================================
// BUILD DIRICHLET FOREST — OpenMP parallel over pure STL
// ============================================================================
// [[Rcpp::export]]
List DirichletForest(NumericMatrix X, NumericMatrix Y, int B = 100,
                     int d_max = 10, int n_min = 5, int m_try = -1,
                     int seed = 123, std::string method = "mom",
                     bool store_samples = false,
                     int num_cores = 1,
                     bool replace = false,
                     double sample_fraction = 1.0,
                     bool compute_oob = false) {

    int n_samples  = X.nrow();
    int n_features = X.ncol();
    int n_classes  = Y.ncol();

    if (m_try <= 0) m_try = std::max(1, (int)std::sqrt((double)n_features));

    // Number of observations each tree is trained on
    int n_inbag = std::max(1, (int)std::floor(sample_fraction * n_samples));

    #ifdef _OPENMP
    omp_set_num_threads(num_cores);
    #endif

    // Copy Rcpp matrices to flat STL vectors BEFORE parallel region
    // — Rcpp objects are NOT thread-safe
    std::vector<double> X_vec(n_samples * n_features);
    std::vector<double> Y_vec(n_samples * n_classes);

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++)
            X_vec[i * n_features + j] = X(i, j);
        for (int j = 0; j < n_classes; j++)
            Y_vec[i * n_classes + j]  = Y(i, j);
    }

    // Per-tree RNGs seeded reproducibly from master seed
    std::vector<std::mt19937> generators(B);
    {
        std::mt19937 master_gen(seed);
        for (int b = 0; b < B; b++) generators[b].seed(master_gen());
    }

    std::vector<Node*> forest(B, nullptr);

    // Per-tree importance accumulators (one vector per tree for thread safety)
    std::vector<std::vector<double>> tree_imp_gain(B,
        std::vector<double>(n_features, 0.0));
    std::vector<std::vector<int>> tree_imp_count(B,
        std::vector<int>(n_features, 0));

    // Per-tree OOB index sets (only populated when compute_oob = true)
    std::vector<std::vector<int>> oob_indices(B);

    // Parallel tree building — all code inside uses pure STL only
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int b = 0; b < B; b++) {
        std::vector<int> inbag(n_inbag);

        if (replace) {
            // Bootstrap: sample n_inbag times with replacement
            std::uniform_int_distribution<int> sampler(0, n_samples - 1);
            for (int i = 0; i < n_inbag; i++)
                inbag[i] = sampler(generators[b]);
        } else {
            // Subsample: draw n_inbag unique observations without replacement
            std::vector<int> all(n_samples);
            std::iota(all.begin(), all.end(), 0);
            std::shuffle(all.begin(), all.end(), generators[b]);
            inbag.assign(all.begin(), all.begin() + n_inbag);
        }

        // OOB indices — only needed when compute_oob = true
        if (compute_oob) {
            std::vector<bool> inbag_flag(n_samples, false);
            for (int idx : inbag) inbag_flag[idx] = true;
            std::vector<int> oob;
            oob.reserve(n_samples - n_inbag);
            for (int i = 0; i < n_samples; i++)
                if (!inbag_flag[i]) oob.push_back(i);
            oob_indices[b] = oob;
        }

        forest[b] = GrowTree(X_vec, Y_vec, n_features, n_classes,
                             inbag, 0, d_max, n_min, m_try,
                             generators[b], method, store_samples,
                             tree_imp_gain[b], tree_imp_count[b]);
    }

    // Aggregate importance across trees
    std::vector<double> imp_gain(n_features, 0.0);
    std::vector<int>    imp_count(n_features, 0);
    for (int b = 0; b < B; b++) {
        for (int f = 0; f < n_features; f++) {
            imp_gain[f]  += tree_imp_gain[b][f];
            imp_count[f] += tree_imp_count[b][f];
        }
    }

    // OOB block — skipped entirely when compute_oob = false
    double        oob_mse     = NA_REAL;
    NumericMatrix oob_pred_mat(compute_oob ? n_samples : 1,
                               compute_oob ? n_classes  : 1);
    oob_pred_mat.fill(NA_REAL);

    // Second OOB matrix for alpha predictions
    NumericMatrix oob_alpha_mat(compute_oob ? n_samples : 1,
                                compute_oob ? n_classes  : 1);
    oob_alpha_mat.fill(NA_REAL);

    if (compute_oob) {
        std::vector<std::vector<double>> oob_mean_sum(n_samples,
            std::vector<double>(n_classes, 0.0));
        std::vector<std::vector<double>> oob_alpha_sum(n_samples,
            std::vector<double>(n_classes, 0.0));
        std::vector<int> oob_pred_count(n_samples, 0);

        for (int b = 0; b < B; b++) {
            for (int i : oob_indices[b]) {
                std::vector<double> sample(n_features);
                for (int f = 0; f < n_features; f++)
                    sample[f] = X_vec[i * n_features + f];
                Node* leaf = FindLeafNode1D(forest[b], sample);
                for (int j = 0; j < n_classes; j++) {
                    oob_mean_sum[i][j]  += leaf->mean_prediction[j];
                    oob_alpha_sum[i][j] += leaf->alpha_prediction[j];
                }
                oob_pred_count[i]++;
            }
        }

        double oob_mse_acc = 0.0;
        int    oob_valid   = 0;
        oob_pred_mat  = NumericMatrix(n_samples, n_classes);
        oob_alpha_mat = NumericMatrix(n_samples, n_classes);
        oob_pred_mat.fill(NA_REAL);
        oob_alpha_mat.fill(NA_REAL);

        for (int i = 0; i < n_samples; i++) {
            if (oob_pred_count[i] == 0) continue;
            double obs_mse = 0.0;
            for (int j = 0; j < n_classes; j++) {
                double pred  = oob_mean_sum[i][j] / oob_pred_count[i];
                double resid = Y_vec[i * n_classes + j] - pred;
                obs_mse += resid * resid;
                oob_pred_mat(i, j)  = pred;
                oob_alpha_mat(i, j) = oob_alpha_sum[i][j] / oob_pred_count[i];
            }
            oob_mse_acc += obs_mse / n_classes;
            oob_valid++;
        }
        if (oob_valid > 0) oob_mse = oob_mse_acc / oob_valid;
    }

    // Wrap back into Rcpp AFTER parallel region
    List forest_ptrs(B);
    for (int i = 0; i < B; i++)
        forest_ptrs[i] = XPtr<Node>(forest[i], true);

    // Convert per-tree OOB index sets to an R list of integer vectors.
    // Each element b is a 1-based integer vector of OOB row indices for tree b
    // (empty when compute_oob = false).
    List oob_indices_list(B);
    for (int b = 0; b < B; b++) {
        IntegerVector iv(oob_indices[b].size());
        for (int k = 0; k < (int)oob_indices[b].size(); k++)
            iv[k] = oob_indices[b][k] + 1;   // convert to 1-based for R
        oob_indices_list[b] = iv;
    }

    List result = List::create(
        Named("forest")                = forest_ptrs,
        Named("n_trees")               = B,
        Named("n_features")            = n_features,
        Named("n_classes")             = n_classes,
        Named("store_samples")         = store_samples,
        Named("importance_gain")       = imp_gain,
        Named("importance_count")      = imp_count,
        Named("oob_mse")               = oob_mse,
        Named("oob_predictions")       = oob_pred_mat,
        Named("oob_alpha_predictions") = oob_alpha_mat,
        Named("oob_indices")           = oob_indices_list
    );

    if (store_samples) {
        result["X_train"] = X;
        result["Y_train"] = Y;
    }

    return result;
}

// ============================================================================
// GET LEAF PREDICTIONS (fast path) — Rcpp only at entry/exit
// ============================================================================
// [[Rcpp::export]]
List GetLeafPredictions(List forest_model, NumericMatrix X_new) {
    List forest_ptrs = forest_model["forest"];
    int  n_trees     = forest_model["n_trees"];
    int  n_classes   = forest_model["n_classes"];
    int  n_samples   = X_new.nrow();
    int  n_features  = X_new.ncol();

    // Copy to STL
    std::vector<double> X_vec(n_samples * n_features);
    for (int i = 0; i < n_samples; i++)
        for (int j = 0; j < n_features; j++)
            X_vec[i * n_features + j] = X_new(i, j);

    NumericMatrix alpha_out(n_samples, n_classes);
    NumericMatrix mean_out(n_samples,  n_classes);

    for (int i = 0; i < n_samples; i++) {
        // Build 1D sample vector for traversal
        std::vector<double> sample(n_features);
        for (int j = 0; j < n_features; j++)
            sample[j] = X_vec[i * n_features + j];

        std::vector<double> alpha_sum(n_classes, 0.0);
        std::vector<double> mean_sum(n_classes,  0.0);

        for (int t = 0; t < n_trees; t++) {
            XPtr<Node> tree_ptr(as<SEXP>(forest_ptrs[t]));
            Node* leaf = FindLeafNode1D(tree_ptr, sample);

            for (int j = 0; j < n_classes; j++) {
                alpha_sum[j] += leaf->alpha_prediction[j];
                mean_sum[j]  += leaf->mean_prediction[j];
            }
        }

        for (int j = 0; j < n_classes; j++) {
            alpha_out(i, j) = alpha_sum[j] / n_trees;
            mean_out(i,  j) = mean_sum[j]  / n_trees;
        }
    }

    return List::create(
        Named("alpha_predictions") = alpha_out,
        Named("mean_predictions")  = mean_out
    );
}

// ============================================================================
// COMPUTE WEIGHTS (store_samples = TRUE)
// ============================================================================
std::unordered_map<int, double> ComputeWeights(
        const std::vector<double>& sample,
        const List&                forest_ptrs,
        int                        n_trees) {

    std::unordered_map<int, double> weights;

    for (int t = 0; t < n_trees; t++) {
        XPtr<Node> tree_ptr(as<SEXP>(forest_ptrs[t]));
        Node* leaf = FindLeafNode1D(tree_ptr, sample);

        if (leaf->leaf_samples.empty()) continue;

        double w = 1.0 / leaf->leaf_samples.size();
        for (int idx : leaf->leaf_samples) weights[idx] += w;
    }

    double total = 0.0;
    for (const auto& e : weights) total += e.second;
    if (total > 0.0)
        for (auto& e : weights) e.second /= total;

    return weights;
}

// ============================================================================
// WEIGHT-BASED PREDICTION (distributional path)
// NOTE: This function is not reachable through the public R API in the current
// version. PredictDirichletForest hardcodes use_leaf_predictions = TRUE, so
// GetLeafPredictions is always used instead. Distributional predictions are
// exposed via PredictWeights and DrawConditionalSample. This function is
// retained for potential future use.
// ============================================================================
// [[Rcpp::export]]
List PredictDirichletForestWeightBased(List forest_model,
                                       NumericMatrix X_new,
                                       std::string method = "mom") {
    List          forest_ptrs = forest_model["forest"];
    int           n_trees     = forest_model["n_trees"];
    int           n_classes   = forest_model["n_classes"];
    int           n_samples   = X_new.nrow();
    int           n_features  = X_new.ncol();
    NumericMatrix Y_train_mat = forest_model["Y_train"];

    int n_train = Y_train_mat.nrow();

    // Copy training Y to STL
    std::vector<double> Y_train(n_train * n_classes);
    for (int i = 0; i < n_train; i++)
        for (int j = 0; j < n_classes; j++)
            Y_train[i * n_classes + j] = Y_train_mat(i, j);

    NumericMatrix alpha_out(n_samples, n_classes);
    NumericMatrix mean_out(n_samples,  n_classes);

    for (int i = 0; i < n_samples; i++) {
        std::vector<double> sample(n_features);
        for (int j = 0; j < n_features; j++)
            sample[j] = X_new(i, j);

        auto weights = ComputeWeights(sample, forest_ptrs, n_trees);

        std::vector<int>    widx;
        std::vector<double> wval;
        for (const auto& e : weights)
            if (e.second > 1e-10) {
                widx.push_back(e.first);
                wval.push_back(e.second);
            }

        if (widx.empty()) {
            for (int j = 0; j < n_classes; j++) {
                alpha_out(i, j) = 1.0;
                mean_out(i,  j) = 1.0 / n_classes;
            }
            continue;
        }

        // Weighted mean
        std::vector<double> mean_pred(n_classes, 0.0);
        for (int k = 0; k < (int)widx.size(); k++)
            for (int j = 0; j < n_classes; j++)
                mean_pred[j] += wval[k] * Y_train[widx[k] * n_classes + j];

        // Replicated dataset for alpha estimation
        const int rep_factor = 100;
        std::vector<int> rep_idx;
        for (int k = 0; k < (int)widx.size(); k++) {
            int nr = std::max(1, (int)(wval[k] * rep_factor));
            for (int r = 0; r < nr; r++) rep_idx.push_back(widx[k]);
        }

        std::vector<double> alpha_pred = (method == "mle")
            ? estimate_mle(Y_train, rep_idx, n_classes)
            : estimate_mom(Y_train, rep_idx, n_classes);

        for (int j = 0; j < n_classes; j++) {
            alpha_out(i, j) = alpha_pred[j];
            mean_out(i,  j) = mean_pred[j];
        }
    }

    return List::create(
        Named("alpha_predictions") = alpha_out,
        Named("mean_predictions")  = mean_out
    );
}

// ============================================================================
// UNIFIED PREDICTION
// ============================================================================
// [[Rcpp::export]]
List PredictDirichletForest(List forest_model, NumericMatrix X_new,
                            std::string method = "mom",
                            bool use_leaf_predictions = true) {
    bool store_samples = as<bool>(forest_model["store_samples"]);

    if (!store_samples || use_leaf_predictions)
        return GetLeafPredictions(forest_model, X_new);
    else
        return PredictDirichletForestWeightBased(forest_model, X_new, method);
}

// ============================================================================
// DRAW CONDITIONAL SAMPLE
// ============================================================================
// Given a single test covariate vector, compute forest proximity weights over
// training observations and return a weighted-bootstrap draw of `size` rows
// from the training Y matrix.  Requires store_samples = TRUE.
// [[Rcpp::export]]
NumericMatrix DrawConditionalSample(List forest_model,
                                    NumericVector test_sample,
                                    int size) {
    List          forest_ptrs = forest_model["forest"];
    int           n_trees     = forest_model["n_trees"];
    NumericMatrix Y_train_mat = forest_model["Y_train"];
    int           n_classes   = Y_train_mat.ncol();
    int           n_train     = Y_train_mat.nrow();

    if (size < 1)
        stop("size must be a positive integer");

    std::vector<double> x(test_sample.begin(), test_sample.end());
    auto weights = ComputeWeights(x, forest_ptrs, n_trees);

    // Collect non-zero weight indices and their probabilities
    std::vector<int>    widx;
    std::vector<double> wval;
    for (const auto& e : weights)
        if (e.second > 1e-10) {
            widx.push_back(e.first);
            wval.push_back(e.second);
        }

    NumericMatrix out(size, n_classes);

    if (widx.empty()) {
        // Fallback: uniform draw from all training rows
        RNGScope rng;
        IntegerVector pool = sample(n_train, size, true);
        for (int i = 0; i < size; i++)
            for (int j = 0; j < n_classes; j++)
                out(i, j) = Y_train_mat(pool[i] - 1, j);
        return out;
    }

    // Weighted-bootstrap draw using R's sample machinery
    int       nw = (int)widx.size();
    NumericVector probs(nw);
    for (int k = 0; k < nw; k++) probs[k] = wval[k];

    RNGScope rng;
    IntegerVector draws = sample(nw, size, true, probs);

    for (int i = 0; i < size; i++) {
        int train_idx = widx[draws[i] - 1];   // R is 1-indexed
        for (int j = 0; j < n_classes; j++)
            out(i, j) = Y_train_mat(train_idx, j);
    }

    return out;
}

// ============================================================================
// OOB PROXIMITY / WEIGHT MATRIX (distributional + OOB path)
// ============================================================================
// [[Rcpp::export]]
NumericMatrix OOBWeightMatrix(List forest_model, NumericMatrix X_train) {

    List forest_ptrs  = forest_model["forest"];
    List oob_idx_list = forest_model["oob_indices"];
    int  n_trees      = forest_model["n_trees"];
    int  n_features   = X_train.ncol();
    int  n            = X_train.nrow();

    std::vector<double> X_vec(n * n_features);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n_features; j++)
            X_vec[i * n_features + j] = X_train(i, j);

    std::vector<std::vector<double>> W(n, std::vector<double>(n, 0.0));
    std::vector<int> oob_tree_count(n, 0);

    for (int b = 0; b < n_trees; b++) {
        IntegerVector oob_r = oob_idx_list[b];
        if (oob_r.size() == 0) continue;

        XPtr<Node> tree_ptr(as<SEXP>(forest_ptrs[b]));

        for (int k = 0; k < (int)oob_r.size(); k++) {
            int i = oob_r[k] - 1;

            std::vector<double> xi(n_features);
            for (int j = 0; j < n_features; j++)
                xi[j] = X_vec[i * n_features + j];

            Node* leaf = FindLeafNode1D(tree_ptr, xi);
            if (leaf->leaf_samples.empty()) continue;

            double w = 1.0 / leaf->leaf_samples.size();
            for (int idx : leaf->leaf_samples)
                W[i][idx] += w;

            oob_tree_count[i]++;
        }
    }

    NumericMatrix out(n, n);
    for (int i = 0; i < n; i++) {
        double denom = (oob_tree_count[i] > 0) ? oob_tree_count[i] : 1.0;
        for (int j = 0; j < n; j++)
            out(i, j) = W[i][j] / denom;
    }

    return out;
}

// ============================================================================
// WEIGHT MATRIX FOR NEW OBSERVATIONS
// ============================================================================
// [[Rcpp::export]]
NumericMatrix PredictWeights(List forest_model, NumericMatrix X_new) {

    List forest_ptrs = forest_model["forest"];
    int  n_trees     = forest_model["n_trees"];
    int  n_features  = X_new.ncol();
    int  n_test      = X_new.nrow();
    int  n_train     = as<NumericMatrix>(forest_model["Y_train"]).nrow();

    NumericMatrix out(n_test, n_train);

    for (int i = 0; i < n_test; i++) {
        std::vector<double> xi(n_features);
        for (int j = 0; j < n_features; j++)
            xi[j] = X_new(i, j);

        auto weights = ComputeWeights(xi, forest_ptrs, n_trees);
        for (const auto& e : weights)
            out(i, e.first) = e.second;
    }

    return out;
}





// [[Rcpp::export]]
DataFrame PermutationImportance(List forest_model,
                                NumericMatrix X,
                                NumericMatrix Y,
                                std::string loss = "aitchison",
                                int num_permutations = 5,
                                int seed = 42) {

    List   forest_ptrs = forest_model["forest"];
    List   oob_idx_list = forest_model["oob_indices"];
    int    n_trees   = forest_model["n_trees"];
    int    n         = X.nrow();
    int    p         = X.ncol();
    int    k         = Y.ncol();

    // ── Copy to STL once ────────────────────────────────────────────────────
    std::vector<double> X_vec(n * p);
    std::vector<double> Y_vec(n * k);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) X_vec[i * p + j] = X(i, j);
        for (int j = 0; j < k; j++) Y_vec[i * k + j] = Y(i, j);
    }

    // ── Loss functions ───────────────────────────────────────────────────────
    // Each takes true and predicted rows (as flat STL slices) and returns
    // a scalar error over the OOB sample.
    const double eps = 1e-10;

    auto aitchison_loss = [&](const std::vector<int>& rows,
                               const std::vector<std::vector<double>>& pred) {
        double acc = 0.0;
        for (int ii = 0; ii < (int)rows.size(); ii++) {
            int i = rows[ii];
            // clr of true
            double lm_true = 0.0, lm_pred = 0.0;
            for (int j = 0; j < k; j++) {
                lm_true += std::log(Y_vec[i * k + j] + eps);
                lm_pred += std::log(pred[ii][j]       + eps);
            }
            lm_true /= k; lm_pred /= k;
            double d = 0.0;
            for (int j = 0; j < k; j++) {
                double diff = (std::log(Y_vec[i * k + j] + eps) - lm_true)
                            - (std::log(pred[ii][j]       + eps) - lm_pred);
                d += diff * diff;
            }
            acc += std::sqrt(d);
        }
        return acc / rows.size();
    };

    auto mse_loss = [&](const std::vector<int>& rows,
                         const std::vector<std::vector<double>>& pred) {
        double acc = 0.0;
        for (int ii = 0; ii < (int)rows.size(); ii++) {
            int i = rows[ii];
            for (int j = 0; j < k; j++) {
                double r = Y_vec[i * k + j] - pred[ii][j];
                acc += r * r;
            }
        }
        return acc / (rows.size() * k);
    };

    auto kl_loss = [&](const std::vector<int>& rows,
                        const std::vector<std::vector<double>>& pred) {
        double acc = 0.0;
        for (int ii = 0; ii < (int)rows.size(); ii++) {
            int i = rows[ii];
            for (int j = 0; j < k; j++) {
                double y  = Y_vec[i * k + j] + eps;
                double yh = pred[ii][j]       + eps;
                acc += y * std::log(y / yh);
            }
        }
        return acc / rows.size();
    };

    // ── Helper: predict OOB rows with a single tree ──────────────────────────
    // Returns a (rows.size() x k) block of mean predictions.
    auto predict_rows = [&](Node* root,
                             const std::vector<int>& rows,
                             const std::vector<double>& X_use) {
        std::vector<std::vector<double>> out(rows.size(),
                                             std::vector<double>(k));
        for (int ii = 0; ii < (int)rows.size(); ii++) {
            int i = rows[ii];
            std::vector<double> xi(p);
            for (int j = 0; j < p; j++) xi[j] = X_use[i * p + j];
            Node* leaf = FindLeafNode1D(root, xi);
            out[ii] = leaf->mean_prediction;
        }
        return out;
    };

    // ── RNG ─────────────────────────────────────────────────────────────────
    std::mt19937 gen(seed);

    // ── Main loop: outer = trees, inner = features ───────────────────────────
    // vi_matrix[b][j] = mean delta error for tree b, feature j
    std::vector<std::vector<double>> vi_matrix(n_trees,
                                               std::vector<double>(p, 0.0));

    for (int b = 0; b < n_trees; b++) {

        IntegerVector oob_r = oob_idx_list[b];
        if ((int)oob_r.size() < 2) continue;

        // 0-based OOB indices
        std::vector<int> oob_rows(oob_r.size());
        for (int ii = 0; ii < (int)oob_r.size(); ii++)
            oob_rows[ii] = oob_r[ii] - 1;

        XPtr<Node> tree_ptr(as<SEXP>(forest_ptrs[b]));
        Node* root = tree_ptr;

        // Baseline error — predict once per tree
        auto baseline_pred = predict_rows(root, oob_rows, X_vec);
        double baseline_err = (loss == "aitchison") ? aitchison_loss(oob_rows, baseline_pred)
                            : (loss == "mse")       ? mse_loss      (oob_rows, baseline_pred)
                                                    : kl_loss       (oob_rows, baseline_pred);

        // Extract OOB X block into a local copy we can permute
        // Shape: oob_rows.size() x p, stored flat row-major
        int m = (int)oob_rows.size();
        std::vector<double> X_oob(m * p);
        for (int ii = 0; ii < m; ii++)
            for (int j = 0; j < p; j++)
                X_oob[ii * p + j] = X_vec[oob_rows[ii] * p + j];

        for (int j = 0; j < p; j++) {

            double perm_err_sum = 0.0;

            for (int r = 0; r < num_permutations; r++) {

                // Permute column j in-place, restore after
                std::vector<double> col_backup(m);
                for (int ii = 0; ii < m; ii++)
                    col_backup[ii] = X_oob[ii * p + j];

                std::vector<int> perm_order(m);
                std::iota(perm_order.begin(), perm_order.end(), 0);
                std::shuffle(perm_order.begin(), perm_order.end(), gen);
                for (int ii = 0; ii < m; ii++)
                    X_oob[ii * p + j] = col_backup[perm_order[ii]];

                // Predict on permuted X_oob
                // Need predict_rows to index into X_oob, not X_vec.
                // Use a local flat X that maps oob_rows[ii] -> X_oob row ii.
                // Easiest: build a temporary full-size X with oob rows swapped.
                // Better: adapt predict_rows to take a compact matrix directly.
                std::vector<std::vector<double>> perm_pred(m,
                    std::vector<double>(k));
                for (int ii = 0; ii < m; ii++) {
                    std::vector<double> xi(p);
                    for (int jj = 0; jj < p; jj++)
                        xi[jj] = X_oob[ii * p + jj];
                    Node* leaf = FindLeafNode1D(root, xi);
                    perm_pred[ii] = leaf->mean_prediction;
                }

                perm_err_sum += (loss == "aitchison")
                    ? aitchison_loss(oob_rows, perm_pred)
                    : (loss == "mse") ? mse_loss(oob_rows, perm_pred)
                                      : kl_loss (oob_rows, perm_pred);

                // Restore column j
                for (int ii = 0; ii < m; ii++)
                    X_oob[ii * p + j] = col_backup[ii];
            }

            vi_matrix[b][j] = (perm_err_sum / num_permutations) - baseline_err;
        }
    }

    // ── Aggregate ────────────────────────────────────────────────────────────
    std::vector<double> vi_mean(p, 0.0);
    std::vector<double> vi_sq  (p, 0.0);

    for (int b = 0; b < n_trees; b++)
        for (int j = 0; j < p; j++) {
            vi_mean[j] += vi_matrix[b][j];
            vi_sq[j]   += vi_matrix[b][j] * vi_matrix[b][j];
        }

    NumericVector importance(p), importance_sd(p), importance_scaled(p);
    for (int j = 0; j < p; j++) {
        double mean = vi_mean[j] / n_trees;
        double var  = vi_sq[j] / n_trees - mean * mean;
        double sd   = (var > 0.0) ? std::sqrt(var) : 0.0;
        importance[j]        = mean;
        importance_sd[j]     = sd;
        importance_scaled[j] = (sd > 0.0) ? mean / sd : 0.0;
    }

    return DataFrame::create(
        Named("importance")        = importance,
        Named("importance_scaled") = importance_scaled,
        Named("importance_sd")     = importance_sd
    );
}
