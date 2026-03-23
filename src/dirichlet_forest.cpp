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
                          const std::string&         method) {
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
               bool                       store_samples) {
    Node* node = new Node();

    if ((int)indices.size() < n_min || current_depth >= d_max ||
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
                                      n_features, n_classes, method);

    if (split.gain <= 0.0 || split.feature == -1) {
        FitTerminalNode(node, Y_vec, indices, n_classes, method, store_samples);
        return node;
    }

    node->feature_index = split.feature;
    node->split_value   = split.split_value;
    node->is_leaf       = false;

    node->left  = GrowTree(X_vec, Y_vec, n_features, n_classes,
                            split.left_indices,  current_depth + 1,
                            d_max, n_min, m_try, gen, method, store_samples);
    node->right = GrowTree(X_vec, Y_vec, n_features, n_classes,
                            split.right_indices, current_depth + 1,
                            d_max, n_min, m_try, gen, method, store_samples);

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
                     int num_cores = 1) {

    int n_samples  = X.nrow();
    int n_features = X.ncol();
    int n_classes  = Y.ncol();

    if (m_try <= 0) m_try = std::max(1, (int)std::sqrt((double)n_features));

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

    // Parallel tree building — all code inside uses pure STL only
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int b = 0; b < B; b++) {
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), generators[b]);

        forest[b] = GrowTree(X_vec, Y_vec, n_features, n_classes,
                             indices, 0, d_max, n_min, m_try,
                             generators[b], method, store_samples);
    }

    // Wrap back into Rcpp AFTER parallel region
    List forest_ptrs(B);
    for (int i = 0; i < B; i++)
        forest_ptrs[i] = XPtr<Node>(forest[i], true);

    List result = List::create(
        Named("forest")        = forest_ptrs,
        Named("n_trees")       = B,
        Named("n_features")    = n_features,
        Named("n_classes")     = n_classes,
        Named("store_samples") = store_samples
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
// GET SAMPLE WEIGHTS (analysis utility)
// ============================================================================
// [[Rcpp::export]]
List GetSampleWeights(List forest_model, NumericVector test_sample) {
    List          forest_ptrs = forest_model["forest"];
    int           n_trees     = forest_model["n_trees"];
    NumericMatrix Y_train_mat = forest_model["Y_train"];
    int           n_classes   = Y_train_mat.ncol();
    int           n_train     = Y_train_mat.nrow();

    std::vector<double> sample(test_sample.begin(), test_sample.end());
    auto weights = ComputeWeights(sample, forest_ptrs, n_trees);

    std::vector<int>    sidx;
    std::vector<double> sval;
    for (const auto& e : weights) {
        sidx.push_back(e.first);
        sval.push_back(e.second);
    }

    int           nw = (int)sidx.size();
    NumericMatrix Y_weighted(nw, n_classes);

    std::vector<double> Y_train(n_train * n_classes);
    for (int i = 0; i < n_train; i++)
        for (int j = 0; j < n_classes; j++)
            Y_train[i * n_classes + j] = Y_train_mat(i, j);

    for (int i = 0; i < nw; i++)
        for (int j = 0; j < n_classes; j++)
            Y_weighted(i, j) = Y_train[sidx[i] * n_classes + j];

    return List::create(
        Named("sample_indices") = sidx,
        Named("weights")        = sval,
        Named("Y_values")       = Y_weighted
    );
}



