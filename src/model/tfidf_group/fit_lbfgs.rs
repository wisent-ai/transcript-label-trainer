use super::*;
use std::collections::VecDeque;

/// The fit's starting point is the origin, as scikit-learn's lbfgs solver's is.
// https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_logistic.py : w0 = np.zeros(...) unless warm_start.
const ORIGIN: f64 = 0.0;

/// The full quasi-Newton step every line search tries first.
// https://github.com/scipy/scipy/blob/main/scipy/optimize/_linesearch.py : scalar_search_armijo starts at alpha0 = 1.
const FULL_STEP: f64 = 1.0;

/// The two in the minimiser of the quadratic through f(0), f'(0) and f(a):
/// a = -f'(0) a^2 / (2 (f(a) - f(0) - f'(0) a)).
// https://github.com/scipy/scipy/blob/main/scipy/optimize/_linesearch.py : scalar_search_armijo's quadratic step.
const QUADRATIC_DENOMINATOR: f64 = 2.0;

/// L-BFGS with a two-loop recursion, the stated number of correction pairs
/// and an Armijo backtracking line search. A curvature pair is kept only
/// when `s·y` exceeds machine precision times the descent along the step,
/// `-g·s`, the test L-BFGS-B's mainlb applies, which keeps the implicit
/// inverse Hessian positive definite without needing a full Wolfe search.
pub(crate) fn fit_lbfgs(
    problem: &Problem<'_>,
    settings: &jobs::TfidfTraining,
) -> (Vec<f64>, usize, bool) {
    let mut x = vec![ORIGIN; problem.dim()];
    let (mut fx, mut g) = problem.loss_grad(&x);
    // Each kept pair: s, y and rho = 1 / (s·y), oldest first.
    let mut history: VecDeque<(Vec<f64>, Vec<f64>, f64)> = VecDeque::new();

    for iterations in 0..settings.max_iter {
        let gradient_max = g.iter().fold(0.0, |acc: f64, v| acc.max(v.abs()));
        if gradient_max <= settings.tol {
            return (x, iterations, true);
        }

        // https://en.wikipedia.org/wiki/Limited-memory_BFGS#Algorithm : the two-loop recursion.
        let mut q = g.clone();
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y, rho) in history.iter().rev() {
            let a = rho * dot(s, &q);
            alphas.push(a);
            for (slot, &yi) in q.iter_mut().zip(y) {
                *slot -= a * yi;
            }
        }
        if let Some((s, y, _)) = history.back() {
            let yy = dot(y, y);
            if yy > 0.0 {
                let gamma = dot(s, y) / yy;
                for slot in q.iter_mut() {
                    *slot *= gamma;
                }
            }
        }
        for ((s, y, rho), alpha) in history.iter().zip(alphas.iter().rev()) {
            let beta = rho * dot(y, &q);
            let scale = alpha - beta;
            for (slot, &si) in q.iter_mut().zip(s) {
                *slot += scale * si;
            }
        }

        let mut direction: Vec<f64> = q.iter().map(|v| -v).collect();
        let mut slope = dot(&direction, &g);
        if !(slope < 0.0) {
            // Not a descent direction, or not finite: take steepest descent,
            // which always is one.
            direction = g.iter().map(|v| -v).collect();
            slope = dot(&direction, &g);
        }

        // Armijo backtracking that picks each next step by minimising the
        // quadratic through f(0), f'(0) and f(step), as SciPy's
        // scalar_search_armijo does, so no shrink factor is chosen; the step
        // starts at the full quasi-Newton step and tries at most
        // max_backtracks candidates.
        let mut step = FULL_STEP;
        let mut accepted = None;
        for _ in 0..settings.max_backtracks {
            let candidate: Vec<f64> = x
                .iter()
                .zip(&direction)
                .map(|(xi, di)| xi + step * di)
                .collect();
            let (candidate_f, candidate_g) = problem.loss_grad(&candidate);
            if candidate_f.is_finite() && candidate_f <= fx + settings.armijo_c1 * step * slope {
                accepted = Some((candidate, candidate_f, candidate_g));
                break;
            }
            let curvature = candidate_f - fx - slope * step;
            let next = -slope * step * step / (QUADRATIC_DENOMINATOR * curvature);
            if !(next.is_finite() && next > f64::MIN_POSITIVE && next < step) {
                break;
            }
            step = next;
        }
        let Some((new_x, new_f, new_g)) = accepted else {
            // No step along a descent direction lowers the objective: the
            // iterate is at the precision floor, which is as converged as
            // this arithmetic gets.
            return (x, iterations, true);
        };

        let s: Vec<f64> = new_x.iter().zip(&x).map(|(new, old)| new - old).collect();
        let y: Vec<f64> = new_g.iter().zip(&g).map(|(new, old)| new - old).collect();
        let sy = dot(&s, &y);
        // https://users.iems.northwestern.edu/~nocedal/lbfgsb.html : mainlb in lbfgsb.f skips the update when s·y (dr) is not above machine precision times -g·s (ddum), the old gradient's descent along the step.
        let descent = -dot(&g, &s);
        if sy > f64::EPSILON * descent {
            if history.len() == settings.lbfgs_memory {
                history.pop_front();
            }
            history.push_back((s, y, sy.recip()));
        }
        x = new_x;
        fx = new_f;
        g = new_g;
    }
    (x, settings.max_iter, false)
}

pub(crate) fn fit_logistic(
    rows: &[SparseRow],
    y: &[usize],
    n_classes: usize,
    n_features: usize,
    settings: &jobs::TfidfTraining,
) -> Fit {
    let problem = Problem {
        rows,
        y,
        n_classes,
        n_features,
        l2: settings.c.recip(),
    };
    let (x, iterations, converged) = fit_lbfgs(&problem, settings);
    let coef = (0..n_classes)
        .map(|c| x[c * n_features..(c + 1) * n_features].to_vec())
        .collect();
    let intercept = x[n_classes * n_features..].to_vec();
    Fit {
        coef,
        intercept,
        iterations,
        converged,
    }
}

/// (class index, probability) for one row, first maximum winning — the same
/// tie-break `numpy.argmax` gives, over the same sorted class order.
pub(crate) fn predict_row(coef: &[Vec<f64>], intercept: &[f64], row: &SparseRow) -> (usize, f64) {
    let z: Vec<f64> = coef
        .iter()
        .zip(intercept)
        .map(|(weights, &bias)| {
            row.iter().fold(bias, |score, &(column, value)| {
                score + weights[column as usize] * value
            })
        })
        .collect();
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = z.iter().map(|s| (s - max).exp()).sum();
    let (best, best_score) = z
        .iter()
        .copied()
        .enumerate()
        .reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .expect("a fitted model holds at least one class");
    (best, (best_score - max).exp() / sum_exp)
}

// ---------------------------------------------------------------------------
// The tfidf-logreg artifact
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub(crate) struct VectorizerFile {
    pub(crate) analyzer: String,
    pub(crate) lowercase: bool,
    pub(crate) token_pattern: String,
    // https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html : ngram_range is the pair (min_n, max_n).
    pub(crate) ngram_range: [usize; 2],
    pub(crate) sublinear_tf: bool,
    pub(crate) smooth_idf: bool,
    pub(crate) norm: String,
    pub(crate) min_df: f64,
    pub(crate) max_df: f64,
    pub(crate) vocabulary: Vec<String>,
    pub(crate) idf: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ClassifierFile {
    pub(crate) kind: String,
    pub(crate) multi_class: String,
    #[serde(rename = "C")]
    pub(crate) c: f64,
    pub(crate) max_iter: usize,
    pub(crate) tol: f64,
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
    pub(crate) classes: Vec<String>,
    pub(crate) intercept: Vec<f64>,
    pub(crate) coef: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ModelFile {
    pub(crate) backend: String,
    pub(crate) format: u32,
    pub(crate) vectorizer: VectorizerFile,
    pub(crate) classifier: ClassifierFile,
}

pub(crate) struct TfidfModel {
    pub(crate) vectorizer: Vectorizer,
    pub(crate) classes: Vec<String>,
    pub(crate) coef: Vec<Vec<f64>>,
    pub(crate) intercept: Vec<f64>,
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
}
