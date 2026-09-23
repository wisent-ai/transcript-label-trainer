use super::*;

/// L-BFGS with a two-loop recursion, `LBFGS_MEMORY` correction pairs and an
/// Armijo backtracking line search. Curvature pairs whose `s·y` is not
/// positive are skipped rather than stored, which keeps the implicit inverse
/// Hessian positive definite without needing a full Wolfe search.
pub(crate) fn fit_lbfgs(problem: &Problem<'_>) -> (Vec<f64>, usize, bool) {
    let dim = problem.dim();
    let mut x = vec![0.0f64; dim];
    let (mut fx, mut g) = problem.loss_grad(&x);
    let mut s_history: Vec<Vec<f64>> = Vec::new();
    let mut y_history: Vec<Vec<f64>> = Vec::new();
    let mut rho: Vec<f64> = Vec::new();
    let mut iterations = 0usize;

    while iterations < MAX_ITER {
        let gradient_max = g.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        if gradient_max <= TOL {
            return (x, iterations, true);
        }

        let mut q = g.clone();
        let mut alpha = vec![0.0f64; s_history.len()];
        for i in (0..s_history.len()).rev() {
            let a = rho[i] * dot(&s_history[i], &q);
            alpha[i] = a;
            for (slot, &yi) in q.iter_mut().zip(y_history[i].iter()) {
                *slot -= a * yi;
            }
        }
        if let Some(last) = s_history.len().checked_sub(1) {
            let yy = dot(&y_history[last], &y_history[last]);
            if yy > 0.0 {
                let gamma = dot(&s_history[last], &y_history[last]) / yy;
                for slot in q.iter_mut() {
                    *slot *= gamma;
                }
            }
        }
        for i in 0..s_history.len() {
            let beta = rho[i] * dot(&y_history[i], &q);
            let scale = alpha[i] - beta;
            for (slot, &si) in q.iter_mut().zip(s_history[i].iter()) {
                *slot += scale * si;
            }
        }

        let mut direction: Vec<f64> = q.iter().map(|v| -v).collect();
        let mut slope = dot(&direction, &g);
        if !(slope < 0.0) {
            // Not a descent direction, or not finite: fall back to steepest
            // descent, which always is one.
            direction = g.iter().map(|v| -v).collect();
            slope = dot(&direction, &g);
        }

        let mut step = 1.0f64;
        let mut accepted = None;
        for _ in 0..MAX_BACKTRACKS {
            let candidate: Vec<f64> = (0..dim).map(|i| x[i] + step * direction[i]).collect();
            let (candidate_f, candidate_g) = problem.loss_grad(&candidate);
            if candidate_f.is_finite() && candidate_f <= fx + ARMIJO_C1 * step * slope {
                accepted = Some((candidate, candidate_f, candidate_g));
                break;
            }
            step *= BACKTRACK;
        }
        let Some((new_x, new_f, new_g)) = accepted else {
            // No step along a descent direction lowers the objective: the
            // iterate is at the precision floor, which is as converged as
            // this arithmetic gets.
            return (x, iterations, true);
        };

        let s: Vec<f64> = (0..dim).map(|i| new_x[i] - x[i]).collect();
        let y: Vec<f64> = (0..dim).map(|i| new_g[i] - g[i]).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 {
            if s_history.len() == LBFGS_MEMORY {
                s_history.remove(0);
                y_history.remove(0);
                rho.remove(0);
            }
            rho.push(1.0 / sy);
            s_history.push(s);
            y_history.push(y);
        }
        x = new_x;
        fx = new_f;
        g = new_g;
        iterations += 1;
    }
    (x, iterations, false)
}

pub(crate) fn fit_logistic(rows: &[SparseRow], y: &[usize], n_classes: usize, n_features: usize) -> Fit {
    let problem = Problem {
        rows,
        y,
        n_classes,
        n_features,
    };
    let (x, iterations, converged) = fit_lbfgs(&problem);
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
    let k = intercept.len();
    let mut z = Vec::with_capacity(k);
    for c in 0..k {
        let weights = &coef[c];
        let mut score = intercept[c];
        for &(column, value) in row {
            score += weights[column as usize] * value;
        }
        z.push(score);
    }
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = z.iter().map(|s| (s - max).exp()).sum();
    let mut best = 0usize;
    for c in 1..k {
        if z[c] > z[best] {
            best = c;
        }
    }
    (best, (z[best] - max).exp() / sum_exp)
}

// ---------------------------------------------------------------------------
// The tfidf-logreg artifact
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub(crate) struct VectorizerFile {
    pub(crate) analyzer: String,
    pub(crate) lowercase: bool,
    pub(crate) token_pattern: String,
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
