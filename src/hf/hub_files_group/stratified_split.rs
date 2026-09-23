use super::*;

/// A stratified holdout of `n_test` rows with every class present on both
/// sides: what `train_test_split(..., stratify=label_ids, random_state=0)`
/// produced for the in-training slice.
pub(crate) fn stratified_split(
    label_ids: &[usize],
    n_classes: usize,
    n_test: usize,
    rng: &mut ChaCha8Rng,
) -> (Vec<usize>, Vec<usize>) {
    let mut per_class: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (row, label) in label_ids.iter().enumerate() {
        per_class[*label].push(row);
    }
    for group in per_class.iter_mut() {
        group.shuffle(rng);
    }

    let fraction = n_test as f64 / label_ids.len().max(1) as f64;
    // Every class keeps at least one row on each side; that is what the extra
    // minimum of 2 sessions per class buys.
    let mut take: Vec<usize> = per_class
        .iter()
        .map(|group| {
            ((group.len() as f64 * fraction).round() as usize)
                .clamp(1, group.len().saturating_sub(1).max(1))
        })
        .collect();

    // Per-class rounding rarely lands on exactly n_test, so single rows move
    // between the sides, largest class first, until it does or nothing can
    // move without emptying a side.
    loop {
        let total: usize = take.iter().sum();
        if total == n_test {
            break;
        }
        let grow = total < n_test;
        let candidate = (0..n_classes)
            .filter(|class| {
                if grow {
                    take[*class] + 1 < per_class[*class].len()
                } else {
                    take[*class] > 1
                }
            })
            .max_by_key(|class| per_class[*class].len());
        match candidate {
            Some(class) if grow => take[class] += 1,
            Some(class) => take[class] -= 1,
            None => break,
        }
    }

    let mut test = Vec::with_capacity(n_test);
    let mut train = Vec::with_capacity(label_ids.len().saturating_sub(n_test));
    for (class, group) in per_class.iter().enumerate() {
        let (held, kept) = group.split_at(take[class].min(group.len()));
        test.extend_from_slice(held);
        train.extend_from_slice(kept);
    }
    train.sort_unstable();
    test.sort_unstable();
    (train, test)
}

/// Clip the gradients to a global L2 norm, the `max_grad_norm=1.0` every
/// transformers `Trainer` applies by default.
pub(crate) fn clip_grads(
    grads: &mut candle_core::backprop::GradStore,
    vars: &[Var],
    max_norm: f64,
) -> Result<()> {
    let mut squares = Vec::with_capacity(vars.len());
    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let square = grad
                .sqr()
                .and_then(|grad| grad.sum_all())
                .map_err(|err| Error(format!("could not measure a gradient: {err}")))?;
            squares.push(square);
        }
    }
    if squares.is_empty() {
        return Err(Error(
            "no parameter received a gradient; the fine-tune would not change the model".into(),
        ));
    }
    let total = Tensor::stack(&squares, 0)
        .and_then(|squares| squares.sum_all())
        .map_err(|err| Error(format!("could not measure the gradient norm: {err}")))?;
    let norm = (scalar(&total)? as f64).sqrt();
    if !norm.is_finite() {
        return Err(Error(
            "the gradient norm is not finite; the fine-tune diverged, lower --lr and retry".into(),
        ));
    }
    if norm <= max_norm {
        return Ok(());
    }
    let scale = max_norm / (norm + 1e-6);
    let mut scaled = Vec::with_capacity(vars.len());
    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let clipped =
                (grad * scale).map_err(|err| Error(format!("could not clip a gradient: {err}")))?;
            scaled.push((var.as_tensor().clone(), clipped));
        }
    }
    for (tensor, grad) in scaled {
        grads.insert(&tensor, grad);
    }
    Ok(())
}

/// The winning class per row of a logits tensor.
pub(crate) fn argmax(logits: &Tensor) -> Result<Vec<u32>> {
    logits
        .argmax(D::Minus1)
        .and_then(|best| best.to_vec1::<u32>())
        .map_err(|err| Error(format!("could not read the predicted classes: {err}")))
}

pub(crate) fn scalar(tensor: &Tensor) -> Result<f32> {
    tensor.to_scalar::<f32>().map_err(|err| {
        Error(format!(
            "could not read a scalar back from the device: {err}"
        ))
    })
}

/// Python's `round(value, 4)`, which is what the metrics have always carried.
pub(crate) fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}
