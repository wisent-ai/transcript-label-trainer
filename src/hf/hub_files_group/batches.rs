use super::*;

/// One encoded split, padded to a single width so any subset of its rows can
/// be stacked into a batch. Inference rows carry no labels.
pub(crate) struct Batches {
    pub(crate) ids: Vec<u32>,
    pub(crate) mask: Vec<u32>,
    pub(crate) labels: Vec<usize>,
    pub(crate) width: usize,
    pub(crate) rows: usize,
    pub(crate) device: Device,
}

impl Batches {
    /// The texts `index` names, with their labels when `label_ids` holds the
    /// label of every text.
    pub(crate) fn encode(
        tokenizer: &Tokenizer,
        texts: &[String],
        label_ids: Option<&[usize]>,
        index: &[usize],
        device: &Device,
    ) -> Result<Self> {
        if index.is_empty() {
            return Ok(Self {
                ids: Vec::new(),
                mask: Vec::new(),
                labels: Vec::new(),
                width: 0,
                rows: 0,
                device: device.clone(),
            });
        }
        let batch: Vec<&str> = index.iter().map(|row| texts[*row].as_str()).collect();
        let encodings = tokenizer
            .encode_batch(batch, true)
            .map_err(|err| Error(format!("tokenization failed: {err}")))?;
        let mut ids = Vec::new();
        let mut mask = Vec::new();
        for encoding in &encodings {
            ids.extend_from_slice(encoding.get_ids());
            mask.extend_from_slice(encoding.get_attention_mask());
        }
        // Padding gives every row the width of the first.
        let width = encodings.first().map_or(0, |encoding| encoding.get_ids().len());
        let labels = match label_ids {
            Some(label_ids) => index.iter().map(|row| label_ids[*row]).collect(),
            None => Vec::new(),
        };
        Ok(Self {
            ids,
            mask,
            labels,
            width,
            rows: encodings.len(),
            device: device.clone(),
        })
    }

    /// `(input_ids, attention_mask)` for the rows named by `rows`.
    pub(crate) fn inputs(&self, rows: &[usize]) -> Result<(Tensor, Tensor)> {
        let mut ids = Vec::with_capacity(rows.len() * self.width);
        let mut mask = Vec::with_capacity(rows.len() * self.width);
        for row in rows {
            let start = row * self.width;
            ids.extend_from_slice(&self.ids[start..start + self.width]);
            mask.extend_from_slice(&self.mask[start..start + self.width]);
        }
        let shape = (rows.len(), self.width);
        let ids = Tensor::from_vec(ids, shape, &self.device)
            .map_err(|err| Error(format!("could not build the input batch: {err}")))?;
        let mask = Tensor::from_vec(mask, shape, &self.device)
            .map_err(|err| Error(format!("could not build the attention mask: {err}")))?;
        Ok((ids, mask))
    }

    /// `(input_ids, attention_mask, targets)` for the rows named by `rows`;
    /// refused for rows encoded without labels.
    pub(crate) fn take(&self, rows: &[usize]) -> Result<(Tensor, Tensor, Tensor)> {
        let targets = rows
            .iter()
            .map(|row| {
                self.labels
                    .get(*row)
                    .map(|label| *label as u32)
                    .ok_or_else(|| Error(format!("row {row} was encoded without a label")))
            })
            .collect::<Result<Vec<u32>>>()?;
        let (ids, mask) = self.inputs(rows)?;
        let targets = Tensor::from_vec(targets, rows.len(), &self.device)
            .map_err(|err| Error(format!("could not build the target batch: {err}")))?;
        Ok((ids, mask, targets))
    }
}
