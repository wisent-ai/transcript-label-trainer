use super::*;

/// Scaled dot-product attention over already-projected queries, keys and
/// values of shape (batch, tokens, hidden). `padding` is 1 at padding
/// positions, shaped (batch, 1, 1, tokens) so it broadcasts over the heads.
pub(crate) fn attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    heads: usize,
    padding: &Tensor,
) -> candle_core::Result<Tensor> {
    let (batch, tokens, hidden) = query.dims3()?;
    let head_dim = hidden / heads;
    let split = |xs: &Tensor| -> candle_core::Result<Tensor> {
        xs.reshape((batch, tokens, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()
    };
    let query = (split(query)? / (head_dim as f64).sqrt())?;
    let key = split(key)?;
    let value = split(value)?;

    let scores = query.matmul(&key.transpose(2, 3)?.contiguous()?)?;
    // Padding keys are scored at -inf so softmax gives them no weight.
    let blocked = Tensor::new(f32::NEG_INFINITY, scores.device())?.broadcast_as(scores.shape())?;
    let scores = padding
        .broadcast_as(scores.shape())?
        .where_cond(&blocked, &scores)?;
    let weights = ops::softmax(&scores, D::Minus1)?;

    weights
        .matmul(&value)?
        .transpose(1, 2)?
        .reshape((batch, tokens, hidden))
}

pub(crate) struct DistilBertLayer {
    pub(crate) query: Linear,
    pub(crate) key: Linear,
    pub(crate) value: Linear,
    pub(crate) attention_out: Linear,
    pub(crate) attention_norm: LayerNorm,
    pub(crate) lin1: Linear,
    pub(crate) lin2: Linear,
    pub(crate) output_norm: LayerNorm,
    pub(crate) heads: usize,
    pub(crate) activation: Activation,
}

impl DistilBertLayer {
    pub(crate) fn load(
        config: &DistilBertConfig,
        activation: Activation,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let dim = config.dim;
        let attention = vb.pp("attention");
        let ffn = vb.pp("ffn");
        Ok(Self {
            query: candle_nn::linear(dim, dim, attention.pp("q_lin"))?,
            key: candle_nn::linear(dim, dim, attention.pp("k_lin"))?,
            value: candle_nn::linear(dim, dim, attention.pp("v_lin"))?,
            attention_out: candle_nn::linear(dim, dim, attention.pp("out_lin"))?,
            attention_norm: LayerNorm::load(dim, DEFAULT_LAYER_NORM_EPS, vb.pp("sa_layer_norm"))?,
            lin1: candle_nn::linear(dim, config.hidden_dim, ffn.pp("lin1"))?,
            lin2: candle_nn::linear(config.hidden_dim, dim, ffn.pp("lin2"))?,
            output_norm: LayerNorm::load(dim, DEFAULT_LAYER_NORM_EPS, vb.pp("output_layer_norm"))?,
            heads: config.n_heads,
            activation,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor, padding: &Tensor) -> candle_core::Result<Tensor> {
        let context = attention(
            &self.query.forward(xs)?,
            &self.key.forward(xs)?,
            &self.value.forward(xs)?,
            self.heads,
            padding,
        )?;
        let attended = self
            .attention_norm
            .forward(&self.attention_out.forward(&context)?.add(xs)?)?;
        let ffn = self
            .lin2
            .forward(&self.activation.apply(&self.lin1.forward(&attended)?)?)?;
        self.output_norm.forward(&ffn.add(&attended)?)
    }
}

pub(crate) struct DistilBert {
    pub(crate) words: Embedding,
    pub(crate) positions: Embedding,
    pub(crate) norm: LayerNorm,
    pub(crate) layers: Vec<DistilBertLayer>,
}

impl DistilBert {
    pub(crate) fn load(config: &DistilBertConfig, vb: &VarBuilder) -> Result<Self> {
        let activation = Activation::parse(&config.activation)?;
        let embeddings = vb.pp("embeddings");
        let layers = vb.pp("transformer").pp("layer");
        let build = || -> candle_core::Result<Self> {
            Ok(Self {
                words: candle_nn::embedding(
                    config.vocab_size,
                    config.dim,
                    embeddings.pp("word_embeddings"),
                )?,
                positions: candle_nn::embedding(
                    config.max_position_embeddings,
                    config.dim,
                    embeddings.pp("position_embeddings"),
                )?,
                norm: LayerNorm::load(
                    config.dim,
                    DEFAULT_LAYER_NORM_EPS,
                    embeddings.pp("LayerNorm"),
                )?,
                layers: (0..config.n_layers)
                    .map(|index| {
                        DistilBertLayer::load(config, activation, layers.pp(index.to_string()))
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?,
            })
        };
        build().map_err(|err| Error(format!("could not load the distilbert encoder: {err}")))
    }

    pub(crate) fn forward(&self, ids: &Tensor, padding: &Tensor) -> candle_core::Result<Tensor> {
        let (_, tokens) = ids.dims2()?;
        let positions = Tensor::arange(0u32, tokens as u32, ids.device())?;
        let mut hidden = self.norm.forward(
            &self
                .words
                .forward(ids)?
                .broadcast_add(&self.positions.forward(&positions)?)?,
        )?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, padding)?;
        }
        Ok(hidden)
    }
}

pub(crate) struct BertLayer {
    pub(crate) query: Linear,
    pub(crate) key: Linear,
    pub(crate) value: Linear,
    pub(crate) attention_out: Linear,
    pub(crate) attention_norm: LayerNorm,
    pub(crate) intermediate: Linear,
    pub(crate) output: Linear,
    pub(crate) output_norm: LayerNorm,
    pub(crate) heads: usize,
    pub(crate) activation: Activation,
}

impl BertLayer {
    pub(crate) fn load(
        config: &BertConfig,
        activation: Activation,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let hidden = config.hidden_size;
        let eps = config.layer_norm_eps;
        let attention = vb.pp("attention");
        let self_attention = attention.pp("self");
        let attention_output = attention.pp("output");
        let output = vb.pp("output");
        Ok(Self {
            query: candle_nn::linear(hidden, hidden, self_attention.pp("query"))?,
            key: candle_nn::linear(hidden, hidden, self_attention.pp("key"))?,
            value: candle_nn::linear(hidden, hidden, self_attention.pp("value"))?,
            attention_out: candle_nn::linear(hidden, hidden, attention_output.pp("dense"))?,
            attention_norm: LayerNorm::load(hidden, eps, attention_output.pp("LayerNorm"))?,
            intermediate: candle_nn::linear(
                hidden,
                config.intermediate_size,
                vb.pp("intermediate").pp("dense"),
            )?,
            output: candle_nn::linear(config.intermediate_size, hidden, output.pp("dense"))?,
            output_norm: LayerNorm::load(hidden, eps, output.pp("LayerNorm"))?,
            heads: config.num_attention_heads,
            activation,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor, padding: &Tensor) -> candle_core::Result<Tensor> {
        let context = attention(
            &self.query.forward(xs)?,
            &self.key.forward(xs)?,
            &self.value.forward(xs)?,
            self.heads,
            padding,
        )?;
        let attended = self
            .attention_norm
            .forward(&self.attention_out.forward(&context)?.add(xs)?)?;
        let intermediate = self
            .activation
            .apply(&self.intermediate.forward(&attended)?)?;
        self.output_norm
            .forward(&self.output.forward(&intermediate)?.add(&attended)?)
    }
}

pub(crate) struct Bert {
    pub(crate) words: Embedding,
    pub(crate) positions: Embedding,
    pub(crate) token_types: Embedding,
    pub(crate) norm: LayerNorm,
    pub(crate) layers: Vec<BertLayer>,
}

impl Bert {
    pub(crate) fn load(config: &BertConfig, vb: &VarBuilder) -> Result<Self> {
        let activation = Activation::parse(&config.hidden_act)?;
        let embeddings = vb.pp("embeddings");
        let layers = vb.pp("encoder").pp("layer");
        let hidden = config.hidden_size;
        let build = || -> candle_core::Result<Self> {
            Ok(Self {
                words: candle_nn::embedding(
                    config.vocab_size,
                    hidden,
                    embeddings.pp("word_embeddings"),
                )?,
                positions: candle_nn::embedding(
                    config.max_position_embeddings,
                    hidden,
                    embeddings.pp("position_embeddings"),
                )?,
                token_types: candle_nn::embedding(
                    config.type_vocab_size,
                    hidden,
                    embeddings.pp("token_type_embeddings"),
                )?,
                norm: LayerNorm::load(hidden, config.layer_norm_eps, embeddings.pp("LayerNorm"))?,
                layers: (0..config.num_hidden_layers)
                    .map(|index| BertLayer::load(config, activation, layers.pp(index.to_string())))
                    .collect::<candle_core::Result<Vec<_>>>()?,
            })
        };
        build().map_err(|err| Error(format!("could not load the bert encoder: {err}")))
    }

    pub(crate) fn forward(&self, ids: &Tensor, padding: &Tensor) -> candle_core::Result<Tensor> {
        let (_, tokens) = ids.dims2()?;
        let positions = Tensor::arange(0u32, tokens as u32, ids.device())?;
        // Every session is a single segment, so the token type is always 0.
        let token_types = ids.zeros_like()?;
        let embedded = self
            .words
            .forward(ids)?
            .add(&self.token_types.forward(&token_types)?)?
            .broadcast_add(&self.positions.forward(&positions)?)?;
        let mut hidden = self.norm.forward(&embedded)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, padding)?;
        }
        Ok(hidden)
    }
}

pub(crate) enum Encoder {
    DistilBert(DistilBert),
    Bert(Bert),
}
