## The tfidf-logreg artifact: `model.json`

A tfidf-logreg model is one JSON document, `model.json`, beside its
`metrics.json` in `<training root>/models/<aspect>/`. It holds everything
inference needs and nothing it has to recompute.

| Field | What it holds |
|---|---|
| `backend` | `tfidf-logreg` |
| `format` | The layout of this document. This binary writes and reads layout `1`, the one described here; `info`, `evaluate` and inference refuse a document of any other layout, naming the layout they found. |
| `vectorizer.analyzer` | `word` |
| `vectorizer.lowercase` | Whether text was lowercased before counting |
| `vectorizer.token_pattern` | The token pattern words were cut with |
| `vectorizer.ngram_range` | The shortest and longest n-gram counted, in words; the shortest is always a single word |
| `vectorizer.sublinear_tf` | Whether term counts were dampened logarithmically |
| `vectorizer.smooth_idf`, `min_df`, `max_df`, `norm` | The settings the vocabulary and idf weights were fitted under |
| `vectorizer.vocabulary`, `vectorizer.idf` | The terms, in column order, and one idf weight per term |
| `classifier.kind`, `classifier.multi_class` | `logistic-regression`, `multinomial` |
| `classifier.c`, `max_iter`, `tol` | The regularisation and stopping settings it was fitted under |
| `classifier.iterations`, `classifier.converged` | How many iterations the fit ran and whether it converged |
| `classifier.classes`, `intercept`, `coef` | The class labels, one intercept per class, one weight row per class |

A document whose vocabulary and idf lengths differ, or whose class list,
intercepts and weight rows differ in length, is refused with both lengths.

`metrics.json` records `cv_accuracy`, the mean accuracy over stratified folds
of the training side, rounded as Python's `round(x, 4)` rounds. When no fold
can be scored (every fold's training side holds a single class, or no fold has
a test row), `cv_accuracy` is `null`: the model was not cross-validated, which
is not the same as an accuracy of zero.
