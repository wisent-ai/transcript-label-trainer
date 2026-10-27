# Models

What the trainer labels and which models it trains, qualifies and publishes,
one page each:

- [Automatic labeling with a Brama teacher](models/autolabel.md): `autolabel`,
  its review gate, and how a command reaches Brama and its credentials.
- [Reviewed Jeden goal model](models/goal.md): `goal-model`, the 3–7 word task
  goals Jeden displays, evaluated with the sampling Jeden Desktop runs it with, and
  `release-publish --model goal`.
- [Reviewed Oko goal-lifecycle model](models/lifecycle.md): the
  `oko-goal-lifecycle-v2` contract, its reviewed splits and curricula,
  `lifecycle-model`, its evaluation with the sampling Oko asks the model with, and
  `release-publish --model lifecycle`.
- [Private personal-voice publication](models/humanizer.md): the humanizer
  adapter, its evaluation with llama-server's own sampling as Echo asks it,
  `humanizer-publish` and the real publication journeys.
