# BiLSTM-CNN-CRF-POStagger
BiLSTM CNN CRF POS tagger in PyTorch

## Possible models

BiLSTM encoder, CNN character model, CRF tagger from:
- Ma & Hovy (2016) [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354)

BiLSTM encoder, LSTM with attention character model, Affine tagger from:
- Dozat et al. (2017) [Stanford's Graph-based Neural Dependency Parser at the CoNLL 2017 Shared Task](https://web.stanford.edu/~tdozat/files/TDozat-CoNLL2017-Paper.pdf)

Code for training heavily based on: https://github.com/bastings/parser
