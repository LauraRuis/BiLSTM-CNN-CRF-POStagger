import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from tagger.encoder import RecurrentEncoder
from tagger.utils import PAD_TOKEN, ROOT_TOKEN

use_cuda = True if torch.cuda.is_available() else False
logger = logging.getLogger(__name__)


class Tagger(nn.Module):

  def __init__(self, dim=0, emb_dim=0, n_words=0, n_tags=0, n_chars=0, char_emb_dim=0, num_filters=0, window_size=0,
               dropout_p=0.,
               num_layers=1, bi=True,
               form_vocab=None, pos_vocab=None, char_vocab=None,
               char_model=None, **kwargs):

    super(Tagger, self).__init__()

    # get padding indices
    self.form_vocab = form_vocab
    self.form_padding_idx = form_vocab.stoi[PAD_TOKEN]

    self.pos_vocab = pos_vocab
    self.pos_padding_idx = pos_vocab.stoi[PAD_TOKEN]
    self.pos_root_idx = pos_vocab.stoi[ROOT_TOKEN]

    self.char_vocab = char_vocab
    self.char_padding_idx = char_vocab.stoi[PAD_TOKEN]

    self.bi = bi

    self.dim = dim

    self.encoder = RecurrentEncoder(dim=self.dim, emb_dim=emb_dim,
                                    char_emb_dim=char_emb_dim,
                                    n_words=n_words, n_tags=n_tags,  n_chars=n_chars,
                                    dropout_p=dropout_p, num_layers=num_layers,
                                    form_padding_idx=self.form_padding_idx,
                                    pos_padding_idx=self.pos_padding_idx,
                                    char_padding_idx=self.char_padding_idx,
                                    bidirectional=bi,
                                    char_model=char_model,
                                    num_filters=num_filters,
                                    window_size=window_size)

    self.pos_criterion = nn.NLLLoss(size_average=False, reduce=True,
                                    ignore_index=self.pos_padding_idx)

    self.tagger_input_dim = self.dim * 2 if bi else self.dim
    self.linear = nn.Linear(self.tagger_input_dim, n_tags)

  def get_accuracy(self, predictions=None, targets=None):

    pos_acc = 0
    if len(predictions["pos"]) > 0:
      pos_eq = torch.eq(predictions["pos"].data, targets["pos"].data).long()
      pos_mask = (targets["pos"] != self.pos_padding_idx)
      pos_mask = pos_mask.long()
      total_pos = pos_mask.sum().data.item()
      pos_match = (pos_eq * pos_mask).sum().data.item()
      pos_acc = 100. * pos_match / total_pos

    return pos_acc

  @staticmethod
  def classification_loss(scores, targets, criterion):
    n = scores.size(2)
    batch_size = scores.size(0)
    scores_2d = scores.contiguous().view(-1, n)  # TODO: look into contiguous
    loss = criterion(scores_2d, targets.view(-1))
    loss = loss / batch_size
    return loss

  def get_loss(self, scores=None, targets=None):

    xpos_scores = scores
    xpos_targets = targets["pos"]
    loss = self.classification_loss(xpos_scores, xpos_targets, self.pos_criterion)

    return dict(loss=loss)

  def encode_input(self, form_var=None, char_var=None, pos_var=None, lengths=None, word_lengths=None):
    """Encode input sentences."""

    output, _ = self.encoder(form_var=form_var, char_var=char_var, pos_var=pos_var,
                                     lengths=lengths, word_lengths=word_lengths)
    return output, _

  def forward(self, form_var=None, char_var=None, pos_var=None,
              lengths=None, word_lengths=None):
    """Predict arcs and then relations from gold arcs. For training only."""

    # encode sentence
    encoder_output, _ = self.encode_input(form_var=form_var, char_var=char_var, pos_var=pos_var,
                                          lengths=lengths, word_lengths=word_lengths)
    # predict tags
    scores = self.linear(encoder_output)
    scores = F.log_softmax(scores, dim=2)

    return dict(scores=scores)
