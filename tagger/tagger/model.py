import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from tagger.encoder import RecurrentEncoder
from tagger.utils import PAD_TOKEN, ROOT_TOKEN, END_TOKEN
from tagger.nn import ChainCRF, MLPTagger
from AllenNLPCRF import ConditionalRandomField

use_cuda = True if torch.cuda.is_available() else False
logger = logging.getLogger(__name__)


class Tagger(nn.Module):

  def __init__(self, dim=0, emb_dim=0, n_words=0, n_tags=0, n_chars=0, char_emb_dim=0, num_filters=0, window_size=0,
               dropout_p=0.,
               num_layers=1, bi=True,
               form_vocab=None, pos_vocab=None, char_vocab=None,
               char_model=None, tagger=None, **kwargs):

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

    self.pos_criterion = nn.CrossEntropyLoss(size_average=False, reduce=True,
                                             ignore_index=self.pos_padding_idx)
    # self.pos_criterion = nn.NLLLoss(size_average=False, reduce=True,
    #                                           ignore_index=self.pos_padding_idx)

    self.tagger_input_dim = self.dim * 2 if bi else self.dim
    self.tagger = tagger

    if tagger == "crf":
      self.tag_model = ChainCRF(n_tags, self.pos_vocab.stoi)
      # self.tag_model = ConditionalRandomField(n_tags, include_start_end_transitions=True)
      self.hid2tag = nn.Linear(self.tagger_input_dim, n_tags)
      self.xavier_uniform("hid2tag")
    elif tagger == "linear":
      self.tag_model = nn.Linear(self.tagger_input_dim, n_tags)
      self.xavier_uniform("tag_model")
    elif tagger == "mlp":
      self.tag_model = MLPTagger(self.tagger_input_dim, dim=128, num_layers=1, output_dim=n_tags, dropout_p=dropout_p)

  def xavier_uniform(self, name, gain=1.):

    # default pytorch initialization
    for name, weight in getattr(self, name).named_parameters():
      if len(weight.size()) > 1:
          nn.init.xavier_uniform_(weight.data, gain=gain)
      elif "bias" in name:
        weight.data.fill_(0.)

  def get_accuracy(self, predictions=None, targets=None):

    pos_acc = 0
    if len(predictions["pos"]) > 0:
      pos_eq = torch.eq(predictions["pos"].data, targets["pos"][0].data[:, 1:-1]).long()  # TODO: change back
      pos_mask = (targets["pos"][0].data[:, 1:-1] != self.pos_padding_idx)
      pos_mask = pos_mask.long()
      total_pos = pos_mask.sum().data.item()
      pos_match = (pos_eq * pos_mask).sum().data.item()
      pos_acc = 100. * pos_match / total_pos

    return pos_acc

  @staticmethod
  def classification_loss(scores, targets, criterion):
    n = scores.size(2)
    batch_size = scores.size(0)
    scores_2d = scores.contiguous().view(-1, n)
    loss = criterion(scores_2d, targets.contiguous().view(-1))
    loss = loss / batch_size
    return loss

  def get_loss(self, scores=None, targets=None):

    xpos_scores = scores["output"]
    xpos_targets = targets["pos"][0]

    if self.tagger == "linear" or self.tagger == "mlp":
      loss = self.classification_loss(xpos_scores, xpos_targets[:, 1:-1], self.pos_criterion)
    elif self.tagger == "crf":
      loss = scores["loss"]
    else:
      loss = -1.

    return dict(loss=loss)

  def remove_aux_tokens(self, sequence, lengths):

    # remove root and end token from pos sequences
    pad_token = torch.LongTensor([self.pos_vocab.stoi[PAD_TOKEN]])
    pad_token = pad_token.cuda() if torch.cuda.is_available() else pad_token
    indices = torch.LongTensor([i for i in range(sequence.size(0))])
    indices = indices.cuda() if torch.cuda.is_available() else indices
    root_indices = torch.LongTensor([0 for _ in range(sequence.size(0))])
    root_indices = root_indices.cuda() if torch.cuda.is_available() else root_indices
    sequence.index_put_((indices, lengths - 1),
                         pad_token)
    sequence.index_put_((indices, root_indices),
                         pad_token)
    return sequence[:, 1:-1]

  def pos_tagger(self, input=None, pos_var=None, lengths=None, pos_lengths=None):

    pos_var = self.remove_aux_tokens(pos_var, pos_lengths)

    if self.tagger == "linear":
      scores = self.tag_model(input)
      # scores = F.log_softmax(scores, dim=2)
      tagger_output = dict(loss=None, output=scores, score=None, sequence=None)

    elif self.tagger == "crf":

      # prepare padding mask
      mask = (pos_var != self.pos_vocab.stoi[PAD_TOKEN]).long()

      # get rid of root and end token
      mask = mask.cuda() if torch.cuda.is_available() else mask

      # prepare input for crf
      input = self.hid2tag(input)

      # run crf
      if isinstance(lengths, list):
        lengths = torch.LongTensor(lengths)
        lengths = lengths.cuda() if torch.cuda.is_available() else lengths
      pos_loss, pos_score, pos_seq = self.tag_model(input, pos_var, mask, lengths, self.training)

      tagger_output = dict(loss=pos_loss, output=None, score=pos_score, sequence=pos_seq)
    elif self.tagger == "mlp":

      scores = self.tag_model(input)
      tagger_output = dict(loss=None, output=scores, score=None, sequence=None)

    else:
      tagger_output = None

    return tagger_output

  def encode_input(self, form_var=None, char_var=None, pos_var=None, lengths=None, word_lengths=None):
    """Encode input sentences."""

    output, _ = self.encoder(form_var=form_var, char_var=char_var, pos_var=pos_var,
                             lengths=lengths, word_lengths=word_lengths)
    return output, _

  def forward(self, form_var=None, char_var=None, pos_var=None,
              lengths=None, word_lengths=None, pos_lengths=None):
    """Predict arcs and then relations from gold arcs. For training only."""

    # encode sentence
    encoder_output, _ = self.encode_input(form_var=form_var, char_var=char_var, pos_var=pos_var,
                                          lengths=lengths, word_lengths=word_lengths)
    # predict tags
    output = self.pos_tagger(encoder_output, pos_var, lengths, pos_lengths)

    return output
