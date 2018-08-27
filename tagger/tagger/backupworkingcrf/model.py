import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from tagger.encoder import RecurrentEncoder
from tagger.utils import PAD_TOKEN, ROOT_TOKEN, END_TOKEN
from tagger.nn import ChainCRF, ChainCRFallenNLP
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

    self.tagger_input_dim = self.dim * 2 if bi else self.dim
    self.tagger = tagger

    if tagger == "crf":
      self.tag_model_1 = ChainCRFallenNLP(n_tags, self.pos_vocab.stoi, self.tagger_input_dim)
      self.hid2tag = nn.Linear(self.tagger_input_dim, n_tags)

      # self.tag_model = ConditionalRandomField(n_tags, include_start_end_transitions=True)
      self.xavier_uniform()
    elif tagger == "linear":
      self.tag_model = nn.Linear(self.tagger_input_dim, n_tags)
      self.xavier_uniform()

  def xavier_uniform(self, gain=1.):

    # default pytorch initialization
    for name, weight in self.hid2tag.named_parameters():  # TODO: change back to self.
      if len(weight.size()) > 1:
          nn.init.xavier_uniform_(weight.data, gain=gain)
      elif "bias" in name:
        weight.data.fill_(0.)
    # self.tag_model.start_transitions.data = self.tag_model_1.log_transitions[self.pos_vocab.stoi[ROOT_TOKEN], :].data
    # self.tag_model.transitions.data = self.tag_model_1.log_transitions
    # self.tag_model.end_transitions.data = self.tag_model_1.log_transitions[:, self.pos_vocab.stoi[END_TOKEN]].data
    return

  def get_accuracy(self, predictions=None, targets=None):

    pos_acc = 0
    if len(predictions["pos"]) > 0:
      pos_eq = torch.eq(predictions["pos"].data, targets["pos"].data).long()  # TODO: change back
      pos_mask = (targets["pos"].data != self.pos_padding_idx)
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

    xpos_scores = scores["output"]
    xpos_targets = targets["pos"]

    if self.tagger == "linear":
      loss = self.classification_loss(xpos_scores, xpos_targets, self.pos_criterion)
    elif self.tagger == "crf":
      loss = scores["loss"]
    else:
      loss = -1.

    return dict(loss=loss)

  def pos_tagger(self, input=None, pos_var=None, lengths=None):

    if self.tagger == "linear":
      scores = self.tag_model(input)
      tagger_output = dict(loss=None, output=scores, score=None, sequence=None)
    # elif self.tagger == "crf":
    #   pos_loss, pos_score, pos_seq = self.tag_model(input, pos_var[:, 1:], lengths, self.training)
    #   tagger_output = dict(loss=pos_loss, output=None, score=pos_score, sequence=pos_seq)
    elif self.tagger == "crf":

      mask = ((pos_var != self.pos_vocab.stoi[PAD_TOKEN]) - (pos_var == self.pos_vocab.stoi[END_TOKEN])).long()
      mask = mask[:, 1:-1]
      mask = mask.cuda() if torch.cuda.is_available() else mask
      input = self.hid2tag(input)
      pad_token = torch.LongTensor([self.pos_vocab.stoi[PAD_TOKEN]])
      pad_token = pad_token.cuda() if torch.cuda.is_available() else pad_token
      lengths = torch.LongTensor(lengths)
      lengths = lengths.cuda() if torch.cuda.is_available() else lengths
      indices = torch.LongTensor([i for i in range(pos_var.size(0))])
      indices = indices.cuda() if torch.cuda.is_available() else indices
      tags = pos_var.index_put_((indices, lengths + 1),
                                            pad_token)

      pos_loss, pos_score, pos_seq = self.tag_model_1(input, tags[:, 1:-1], mask, lengths, self.training)
      tagger_output = dict(loss=pos_loss, output=None, score=pos_score, sequence=pos_seq)

      # if self.training:
      #   pos_loss, pos_score, pos_seq = self.tag_model_1(input, tags[:, 1:-1], mask, lengths, self.training)
      #   tagger_output = dict(loss=pos_loss, output=None, score=pos_score, sequence=pos_seq)
      #   loglikeli = self.tag_model(input, tags[:, 1:-1], mask)
      # else:
      #   loglikeli = 0
      # with torch.no_grad():
      #   best_paths = self.tag_model.viterbi_tags(input, mask)
      #   max_time = input.size(1)
      #   test = [torch.LongTensor(x + [self.pos_vocab.stoi[PAD_TOKEN]] * (max_time - len(x))).unsqueeze(0) for x, y in
      #           best_paths]
      #   best_paths = torch.cat(test)
      #   best_paths = best_paths.cuda() if torch.cuda.is_available() else best_paths
      #   end_token_tensor = torch.zeros(input.size(0)).unsqueeze(1).long()
      #   end_token_tensor = end_token_tensor.cuda() if torch.cuda.is_available() else end_token_tensor
      #   best_paths = torch.cat((best_paths, end_token_tensor), dim=1)
      #   end_token = torch.LongTensor([self.pos_vocab.stoi[END_TOKEN]])[0]
      #   end_token = end_token.cuda() if torch.cuda.is_available() else end_token
      #   pos_seq = best_paths.index_put_((torch.LongTensor([i for i in range(best_paths.size(0))]), lengths),
      #                                      end_token)

      tagger_output = dict(loss=pos_loss, output=None, score=pos_score, sequence=pos_seq)
    else:
      tagger_output = None

    return tagger_output

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
    output = self.pos_tagger(encoder_output, pos_var, lengths)

    return output
