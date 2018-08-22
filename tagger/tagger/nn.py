import logging
import torch
import math
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from tagger.lstm import LSTM
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


class MLP(nn.Module):
  """Simple MLP class."""

  def __init__(self, in_dim=0, dim=0, out_dim=0, depth=1,
               activation=F.leaky_relu, dropout_p=0.):
    super(MLP, self).__init__()

    self.depth = depth

    self.inner = nn.Linear(in_dim, dim)
    if depth > 1:
      self.outer = nn.Linear(dim, out_dim)

    self.dropout = nn.Dropout(p=dropout_p)

    self.activation = activation

  def forward(self, x):
    if self.depth > 1:
      return self.outer(self.activation(self.inner(self.dropout(x))))
    else:
      return self.activation(self.inner(self.dropout(x)))


class CharModel(nn.Module):

  def __init__(self, n_chars, padding_idx, emb_dim=30, hidden_size=50, output_dim=50, dropout_p=0.5,
               bi=False):
    super(CharModel, self).__init__()

    self.input_dim = n_chars
    self.output_dim = output_dim
    self.dropout_p = dropout_p
    self.padding_idx = padding_idx
    self.hidden_size = hidden_size
    self.emb_dim = emb_dim

    self.embedding = nn.Embedding(n_chars, emb_dim, padding_idx=padding_idx)
    self.init_embedding()
    self.char_emb_dropout = nn.Dropout(p=dropout_p)

    self.size = hidden_size * 2 if bi else hidden_size

  def init_embedding(self):
    init_range = math.sqrt(3 / self.emb_dim)
    embed = self.embedding.weight.clone()
    embed.uniform_(-init_range, init_range)
    self.embedding.weight.data.copy_(embed)

  def forward(self, char_input=None, lengths=None):

    batch_size, seq_len, time = char_input.size()

    # make bsz * seq_len new batch size (only sequential part is word lengths)
    char_input = char_input.view(batch_size * seq_len, time)  # (bsz * T, time)

    # sort word lengths and input accordingly
    lengths = lengths.view(batch_size * seq_len)
    sorted_lengths, sort_idx = torch.sort(lengths, dim=0, descending=True)
    sort_idx = sort_idx.cuda() if torch.cuda.is_available() else sort_idx
    char_input = char_input.index_select(0, sort_idx)

    # get only non padding sequences (namely remove all sequences with length 0 coming from sentence padding)
    non_padding_idx = (sorted_lengths != 0).long().sum()
    char_input_no_pad = char_input[:non_padding_idx]
    sorted_lengths_no_pad = sorted_lengths[:non_padding_idx]

    # embed chars
    embedded = self.embedding(char_input_no_pad)
    embedded = self.char_emb_dropout(embedded)

    # character model
    output = self.char_model(embedded, char_input_no_pad, sorted_lengths_no_pad)

    # put padding back
    padding_length = sorted_lengths.size(0) - non_padding_idx
    dim = output.size(1)
    zeros_tensor = torch.zeros((padding_length, dim))
    zeros_tensor = zeros_tensor.cuda() if torch.cuda.is_available() else zeros_tensor
    output = torch.cat((output, zeros_tensor), dim=0)

    # put back in right order
    odx = torch.unsqueeze(sort_idx, 1).expand(sort_idx.size(0), dim)
    empty_out = torch.zeros(batch_size * seq_len, dim)
    empty_out = empty_out.cuda() if torch.cuda.is_available() else empty_out
    output = empty_out.scatter_(0, odx, output)
    output = output.view(batch_size, seq_len, dim)

    return output, lengths

  def char_model(self, embedded=None, char_input_no_pad=None, lengths=None):
    raise NotImplementedError


class DozatCharModel(CharModel):

  def __init__(self, n_chars, padding_idx, emb_dim=100, hidden_size=400, output_dim=100, dropout_p=0.5,
               bi=False):
    super(DozatCharModel, self).__init__(n_chars, padding_idx, emb_dim=emb_dim, hidden_size=hidden_size,
                                         output_dim=output_dim, dropout_p=0.5, bi=False)

    self.size = hidden_size * 2 if bi else hidden_size

    self.attention_weights = nn.Parameter(data=torch.Tensor(1, self.size), requires_grad=True)
    self.init_parameter()
    self.linear = nn.Linear(self.size * 2, output_dim, bias=False)

    # this is our own LSTM that supports variational dropout
    self.char_rnn = LSTM(emb_dim, hidden_size, 1, bias=True,
                         batch_first=False, dropout=dropout_p, bidirectional=bi)

  def init_parameter(self):

    # copied from nn.Linear()
    stdv = 1. / math.sqrt(self.attention_weights.size(1))
    self.attention_weights.data.uniform_(-stdv, stdv)

  def char_model(self, embedded=None, char_input_no_pad=None, lengths=None):

    embedded = torch.transpose(embedded, 0, 1)  # (time, bsz, dim)
    embedded = pack_padded_sequence(embedded, lengths)

    # run lstm
    output, (all_hid, all_cell) = self.char_rnn(embedded)

    # get hidden states
    output, output_lengths = pad_packed_sequence(output)
    output = torch.transpose(output, 1, 0)  # (bsz, time, dim)

    # get final layer cell states
    cell_state, cell_lengths = pad_packed_sequence(all_cell[-1])
    cell_state = torch.transpose(cell_state, 1, 0)  # (bsz, time, dim)

    # attention # TODO: add dropout on attention connections (Dozat)
    attention_scores = torch.bmm(output, torch.unsqueeze(self.attention_weights.repeat(output.size(0), 1), dim=2))
    mask = (char_input_no_pad == self.padding_idx)
    attention_scores.data.masked_fill_(torch.unsqueeze(mask, dim=2), -float('inf'))
    attention = F.softmax(attention_scores, dim=1)

    h_hat = torch.bmm(torch.transpose(output, 2, 1), attention)
    # TODO: make memory efficient
    dim = cell_state.size(2)
    indices = lengths.view(-1, 1).unsqueeze(2).repeat(1, 1, dim) - 1
    indices = indices.cuda()if torch.cuda.is_available() else indices
    final_cell_states = torch.squeeze(torch.gather(cell_state, 1, indices), dim=1)
    v_hat = self.linear(torch.squeeze(torch.cat((h_hat, torch.unsqueeze(final_cell_states, dim=2)), dim=1)))

    return v_hat


class SimpleCharModel(CharModel):

  def __init__(self, n_chars, padding_idx, emb_dim=100, hidden_size=400, output_dim=100, dropout_p=0.33, bi=True):
    super(SimpleCharModel, self).__init__(n_chars, padding_idx, emb_dim=emb_dim, hidden_size=hidden_size,
                                          output_dim=output_dim, dropout_p=dropout_p, bi=bi)

    self.size = hidden_size * 2 if bi else hidden_size
    self.linear = nn.Linear(self.size, output_dim, bias=False)

    # this is our own LSTM that supports variational dropout
    self.char_rnn = LSTM(emb_dim, hidden_size, 1, bias=True,
                         batch_first=False, dropout=dropout_p, bidirectional=bi)

  def char_model(self, embedded=None, char_input_no_pad=None, lengths=None):

    embedded = torch.transpose(embedded, 0, 1)  # (time, bsz, dim)
    embedded = pack_padded_sequence(embedded, lengths)

    # run lstm
    output, (all_hid, all_cell) = self.char_rnn(embedded)

    # get final layer cell states
    cell_state, cell_lengths = pad_packed_sequence(all_cell[-1])
    cell_state = torch.transpose(cell_state, 1, 0)  # (bsz, time, dim)
    dim = cell_state.size(2)
    indices = lengths.view(-1, 1).unsqueeze(2).repeat(1, 1, dim) - 1
    indices = indices.cuda() if torch.cuda.is_available() else indices
    final_cell_states = torch.squeeze(torch.gather(cell_state, 1, indices), dim=1)

    # project it to the right dimension
    output = self.linear(torch.squeeze(final_cell_states))

    return output


class CharCNN(CharModel):

  def __init__(self, n_chars, padding_idx, emb_dim, num_filters, window_size, dropout_p):

    super(CharCNN, self).__init__(n_chars, padding_idx, emb_dim=emb_dim, hidden_size=400, output_dim=100,
                                  dropout_p=0.33, bi=False)

    self.conv = nn.Conv1d(100, num_filters, window_size, padding=window_size - 1)

  def char_model(self, embedded=None, char_input_no_pad=None, lengths=None):

    embedded = torch.transpose(embedded, 1, 2)  # (bsz, dim, time)
    chars = self.conv(embedded)
    chars = F.max_pool1d(chars, kernel_size=chars.size(2)).squeeze(2)

    return chars


class RnnTagger(nn.Module):

  def __init__(self, input_dim=512, n_tags=1, tag_emb_dim=28, dim=128, num_layers=1,
               dropout_p=0.33, bi=False, tag_padding_idx=None, tag_root_idx=None):
    super(RnnTagger, self).__init__()

    assert num_layers == 1, "RnnTagger only implemented for single layer"
    assert not bi, "RnnTagger only implemented in one direction"

    self.tag_dim = tag_emb_dim
    # self.input_dim = input_dim + tag_emb_dim
    self.input_dim = input_dim
    self.dropout_p = dropout_p
    self.n_tags = n_tags

    self.hidden_size = dim
    self.bi = bi

    self.root_index = tag_root_idx
    # self.tag_embedding = nn.Embedding(n_tags, tag_emb_dim, padding_idx=tag_padding_idx)
    # self.tag_dropout = nn.Dropout(p=dropout_p)

    self.rnn = nn.LSTM(self.input_dim, self.hidden_size, num_layers, bias=True,
                       batch_first=False, dropout=dropout_p, bidirectional=bi)
    # self.rnn = LSTM(self.input_dim, self.hidden_size, num_layers, bias=True,
    #                 batch_first=False, dropout=dropout_p, bidirectional=bi)

    self.mlp_input = self.hidden_size * 2 if bi else self.hidden_size
    # self.mlp = MLP(self.mlp_input, dim, dim, dropout_p=dropout_p, depth=1)
    self.linear = nn.Linear(self.mlp_input, self.n_tags)

  def _get_hidden(self, batch_size):
    """Returns empty initial hidden state for each cell."""
    first_dim = 1 if not self.bi else 2
    hx = Variable(torch.zeros(first_dim, batch_size, self.hidden_size))
    cx = Variable(torch.zeros(first_dim, batch_size, self.hidden_size))
    hx = hx.cuda() if use_cuda else hx
    cx = cx.cuda() if use_cuda else cx
    return hx, cx

  def forward(self, input=None, tag_input=None, lengths=None, training=False):

    # LSTM  # TODO!
    input = torch.transpose(input, 0, 1)  # time, batch, dim
    input = pack_padded_sequence(input, lengths)
    rnn_input, packed_lengths = input

    # if training:
    #   # move all tags one dimension up ensuring input to rnn is previous tag
    #   #  TODO: this is teacher forcing, also implement predicted and mix
    #   zeros_tensor = torch.zeros((tag_input.size(0), 1)).long()
    #   zeros_tensor = zeros_tensor.cuda() if torch.cuda.is_available() else zeros_tensor
    #   tag_input = torch.cat((zeros_tensor, tag_input[:, :-1]), dim=1)
    #   tag_embed = self.tag_embedding(tag_input)
    #   tag_embed = self.tag_dropout(tag_embed)
    #   tag_embed = torch.transpose(tag_embed, 0, 1)  # (time, batch, dim)
    #   packed_tags, packed_lengths = pack_padded_sequence(tag_embed, lengths)
    #
    #   rnn_input = torch.cat((rnn_input, packed_tags), dim=1)

    # teacher forcing
    hx, cx = self._get_hidden(packed_lengths[0].item())
    output, _ = self.rnn(PackedSequence(rnn_input, packed_lengths), (hx, cx))

    # MLP
    scores = self.predict_tag(output)
    # else:
    #
    #   # unroll rnn to feed predicted tags to next time step (at test time)
    #   first_tag = torch.Tensor([0] * packed_lengths[0].item()).long()
    #   tag_input = first_tag.cuda() if torch.cuda.is_available() else first_tag
    #   hx, cx = self._get_hidden(packed_lengths[0].item())
    #
    #   start = 0
    #   scores = []
    #   sample_vardrop_mask = True
    #
    #   for t, n_examples in enumerate(packed_lengths):
    #
    #     # prepare inputs
    #     tag_input = tag_input[:n_examples]
    #     hx, cx = hx[:, :n_examples, :], cx[:, :n_examples, :]
    #     tag_embed = self.tag_embedding(tag_input)
    #     real_input = rnn_input[start:start+n_examples]
    #     real_lengths = torch.Tensor([n_examples]).long()
    #     real_lengths = real_lengths.cuda() if torch.cuda.is_available() else real_lengths
    #     real_input = PackedSequence(torch.cat((real_input, tag_embed), 1), real_lengths)
    #
    #     # one time step
    #     hx = torch.squeeze(hx, 0)
    #     cx = torch.squeeze(cx, 0)
    #     hx, cx = self.rnn.step_func(real_input, (hx, cx), sample_vardrop_mask)
    #
    #     # set next inputs
    #     real_input = PackedSequence(hx, real_lengths)
    #     hx = torch.unsqueeze(hx, 0)
    #     cx = torch.unsqueeze(cx, 0)
    #
    #     # predict tag
    #     current_scores = self.predict_tag(real_input)
    #     scores.append(current_scores)
    #     tag_input = current_scores.max(dim=2)[1].squeeze(1)
    #
    #     start = start + n_examples
    #     sample_vardrop_mask = False
    #
    #   scores = torch.cat(scores, dim=0).squeeze(1)
    #   scores = PackedSequence(scores, packed_lengths)
    #   scores, _ = pad_packed_sequence(scores)
    #   scores = scores.transpose(0, 1)

    return scores

  def predict_tag(self, input):

    # MLP
    # mlp_out = self.mlp(input[0])
    # output = PackedSequence(input)
    output, _ = pad_packed_sequence(input)
    output = output.transpose(0, 1)

    # Linear transform
    scores = self.linear(output)
    scores = F.log_softmax(scores, dim=2)

    return scores


class Tagger(nn.Module):

  def __init__(self, input_dim=512, dim=128, num_layers=1, output_dim=None, dropout_p=0.33):
    super(Tagger, self).__init__()

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.dropout_p = dropout_p

    self.mlp = MLP(input_dim, dim, dim, dropout_p=dropout_p, depth=num_layers)
    self.linear = MLP(dim, output_dim)

  def init_parameter(self):
    # copied from nn.Linear()
    stdv = 1. / math.sqrt(self.attention_weights.size(1))
    self.attention_weights.data.uniform_(-stdv, stdv)

  def forward(self, input=None, tags=None, lengths=None, training=False):

    # MLP
    input, lengths = input
    output = self.mlp(input)
    output = PackedSequence(output, lengths)
    output, _ = pad_packed_sequence(output)
    output = output.transpose(0, 1)

    # Linear transform
    scores = self.linear(output)
    scores = F.log_softmax(scores, dim=2)

    return scores
