import math
import torch

from torch import nn
from torch.nn.init import orthogonal_
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable

from torch.nn.modules.rnn import LSTMCell

use_cuda = True if torch.cuda.is_available() else False


class LSTM(nn.Module):
  """This is a custom LSTM class that should behave as a native PyTorch
  LSTM class.
  It is slower but can use any kind of RNN cell (hopefully).
  """

  def __init__(self, input_size, hidden_size, num_layers, bias=True,
               batch_first=False, dropout=0., bidirectional=False, enlarge_hidden=None):
    super(LSTM, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bias = bias
    self.batch_first = batch_first
    self.dropout = dropout
    self.bidirectional = bidirectional

    if enlarge_hidden:
      assert isinstance(enlarge_hidden, tuple), "provide layer and size in enlarge_hidden"
      enlarge_layer = enlarge_hidden[0]
      enlarge_size = enlarge_hidden[1]
    else:
      enlarge_layer = 0
      enlarge_size = 0

    num_directions = 2 if bidirectional else 1

    assert not batch_first, "currently only supports batch_first=False"

    all_fwd_cells = []
    for i in range(num_layers):
      input_size = hidden_size * num_directions if i > 0 else self.input_size
      if i == enlarge_layer:
          input_size += enlarge_size
      all_fwd_cells.append(VarLSTMCell(input_size, hidden_size, bias=True, dropout=dropout))

    all_bwd_cells = []
    if bidirectional:
      for i in range(num_layers):
        input_size = hidden_size * num_directions if i > 0 else self.input_size
        if i == enlarge_layer:
            input_size += enlarge_size
        all_bwd_cells.append(VarLSTMCell(input_size, hidden_size, bias=True, dropout=dropout))

    # this registers the parameters of the cells so that they are optimized
    self.fwd_cells = nn.ModuleList(all_fwd_cells)
    self.bwd_cells = nn.ModuleList(all_bwd_cells)

    self.dropout_layer = nn.Dropout(p=dropout)

    self.current_mask = None

  def _get_hidden(self, batch_size):
    """Returns empty initial hidden state for each cell."""
    hx = Variable(torch.zeros(batch_size, self.hidden_size))
    cx = Variable(torch.zeros(batch_size, self.hidden_size))
    hx = hx.cuda() if use_cuda else hx
    cx = cx.cuda() if use_cuda else cx
    return hx, cx

  def _reverse(self, tensor):
    _idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
    _idx = Variable(torch.LongTensor(_idx))
    _idx = _idx.cuda() if use_cuda else _idx
    return tensor.index_select(0, _idx)

  def _unroll(self, rnn, inputs, bwd=False):

    # support packed input
    if isinstance(inputs, tuple):
      inputs, lengths = inputs
      batch_size = lengths[0].item()
      max_time = lengths.size(0)
    else:
      raise NotImplementedError('must provide packed sequence')

    # here is a list of all hidden states
    # once a sequence has finished, we do not add any more states to this list
    output = []
    cell_states = []

    hx, cx = self._get_hidden(batch_size)

    mask = rnn.sample_recurrent_dropout_mask(batch_size)

    idx = inputs.size(0) if bwd else 0
    start = max_time - 1 if bwd else 0
    stop = -1 if bwd else max_time
    step = -1 if bwd else 1

    # when going backward, we start with only the longest sequences
    if bwd:
      _hx = hx[:lengths[-1]]
      _cx = cx[:lengths[-1]]
    else:
      _hx, _cx = hx, cx

    # fwd: 0 .. max_time-1
    # bwd: max_time-1 ... 0
    for i in range(start, stop, step):

      # this is the number of sequences not yet complete
      # it is initially equal to the batch size, but will diminish over time
      num_active = lengths[i].item()

      _hx = _hx[:num_active]
      _cx = _cx[:num_active]

      if num_active > _hx.size(0):
        difference = num_active - _hx.size(0)
        extra_hx, extra_cx = self._get_hidden(difference)
        _hx = torch.cat((_hx, extra_hx), dim=0)
        _cx = torch.cat((_cx, extra_cx), dim=0)

      # get the inputs for this time step
      if bwd:
        rnn_input = inputs[idx - num_active:idx]  # cells are batch-first
      else:
        rnn_input = inputs[idx:idx + num_active]  # cells are batch-first

      # perform an rnn step
      _hx, _cx = rnn(rnn_input, (_hx, _cx), mask=mask[:num_active])

      output.append(_hx)
      cell_states.append(_cx)

      if bwd:
        idx = idx - num_active
      else:
        idx = idx + num_active

    if bwd:
      output = output[::-1]

    output = torch.cat(output, 0)
    cell_states = torch.cat(cell_states, 0)

    all_states = (output, cell_states)

    return PackedSequence(output, lengths), all_states

  def forward(self, inputs, *args, **kwargs):
    """This is where we run multiple layers of (bi-)LSTMs"""

    all_hiddens = []
    all_cells = []

    for i in range(self.num_layers):
      fwd_rnn = self.fwd_cells[i]
      (out, lengths), all_states = self._unroll(fwd_rnn, inputs)

      if self.bidirectional:
        bwd_rnn = self.bwd_cells[i]
        (bwd_out, _), bwd_states = self._unroll(bwd_rnn, inputs, bwd=True)

        out = torch.cat((out, bwd_out), dim=1)
        all_hiddens.append(PackedSequence(torch.cat((all_states[0], bwd_states[0]), dim=1), lengths))
        all_cells.append(PackedSequence(torch.cat((all_states[1], bwd_states[1]), dim=1), lengths))
      else:
        all_hiddens.append(PackedSequence(all_states[0], lengths))
        all_cells.append(PackedSequence(all_states[1], lengths))

      inputs = (self.dropout_layer(out), lengths)  # dropout in-between layers

    return PackedSequence(out, lengths), (all_hiddens, all_cells)

  def layer_func(self, inputs, layer, *args, **kwargs):
    """This is where we run multiple layers of (bi-)LSTMs"""

    # for i in range(self.num_layers):
    fwd_rnn = self.fwd_cells[layer]
    (out, lengths), all_states = self._unroll(fwd_rnn, inputs)

    if self.bidirectional:
      bwd_rnn = self.bwd_cells[layer]
      (bwd_out, _), bwd_states = self._unroll(bwd_rnn, inputs, bwd=True)

      out = torch.cat((out, bwd_out), dim=1)
      hidden = PackedSequence(torch.cat((all_states[0], bwd_states[0]), dim=1), lengths)
      cell = PackedSequence(torch.cat((all_states[1], bwd_states[1]), dim=1), lengths)
    else:
      hidden = PackedSequence(all_states[0], lengths)
      cell = PackedSequence(all_states[1], lengths)

    if layer < self.num_layers:
        out = self.dropout_layer(out)  # dropout in-between layers

    return PackedSequence(out, lengths), (hidden, cell)

  def step_func(self, input, states, sample_mask=True):

    assert self.num_layers == 1, "step function only implemented for single layer for now"
    assert not self.bidirectional, "step function only implemented in one direction for now"
    assert isinstance(states, tuple), "please provide hidden state and memory state in tuple"

    # support packed input
    if isinstance(input, tuple):
      rnn_input, lengths = input
      batch_size = lengths[0].item()
    else:
      raise NotImplementedError('must provide packed sequence')

    rnn = self.fwd_cells[0]

    # only sample new mask once per sequence
    if sample_mask:
      self.current_mask = rnn.sample_recurrent_dropout_mask(batch_size)

    _hx, _cx = rnn(rnn_input, states, mask=self.current_mask)

    return _hx, _cx


class VarLSTMCell(LSTMCell):

  """
  LSTM cell with Variational/recurrent dropout.
  This is slower than the PyTorch cell since we cannot use
  the CUDA backend for it.
  """

  def __init__(self, input_size, hidden_size, bias=True, dropout=0.):
    super(VarLSTMCell, self).__init__(input_size, hidden_size, bias=bias)
    self.dropout = dropout
    self.xavier_uniform()

  def reset_parameters(self, gain=1., ortho_init=True, forget_bias=0.):

    # default pytorch initialization
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

    # orthogonal init
    if ortho_init:
      for hh in (self.weight_ih, self.weight_hh):
        for i in range(0, hh.size(0), self.hidden_size):
          orthogonal_(hh[i:i + self.hidden_size], gain=gain)

    # set forget gate bias
    if self.bias_ih is not None:
      self.bias_ih.data.fill_(forget_bias)

    if self.bias_hh is not None:
      self.bias_hh.data.fill_(forget_bias)

  def xavier_uniform(self, gain=1., forget_bias=1.):

    # default pytorch initialization
    for weight in self.parameters():
      if len(weight.size()) > 1:
          nn.init.xavier_uniform_(weight.data, gain=gain)

    # set forget gate bias
    if self.bias_ih is not None:
      self.bias_ih.data.fill_(forget_bias)

    if self.bias_hh is not None:
      self.bias_hh.data.fill_(forget_bias)

  def sample_recurrent_dropout_mask(self, batch_size=1):
    """Returns a dropout mask (FloatTensor) of ones and zeros."""
    mask = Variable(torch.bernoulli(
      torch.Tensor(batch_size, self.hidden_size).fill_(1-self.dropout)))
    return mask.cuda() if use_cuda else mask

  def forward(self, input, hx, mask=None):
    """
    input is (batch, input_size)
    hx is ((batch, hidden_size), (batch, hidden_size))
    """
    prev_h, prev_c = hx

    all_ih = torch.matmul(self.weight_ih, torch.transpose(input, 0, 1))
    all_hh = torch.matmul(self.weight_hh, torch.transpose(prev_h, 0, 1))

    if self.bias_ih is not None:
      all_ih = all_ih.add(self.bias_ih.unsqueeze(1))

    if self.bias_hh is not None:
      all_hh = all_hh.add(self.bias_hh.unsqueeze(1))

    ii, fi, gi, oi = torch.chunk(all_ih, 4, dim=0)
    ih, fh, gh, oh = torch.chunk(all_hh, 4, dim=0)

    i = torch.sigmoid(ii + ih)
    f = torch.sigmoid(fi + fh).transpose(0, 1)
    g = torch.tanh(gi + gh)
    o = torch.sigmoid(oi + oh).transpose(0, 1)

    c = f * prev_c + torch.transpose(i * g, 0, 1)
    h = o * torch.tanh(c)

    if self.training and self.dropout > 0.:
      h = h.mul(mask).div(1.0 - self.dropout)

    return h, c
