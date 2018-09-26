import logging
import torch
import torch.nn as nn
from tagger.lstm import LSTM
from torch.autograd import Variable
from tagger.nn import DozatCharModel, SimpleCharModel, CharCNN

use_cuda = True if torch.cuda.is_available() else False
logger = logging.getLogger(__name__)


class RecurrentEncoder(nn.Module):
  def __init__(self, dim=0, emb_dim=0, char_emb_dim=100, n_words=0, n_tags=0, n_chars=0,
               dropout_p=0.33, bidirectional=True, num_layers=1,
               num_filters=0, window_size=0,
               form_padding_idx=1, pos_padding_idx=0, char_padding_idx=0,
               char_model=None, var_drop=False):
    super(RecurrentEncoder, self).__init__()

    self.dim = dim
    self.emb_dim = emb_dim
    self.n_words = n_words
    self.n_tags = n_tags
    self.bidirectional = bidirectional

    self.form_padding_idx = form_padding_idx
    self.pos_padding_idx = pos_padding_idx
    self.char_padding_idx = char_padding_idx

    self.dropout_p = dropout_p

    self.emb_dropout = nn.Dropout(p=dropout_p)
    self.embedding = nn.Embedding(n_words, emb_dim, padding_idx=form_padding_idx)

    self.char_input = True if char_emb_dim > 0 else False
    self.char_emb_dim = char_emb_dim

    self.pos_embedding = None
    self.pos_dropout = None
    self.rnn_input_size = emb_dim

    if self.char_input:
      if char_model == "simple":
        self.encode_characters = SimpleCharModel(n_chars, char_padding_idx,
                                                 emb_dim=char_emb_dim, hidden_size=100, output_dim=emb_dim, bi=True)
      elif char_model == "dozat":
        self.encode_characters = DozatCharModel(n_chars, char_padding_idx,
                                                emb_dim=char_emb_dim, hidden_size=400, output_dim=emb_dim, bi=False)
      elif char_model == "cnn":
        self.encode_characters = CharCNN(n_chars, char_padding_idx, char_emb_dim, num_filters, window_size, dropout_p=dropout_p)
      else:
        raise NotImplementedError("Unknown char. model type in RecurrentEncoder (options: dozat, simple, cnn)")

      self.char_model = char_model
      if char_model == "dozat" or char_model == "simple":
        self.rnn_input_size += emb_dim
      elif char_model == "cnn":
        self.rnn_input_size += num_filters

    # this is our own LSTM that supports variational dropout
    self.num_layers = num_layers
    if var_drop:
      var_dropout_p = 0.5
    else:
      var_dropout_p = 0
    self.rnn = LSTM(self.rnn_input_size, self.dim, num_layers, bias=True,
                    batch_first=False, dropout=dropout_p,
                    bidirectional=bidirectional)
    self.bi = bidirectional
    self.hidden_size = self.dim

    self.xavier_uniform()

  def xavier_uniform(self, gain=1., forget_bias=1.):

    # default pytorch initialization
    for name, weight in self.rnn.named_parameters():
      if len(weight.size()) > 1:
          nn.init.xavier_uniform_(weight.data, gain=gain)
      elif "bias_ih" or "bias_hh" in name:

        # all biases to 0 except forget gate
        weight.data.fill_(0.)
        forget_idx = weight.data.size()[0] // 4
        weight.data[forget_idx:2*forget_idx] = forget_bias

  def _get_hidden(self, batch_size):
    """Returns empty initial hidden state for each cell."""
    first_dim = 2 if not self.bi else 4
    hx = Variable(torch.zeros(first_dim, batch_size, self.hidden_size))
    cx = Variable(torch.zeros(first_dim, batch_size, self.hidden_size))
    hx = hx.cuda() if use_cuda else hx
    cx = cx.cuda() if use_cuda else cx
    return hx, cx

  def forward(self, form_var=None, char_var=None, pos_var=None, lengths=None, word_lengths=None):

    assert form_var is not None, 'must provide words (form) to RecurrentEncoder'

    # embed words
    embedded = self.embedding(form_var)

    # embed chars
    if self.char_input:
      embedded_words, word_lengths = self.encode_characters(char_var, word_lengths)
      embedded = torch.cat((embedded_words, embedded), dim=2)

    embedded = self.emb_dropout(embedded)
    embedded = torch.transpose(embedded, 0, 1)  # (time, batch, dim)
    embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)

    hx, cx = self._get_hidden(embedded[1][0].item())
    output, states = self.rnn(embedded, (hx, cx))

    output, lengths = nn.utils.rnn.pad_packed_sequence(output)

    # make batch-major again
    output = output.transpose(0, 1)
    output = self.emb_dropout(output)

    return output, states
