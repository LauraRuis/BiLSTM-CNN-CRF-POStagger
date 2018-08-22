import logging
import torch
import torch.nn as nn
from tagger.lstm import LSTM
from tagger.nn import DozatCharModel, SimpleCharModel, CharCNN

use_cuda = True if torch.cuda.is_available() else False
logger = logging.getLogger(__name__)


class RecurrentEncoder(nn.Module):
  def __init__(self, dim=0, emb_dim=0, char_emb_dim=100, n_words=0, n_tags=0, n_chars=0,
               dropout_p=0.33, bidirectional=True, num_layers=1,
               num_filters=0, window_size=0,
               form_padding_idx=1, pos_padding_idx=0, char_padding_idx=0,
               char_model=None):
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
                                                 emb_dim=char_emb_dim, hidden_size=100, output_dim=50, bi=True)
      elif char_model == "dozat":
        self.encode_characters = DozatCharModel(n_chars, char_padding_idx,
                                                emb_dim=char_emb_dim, hidden_size=400, output_dim=emb_dim, bi=False)
      elif char_model == "cnn":
        self.encode_characters = CharCNN(n_chars, char_padding_idx, emb_dim, num_filters, window_size, dropout_p=0.33)
      else:
        raise NotImplementedError("Unknown char. model type in RecurrentEncoder (options: dozat, simple, cnn)")

      self.char_model = char_model
      if char_model == "dozat" or char_model == "simple":
        self.rnn_input_size += 50
      elif char_model == "cnn":
        self.rnn_input_size += num_filters

    # this is our own LSTM that supports variational dropout
    self.num_layers = num_layers
    self.rnn = LSTM(self.rnn_input_size, self.dim, num_layers, bias=True,
                    batch_first=False, dropout=dropout_p,
                    bidirectional=bidirectional)

  def forward(self, form_var=None, char_var=None, pos_var=None, lengths=None, word_lengths=None):

    assert form_var is not None, 'must provide words (form) to RecurrentEncoder'

    # embed words
    embedded = self.embedding(form_var)
    embedded = self.emb_dropout(embedded)

    # embed chars
    if self.char_input:
      embedded_words, word_lengths = self.encode_characters(char_var, word_lengths)
      embedded_words = self.emb_dropout(embedded_words)
      embedded = torch.cat((embedded_words, embedded), dim=2)

    embedded = torch.transpose(embedded, 0, 1)  # (time, batch, dim)
    embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
    output, states = self.rnn(embedded)

    output, lengths = nn.utils.rnn.pad_packed_sequence(output)

    # make batch-major again
    output = output.transpose(0, 1)

    return output, states
