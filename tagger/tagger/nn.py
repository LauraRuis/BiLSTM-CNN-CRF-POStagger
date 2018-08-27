import logging
import torch
import math
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from tagger.lstm import LSTM
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from tagger.utils import ROOT_TOKEN, START_TOKEN, END_TOKEN, PAD_TOKEN, log_sum_exp, logsumexp, viterbi_decode
from typing import List, Tuple, Dict

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
    sorted_char_input = char_input.index_select(0, sort_idx)

    # get only non padding sequences (namely remove all sequences with length 0 coming from sentence padding)
    non_padding_idx = (sorted_lengths != 0).long().sum()
    char_input_no_pad = sorted_char_input[:non_padding_idx]
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
    unsorted_out = torch.zeros(batch_size * seq_len, dim)
    unsorted_out = unsorted_out.cuda() if torch.cuda.is_available() else unsorted_out
    unsorted_out = unsorted_out.scatter_(0, odx, output)
    unsorted_out = unsorted_out.view(batch_size, seq_len, dim)

    return unsorted_out, lengths

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
                                  dropout_p=dropout_p, bi=False)

    self.conv = nn.Conv1d(emb_dim, num_filters, window_size, padding=window_size - 1)
    self.xavier_uniform()

  def xavier_uniform(self, gain=1.):

    # default pytorch initialization
    for name, weight in self.conv.named_parameters():
      if len(weight.size()) > 1:
          nn.init.xavier_uniform_(weight.data, gain=gain)
      elif "bias" in name:
        weight.data.fill_(0.)

  def char_model(self, embedded=None, char_input_no_pad=None, lengths=None):

    embedded = torch.transpose(embedded, 1, 2)  # (bsz, dim, time)
    chars_conv = self.conv(embedded)
    chars = F.max_pool1d(chars_conv, kernel_size=chars_conv.size(2)).squeeze(2)

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


class ChainCRF(nn.Module):

    def __init__(self, n_tags, tag_to_ix):
        super(ChainCRF, self).__init__()

        self.tag_to_ix = tag_to_ix
        self.n_tags = n_tags

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.log_transitions = nn.Parameter(torch.randn(self.n_tags, self.n_tags))

        self.xavier_uniform()

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.log_transitions.data[:, tag_to_ix[ROOT_TOKEN]] = -10000.
        self.log_transitions.data[tag_to_ix[END_TOKEN], :] = -10000.

    def xavier_uniform(self, gain=1.):
        torch.nn.init.xavier_normal_(self.log_transitions)

    def _select_from_matrix(self, matrix, rows=None, cols=None):
        bsz = rows.size(0)
        selection = (
                    matrix
                    # Choose the current_tag-th row for each input
                    .gather(1, rows.view(bsz, 1, 1).expand(bsz, 1, self.n_tags))
                    # Squeeze down to (batch_size, num_tags)
                    .squeeze(1)
                    # Then choose the next_tag-th column for each of those
                    .gather(1, cols.view(bsz, 1))
                    # And squeeze down to (batch_size,)
                    .squeeze(1)
            )
        return selection

    def _forward_belief_prop_logspace(self, feats, mask):

        bsz, max_time, dim = feats.size()

        # initialize the recursion variables with transitions from root token + first emission probabilities
        init_alphas = self.log_transitions[self.tag_to_ix[ROOT_TOKEN], :] + feats[:, 0, :]

        # set recursion variable
        forward_var = init_alphas

        # make time major
        feats = torch.transpose(feats, 0, 1)  # (time, batch_size, dim)
        mask = torch.transpose(mask.float(), 0, 1)  # (time, batch_size)

        # loop over sequence and calculate the transition probability for the next tag at each step (from t-1 to t)
        # current tag at t - 1, next tag at t
        # emission probabilities: (example, next tag)
        # transition probabilities: (current tag, next tag)
        # forward var: (instance, current tag)
        # next tag var: (instance, current tag, next tag)
        for t in range(1, max_time):

            # get emission scores for this time step
            feat = feats[t]

            # broadcast emission probabilities
            emit_scores = feat.view(bsz, self.n_tags).unsqueeze(1)

            # calculate transition probabilities (broadcast over example axis, same for all examples in batch)
            trans_scores = self.log_transitions.unsqueeze(0)

            # calculate next tag probabilities
            next_tag_var = forward_var.unsqueeze(2) + emit_scores + trans_scores

            # calculate next forward var by taking logsumexp over next tag axis, mask all instances that ended
            # and keep old forward var for instances those
            forward_var = (logsumexp(next_tag_var, 1) * mask[t].view(bsz, 1) +
                           forward_var * (1 - mask[t]).view(bsz, 1))

        final_transitions = self.log_transitions[:, self.tag_to_ix[END_TOKEN]]

        alphas = forward_var + final_transitions.unsqueeze(0)
        partition_function = logsumexp(alphas)

        return partition_function

    def _score_sentence_logspace(self, feats, tags, mask):

        bsz, max_time, dim = feats.size()

        # make time major
        feats = feats.transpose(0, 1)  # (time, batch_size, dim)
        mask = mask.float().transpose(0, 1)  # (time, batch_size)
        tags = tags.transpose(0, 1)  # (time, batch_size)

        # expand transitions for each example in bach
        transitions = self.log_transitions.view(
            1, self.n_tags, self.n_tags).expand(
            bsz, self.n_tags, self.n_tags)

        # get tensor of root tokens and tensor of next tags (first tags)
        root_tags = torch.LongTensor([self.tag_to_ix[ROOT_TOKEN]] * bsz).unsqueeze(1)
        root_tags = root_tags.cuda() if torch.cuda.is_available() else root_tags
        next_tags = tags[0]

        # initial transition is from root token to first tags
        initial_transition = self._select_from_matrix(transitions, rows=root_tags, cols=next_tags)

        # initialize scores
        scores = initial_transition

        # loop over time and add each time calculate the score from t to t + 1
        for t in range(max_time - 1):

            # get emission scores, transition scores and calculate score for current time step
            feat = feats[t]
            next_tags = tags[t + 1]
            current_tags = tags[t]
            emis = torch.gather(feat, 1, current_tags.unsqueeze(1)).squeeze()
            trans = self._select_from_matrix(transitions, rows=current_tags, cols=next_tags)

            # add scores
            scores = scores + trans * mask[t + 1] + emis * mask[t]

        # add scores for transitioning to stop tag
        last_tag_index = mask.sum(0).long() - 1
        last_tags = torch.gather(tags, 0, last_tag_index.view(1, bsz)).view(-1, 1)  # TODO: add this line to AllenNLP

        # end_tags
        end_tags = torch.LongTensor([self.tag_to_ix[END_TOKEN]] * bsz).unsqueeze(1)
        end_tags = end_tags.cuda() if torch.cuda.is_available() else end_tags

        last_transition = self._select_from_matrix(transitions, rows=last_tags, cols=end_tags)

        # add the last input if its not masked
        last_inputs = feats[-1]
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))
        last_input_score = last_input_score.squeeze()

        scores = scores + last_transition + last_input_score * mask[-1]

        # sum over sequence length
        return scores

    def _viterbi_decode(self, feats, lengths):

        bsz, max_time, dim = feats.size()

        # initialize the viterbi variables in log space
        init_vars = torch.full((1, self.n_tags), -10000.)
        init_vars[0][self.tag_to_ix[ROOT_TOKEN]] = 0

        # initialize tensor and list to keep track of backpointers
        backpointers = torch.zeros(bsz, max_time, self.n_tags).long() - 1
        backpointers = backpointers.cuda() if torch.cuda.is_available() else backpointers
        best_last_tags = []
        best_path_scores = []

        # forward_var at step t holds the viterbi variables for step t - 1, diff per example in batch
        forward_var = init_vars.unsqueeze(0).repeat(bsz, 1, 1)
        forward_var = forward_var.cuda() if torch.cuda.is_available() else forward_var

        # counter counting down from number of examples in batch to 0
        counter = bsz

        # loop over sequence
        for t in range(max_time):

            # if time equals some lengths in the batch, these sequences are ending
            ending = (lengths == t).nonzero()
            n_ending = len(ending)

            # if there are sequences ending
            if n_ending > 0:

                # grab their viterbi variables
                forward_ending = forward_var[(counter - n_ending):counter]

                # the terminal var giving the best last tag is the viterbi variables + trans. prob. to end token
                terminal_var = forward_ending + self.log_transitions[:, self.tag_to_ix[END_TOKEN]].unsqueeze(0)
                path_scores, best_tag_idx = torch.max(terminal_var, 1)

                # first sequence to end is last sequence in batch (sorted on sequence length)
                for tag, score in zip(reversed(list(best_tag_idx)), reversed(list(path_scores))):
                    best_last_tags.append(tag)
                    best_path_scores.append(score)

                # update counter keeping track of how many sequences already ended
                counter -= n_ending

            # get emission probabilities at current time step
            feat = feats[:, t, :].view(bsz, self.n_tags)

            # calculate scores of next tag
            forward_var = forward_var.view(bsz, self.n_tags, 1)
            trans_scores = self.log_transitions.unsqueeze(0)
            next_tag_vars = forward_var + trans_scores

            # get best next tags and viterbi vars
            viterbivars_t, idx = torch.max(next_tag_vars, 1)
            best_tag_ids = idx.view(bsz, -1)

            # add emission scores and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (viterbivars_t + feat).view(bsz, -1)

            # save best tags as backpointers
            backpointers[:, t, :] = best_tag_ids.long()

        # get final ending sequence(s) and calculate the best last tag(s)
        ending = (lengths == max_time).nonzero()
        ending = ending.cuda() if torch.cuda.is_available() else ending
        n_ending = len(ending)

        if n_ending > 0:

            forward_ending = forward_var[(counter - n_ending):counter]

            # transition to STOP_TAG
            terminal_var = forward_ending + self.log_transitions[:, self.tag_to_ix[END_TOKEN]].unsqueeze(0)
            path_scores, best_tag_idx = torch.max(terminal_var, 1)

            for tag, score in zip(reversed(list(best_tag_idx)), reversed(list(path_scores))):
                best_last_tags.append(tag)
                best_path_scores.append(score)

        # reverse the best last tags (and scores) to put them back in the original batch order
        best_last_tags = torch.LongTensor(list(reversed(best_last_tags)))
        best_last_tags = best_last_tags.cuda() if torch.cuda.is_available() else best_last_tags
        best_path_scores = torch.LongTensor(list(reversed(best_path_scores)))
        best_path_scores = best_path_scores.cuda() if torch.cuda.is_available() else best_path_scores

        # follow the back pointers to decode the best path
        best_paths = torch.zeros(bsz, max_time + 1).long()
        best_paths = best_paths.cuda() if torch.cuda.is_available() else best_paths
        best_paths = best_paths.index_put_((torch.LongTensor([i for i in range(backpointers.size(0))]), lengths),
                                            best_last_tags)

        # counter keeping track of number of active sequences
        num_active = 0

        # loop from max time to 0
        for t in range(max_time - 1, -1, -1):

            # if time step equals lengths of some sequences, they are starting
            starting = (lengths - 1 == t).nonzero()
            n_starting = len(starting)

            # if there are sequences starting, grab their best last tags
            if n_starting > 0:
                if t == max_time - 1:
                    best_tag_id = best_paths[num_active:num_active + n_starting, t + 1]
                else:
                    last_tags = best_paths[num_active:num_active + n_starting, t + 1]
                    best_tag_id = torch.cat((best_tag_id, last_tags.unsqueeze(1)), dim=0)

                # update number of active sequences
                num_active += n_starting

            # get currently relevant backpointers based on sequences that are active
            active = backpointers[:num_active, t]

            # follow the backpointers to the best previous tag
            best_tag_id = best_tag_id.view(num_active, 1)
            best_tag_id = torch.gather(active, 1, best_tag_id)
            best_paths[:num_active, t] = best_tag_id.squeeze()

        # add end tokens at the end of every sequence in the batch
        end_token_tensor = torch.zeros(bsz).unsqueeze(1).long()
        end_token_tensor = end_token_tensor.cuda() if torch.cuda.is_available() else end_token_tensor
        best_paths = torch.cat((best_paths, end_token_tensor), dim=1)
        end_token = torch.LongTensor([self.tag_to_ix[END_TOKEN]])[0]
        end_token = end_token.cuda() if torch.cuda.is_available() else end_token
        best_paths = best_paths.index_put_((torch.LongTensor([i for i in range(best_paths.size(0))]), lengths + 1),
                                           end_token)

        # sanity check that first tag is the root token
        assert best_paths[:, 0].sum().item() == best_paths.size(0) * self.tag_to_ix[ROOT_TOKEN]

        return best_path_scores, best_paths

    def neg_log_likelihood(self, feats, tags, mask):
        forward_score = self._forward_belief_prop_logspace(feats, mask)
        gold_score = self._score_sentence_logspace(feats, tags, mask)
        return forward_score - gold_score

    def forward(self, lstm_feats, tags, mask, lengths, training=True):

        if training:
          loss = self.neg_log_likelihood(lstm_feats, tags, mask)
          with torch.no_grad():
              score, tag_seq = self._viterbi_decode(lstm_feats, lengths)
          loss = loss.sum() / loss.size(0)
        else:
          loss = None
          score, tag_seq = self._viterbi_decode(lstm_feats, lengths)

        return loss, score, tag_seq