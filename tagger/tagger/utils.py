#!/usr/bin/env python3

import os
from itertools import count
import shutil
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.init import orthogonal

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
ROOT_TOKEN = '<root>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'

use_cuda = True if torch.cuda.is_available() else False


def init_gru(cell, gain=1):
  cell.reset_parameters()

  # orthogonal initialization of recurrent weights
  for _, hh, _, _ in cell.all_weights:
    for i in range(0, hh.size(0), cell.hidden_size):
      orthogonal(hh[i:i + cell.hidden_size], gain=gain)


def init_lstm(cell, gain=1, forget_bias=1.):
  init_gru(cell, gain)

  # positive forget gate bias (Jozefowicz et al., 2015)
  for _, _, ih_b, hh_b in cell.all_weights:
    l = len(ih_b)
    ih_b[l // 4:l // 2].data.fill_(forget_bias)
    hh_b[l // 4:l // 2].data.fill_(forget_bias)


def get_conllx_line(tid=1, form='_', lemma='_', cpos='_', pos='_',
                    feats='_', head='_', deprel='_', phead='_', pdelrel='_'):
  return '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t' % (
    tid, form, lemma, cpos, pos, feats, head, deprel, phead, pdelrel)


def save_checkpoint(output_dir, state, is_best, filename='checkpoint.pth.tar'):
  path = os.path.join(output_dir, filename)
  torch.save(state, path)
  if is_best:
    best_path = os.path.join(output_dir, 'model_best.pth.tar')
    shutil.copyfile(path, best_path)


def one_hot(indices, length=10):
  """Return a one-hot tensor based on indices."""
  one_hot = torch.zeros(indices.size() + (length,))
  one_hot = Variable(one_hot)
  one_hot = one_hot.cuda() if use_cuda else one_hot
  indices = indices.unsqueeze(-1)
  one_hot.scatter_(-1, indices, 1)
  return one_hot


def print_parameters(model):
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  n_params = sum([np.prod(p.size()) for p in model_parameters])
  print("Total params: %d" % n_params)
  for name, p in model.named_parameters():
    if p.requires_grad:
      print("%s : %s" % (name, list(p.size())))


class XToken:
  """Conll-X Token Representation"""

  def __init__(self, tid, form, lemma, cpos, pos, feats,
               head, deprel, phead, pdelrel):
    self.id = int(tid)
    self.form = form
    self.lemma = lemma
    self.cpos = cpos
    self.pos = pos
    self.feats = feats
    self.head = int(head)
    self.deprel = deprel
    self.phead = phead
    self.pdeprel = pdelrel

  def __str__(self):
    return '%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s' % (
      self.id, self.form, self.lemma, self.cpos, self.pos, self.feats,
      self.head, self.deprel, self.phead, self.pdeprel)

  def __repr__(self):
    return self.__str__()


class UToken:
  """Conll-U Token Representation """

  def __init__(self, tid, form, lemma, upos, xpos, feats,
               head, deprel, deps, misc):
    """
    Args:
      tid: Word index, starting at 1; may be a range for multi-word tokens;
        may be a decimal number for empty nodes.
      form: word form or punctuation symbol.
      lemma: lemma or stem of word form
      upos: universal part-of-speech tag
      xpos: language specific part-of-speech tag
      feats: morphological features
      head: head of current word (an ID or 0)
      deprel: universal dependency relation to the HEAD (root iff HEAD = 0)
      deps: enhanced dependency graph in the form of a list of head-deprel pairs
      misc: any other annotation
    """
    self.id = float(tid)
    self.form = form
    self.lemma = lemma
    self.upos = upos
    self.xpos = xpos
    self.feats = feats
    self.head = head
    self.deprel = deprel
    self.deps = deps
    self.misc = misc

  def __str__(self):
    return '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
      self.id, self.form, self.lemma, self.upos, self.xpos, self.feats,
      self.head, self.deprel, self.deps, self.misc)

  def __repr__(self):
    return self.__str__()


class POSToken:
  """Conll-U Token Representation """

  def __init__(self, form, pos):
    """
    Args:
      tid: Word index, starting at 1; may be a range for multi-word tokens;
        may be a decimal number for empty nodes.
      form: word form or punctuation symbol.
      lemma: lemma or stem of word form
      upos: universal part-of-speech tag
      xpos: language specific part-of-speech tag
      feats: morphological features
      head: head of current word (an ID or 0)
      deprel: universal dependency relation to the HEAD (root iff HEAD = 0)
      deps: enhanced dependency graph in the form of a list of head-deprel pairs
      misc: any other annotation
    """
    self.form = form
    self.pos = pos

  def __str__(self):
    return '%s\t%s' % (
      self.form, self.pos)

  def __repr__(self):
    return self.__str__()


def read_conllx(f):

  tokens = []

  for line in f:
    line = line.strip()

    if not line:
      yield tokens
      tokens = []
      continue

    parts = line.split()
    assert len(parts) == 2, "invalid conllx line"
    tokens.append(POSToken(*parts))

  # possible last sentence without newline after
  if len(tokens) > 0:
    yield tokens


def read_conllu(f):

  tokens = []

  for line in f:
    line = line.strip()

    if not line:
      yield tokens
      tokens = []
      continue

    if line[0] == "#":
      continue

    parts = line.split()
    assert len(parts) == 10, "invalid conllu line"
    tokens.append(UToken(*parts))

  # possible last sentence without newline after
  if len(tokens) > 0:
    yield tokens


def print_example(ex):

  r = ["%2d %12s %5s" % (i, f, p) for i, f, p in zip(
      count(start=1), ex.form, ex.pos)]
  print("\n".join(r))
  print()


if __name__ == '__main__':
  pass
