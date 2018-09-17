import io
import os

from torchtext.data import Field, NestedField
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from tagger.utils import PAD_TOKEN, UNK_TOKEN, ROOT_TOKEN, START_TOKEN, END_TOKEN
from collections import Counter


def get_data_fields_conllu():
  """Creates torchtext fields for the I/O pipeline."""
  form = Field(
    include_lengths=True, batch_first=True,
    init_token=None, eos_token=None, pad_token=PAD_TOKEN, lower=True)
  pos = Field(
    include_lengths=True, batch_first=True, init_token=ROOT_TOKEN, eos_token=END_TOKEN, pad_token=PAD_TOKEN,
    unk_token=None)
  nesting_field = Field(tokenize=list, pad_token=PAD_TOKEN, batch_first=True,
                        init_token=None, eos_token=None)
  chars = NestedField(nesting_field, init_token=None, pad_token=PAD_TOKEN, eos_token=None, include_lengths=True)


  fields = {
    'form':   ('form', form),
    'pos':    ('pos', pos),
    'chars': ('chars', chars)
  }

  return fields


def empty_conllu_example_dict():
  ex = {
    'id':      [],
    'form':    [],
    'lemma':   [],
    'pos':    [],
    'upos':     [],
    'feats':   [],
    'head':    [],
    'deprel':  [],
    'deps':   [],
    'misc': [],
    'chars': [],
    'crf_pos': []
  }
  return ex


def conllu_reader(f):
  """
  Return examples as a dictionary.
  Args:
    f:
  Returns:
  """

  ex = empty_conllu_example_dict()

  for line in f:
    line = line.strip()

    if not line:
      yield ex
      ex = empty_conllu_example_dict()
      continue

    # comments
    if line[0] == "#":
      continue

    parts = line.split()
    assert len(parts) == 10, "invalid conllx line: %s" % line

    _id, _form, _lemma, _upos, _xpos, _feats, _head, _deprel, _deps, _misc = parts

    chars = []
    for char in list(_form):
      # if char.isdigit():
      #   chars.append('0')
      # else:
      #   chars.append(char)
      chars.append(char)
    ex['form'].append(''.join(chars))
    ex['pos'].append(_xpos)
    ex['chars'].append(chars)

  # possible last sentence without newline after
  if len(ex['form']) > 0:
    yield ex


class ConllUDataset(Dataset):
  """Defines a CONLL-U Dataset. """

  @staticmethod
  def sort_key(ex):
    return len(ex.form)

  def __init__(self, path, fields, **kwargs):
    """Create a ConllUDataset given a path and field list.
    Arguments:
        path (str): Path to the data file.
        fields (dict[str: tuple(str, Field)]):
            The keys should be a subset of the columns, and the
            values should be tuples of (name, field).
            Keys not present in the input dictionary are ignored.
    """

    with io.open(os.path.expanduser(path), encoding="utf8") as f:

      # examples = [Example.fromdict(d, fields) for d in conllu_reader(f)]
      # count = 0
      if "train" in path:
        examples = []
        for d in conllu_reader(f):
          if len(Example.fromdict(d, fields).form) <= 70:
            examples.append(Example.fromdict(d, fields))
      else:
        examples = [Example.fromdict(d, fields) for d in conllu_reader(f)]

        # if len(Example.fromdict(d, fields).form) > 60:
      #     count += 1
      # print(count)

    if isinstance(fields, dict):
      fields, field_dict = [], fields
      for field in field_dict.values():
        if isinstance(field, list):
          fields.extend(field)
        else:
          fields.append(field)

    super(ConllUDataset, self).__init__(examples, fields, **kwargs)
