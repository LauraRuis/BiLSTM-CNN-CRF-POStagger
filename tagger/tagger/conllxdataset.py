import io
import os

from torchtext.data import Field, NestedField
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from tagger.utils import PAD_TOKEN, UNK_TOKEN, ROOT_TOKEN, END_TOKEN


def get_data_fields():
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


def empty_conllx_example_dict():
  ex = {
    'id':      [],
    'form':    [],
    'lemma':   [],
    'cpos':    [],
    'pos':     [],
    'feats':   [],
    'head':    [],
    'deprel':  [],
    'phead':   [],
    'pdeprel': [],
    'chars': []
  }
  return ex


def conllx_reader(f):
  """
  Return examples as a dictionary.
  Args:
    f:
  Returns:
  """

  ex = empty_conllx_example_dict()

  for line in f:
    line = line.strip()

    if not line:
      yield ex
      ex = empty_conllx_example_dict()
      continue

    parts = line.split()

    assert len(parts) == 2, "invalid conllx line: %s" % line

    _form, _pos = parts
    chars = []
    for char in list(_form):
      # if char.isdigit():
      #   chars.append('0')
      # else:
      #   chars.append(char)
      chars.append(char)
    ex['form'].append(''.join(chars))
    ex['pos'].append(_pos)
    ex['chars'].append(chars)

  # possible last sentence without newline after
  if len(ex['form']) > 0:
    yield ex


class ConllXDataset(Dataset):
  """Defines a CONLL-X Dataset. """

  @staticmethod
  def sort_key(ex):
    return len(ex.form)

  def __init__(self, path, fields, **kwargs):
    """Create a ConllXDataset given a path and field list.
    Arguments:
        path (str): Path to the data file.
        fields (dict[str: tuple(str, Field)]):
            The keys should be a subset of the columns, and the
            values should be tuples of (name, field).
            Keys not present in the input dictionary are ignored.
    """
    self.n_tokens = 0
    with io.open(os.path.expanduser(path), encoding="utf8") as f:
      examples = []
      for d in conllx_reader(f):
        if len(d["form"]) >= 70 and "train" in path:
          continue
        else:
          self.n_tokens += len(Example.fromdict(d, fields).form)
          examples.append(Example.fromdict(d, fields))
        # examples.append(Example.fromdict(d, fields))

    if isinstance(fields, dict):
      fields, field_dict = [], fields
      for field in field_dict.values():
        if isinstance(field, list):
          fields.extend(field)
        else:
          fields.append(field)

    super(ConllXDataset, self).__init__(examples, fields, **kwargs)