import time
import logging
import torch.nn.functional as F

import torch
from torchtext.data import Iterator

from tagger.utils import read_conllx, read_conllu

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


# def get_conll_str(example=None, heads=None, deprels=None):
#   s = []
#   for tid, form, pos, head, deprel in zip(
#     count(start=1), example.form, example.pos, heads, deprels):
#     s.append(get_conllx_line(tid=tid, form=form, pos=pos, cpos=pos,
#                              head=head, deprel=deprel))
#
#   return "\n".join(s)


def predict_and_save(dataset=None, model=None, dataset_path='dev.conll',
                     out_path='predict.conll', **kwargs):
  """Combine original CONLL-X file with predictions.
  This is required since the iterator might have changed certain fields
  (e.g. lowercasing).
  We read the dataset_path separately and replace the fields we predicted.
  """
  device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
  data_iter = Iterator(dataset, 1, train=False, sort=False, shuffle=False,
                       device=device)
  start_time = time.time()

  i2pos = dataset.fields['pos'].vocab.itos

  with open(out_path, mode='w', encoding='utf-8') as f:
    with open(dataset_path, mode='r', encoding='utf-8') as data_f:
      with torch.no_grad():

        if "ud" in dataset_path:
          original_iter = read_conllu(data_f)
          ud = True
        else:
          original_iter = read_conllx(data_f)
          ud = False

        for pred in predict(data_iter=data_iter, model=model):

          tokens = next(original_iter)

          pred_tags = [-1] * len(tokens)
          write_tag = False
          if len(pred["pos"]) > 0:
            pred_tags = pred["pos"].data.view(-1).tolist()
            write_tag = True

          for tok, pred_tag in \
                  zip(tokens, pred_tags):
            if write_tag:
              if ud:
                tok.xpos = i2pos[pred_tag]
              else:
                tok.pos = i2pos[pred_tag]

            f.write(str(tok) + '\n')
          f.write('\n')


def predict(data_iter=None, model=None):

  model.eval()  # disable dropout
  start_time = time.time()
  n_words = 0
  for i, batch in enumerate(iter(data_iter)):
    form_var, lengths = batch.form
    char_var, sequence_lengths, word_lengths = batch.chars
    lengths = lengths.view(-1).tolist()
    n_words += sum(lengths)

    pos_var, pos_lengths = batch.pos

    # predict
    encoder_output, _ = model.encode_input(form_var=form_var, pos_var=pos_var, char_var=char_var,
                                                   lengths=lengths, word_lengths=word_lengths)

    result = model.pos_tagger(encoder_output, pos_var, lengths, pos_lengths)

    # get predicted pos
    if model.tagger == "linear" or model.tagger == "mlp":
      pos_predictions = result['output'].max(2)[1]
    else:
      pos_predictions = result['sequence']

    predictions = dict(pos=pos_predictions)

    yield predictions

  elapsed_time = time.time() - start_time
  wps = n_words / elapsed_time
