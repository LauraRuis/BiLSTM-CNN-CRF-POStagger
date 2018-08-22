#!/usr/bin/env python3

import logging
import argparse
from tagger.train import train

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger(__name__)


def main():

  ap = argparse.ArgumentParser(description="a simple POS tagger")
  ap.add_argument('--mode', choices=['train', 'predict'], default='train')
  ap.add_argument('--output_dir', type=str, default='output')

  # data
  ap.add_argument('--train_path', type=str,
                  default='data/wsj/pos/train.tsv')
  ap.add_argument('--dev_path', type=str,
                  default='data/wsj/pos/dev.tsv')
  ap.add_argument('--test_path', type=str,
                  default='data/wsj/pos/test.tsv')

  # parameters
  ap.add_argument('--n_iters', type=int, default=150000)
  ap.add_argument('--dim', type=int, default=100)
  ap.add_argument('--emb_dim', type=int, default=100)
  ap.add_argument('--char_emb_dim', type=int, default=30)
  ap.add_argument('--window_size', type=int, default=3)
  ap.add_argument('--num_filters', type=int, default=30)
  ap.add_argument('--char_model', choices=['dozat', 'simple', 'cnn', 'none'], default='cnn')
  ap.add_argument('--batch_size', type=int, default=238)
  ap.add_argument('--num_layers', type=int, default=1)
  ap.add_argument('--print_every', type=int, default=300)
  ap.add_argument('--eval_every', type=int, default=3000)
  ap.add_argument('--uni', dest='bi', default=True, action='store_false',
                  help='use unidirectional encoder')

  ap.add_argument('--glove', dest='glove', default=True, action='store_true')
  ap.add_argument('--no_glove', dest='glove', default=False, action='store_false')

  # dropout
  ap.add_argument('--dropout_p', type=float, default=0.5)
  ap.add_argument('--clip', type=float, default=5.)

  # optimizer
  ap.add_argument('--optimizer', type=str, default='sgd')
  ap.add_argument('--lr', type=float, default=0.01)
  ap.add_argument('--lr_decay', type=float, default=0.95)
  ap.add_argument('--lr_decay_steps', type=float, default=5000)
  ap.add_argument('--adam_beta1', type=float, default=0.9)
  ap.add_argument('--adam_beta2', type=float, default=0.9)
  ap.add_argument('--weight_decay', type=float, default=0.)
  ap.add_argument('--momentum', type=float, default=0.9)
  ap.add_argument('--plateau', dest='plateau', default=False, action='store_true')

  # others
  ap.add_argument('--seed', type=int, default=42)
  ap.add_argument('--resume',  type=str, default='')

  cfg = vars(ap.parse_args())

  if cfg["char_model"] != "None":
    assert cfg["char_emb_dim"] > 0, "if using character model, set char_emb_dim to value larger than 0"
  else:
    assert cfg["char_emb_dim"] == 0, "if not using character model, set char_emb_dim to 0"

  print("Config:")
  for k, v in cfg.items():
    print("  %12s : %s" % (k, v))

  if cfg['mode'] == 'train':
    train(**cfg)
  elif cfg['mode'] == 'predict':
    raise NotImplementedError()


if __name__ == '__main__':
  main()
