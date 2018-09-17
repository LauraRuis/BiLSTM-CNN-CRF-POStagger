import logging
import os
import numpy as np
from torchtext.data import Iterator
from torchtext.vocab import GloVe

from tagger.conllxdataset import ConllXDataset, get_data_fields
from tagger.conlludataset import ConllUDataset, get_data_fields_conllu
from tagger.utils import print_example, save_checkpoint, print_parameters
from tagger.predict import predict_and_save
from tagger.model import Tagger
from tagger.pos_acc import get_pos_acc

import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.optim

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


def train(mode='train', train_path='train.conllx', model='dozat', dataset='conllx',
          dev_path='dev.conllx', test_path='test.conllx',
          ud=True,
          output_dir='output',
          emb_dim=0, char_emb_dim=0,
          char_model=None, tagger=None,
          batch_size=5000,
          n_iters=10, dropout_p=0.33, num_layers=1,
          print_every=1, eval_every=100, bi=True, var_drop=False,
          lr=0.001, adam_beta1=0.9, adam_beta2=0.999, weight_decay=0., plateau=False,
          resume=False, lr_decay=1.0, lr_decay_steps=5000, clip=5., momentum=0,
          optimizer='adam', glove=True, seed=42, dim=0, window_size=0, num_filters=0, **kwargs):

  device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  cfg = locals().copy()

  torch.manual_seed(seed)
  np.random.seed(seed)

  # load data component
  if dataset == "conllx":
    dataset_obj = ConllXDataset
    fields = get_data_fields()
    ud = False
  elif dataset == "conllu":
    dataset_obj = ConllUDataset
    fields = get_data_fields_conllu()
    ud = True
  else:
    raise NotImplementedError()

  _form = fields['form'][-1]
  _pos = fields['pos'][-1]
  _chars = fields['chars'][-1]

  train_dataset = dataset_obj(train_path, fields)
  dev_dataset = dataset_obj(dev_path, fields)
  test_dataset = dataset_obj(test_path, fields)

  logger.info("Loaded %d train examples" % len(train_dataset))
  logger.info("Loaded %d dev examples" % len(dev_dataset))
  logger.info("Loaded %d test examples" % len(test_dataset))

  form_vocab_path = os.path.join(output_dir, 'vocab.form.pth.tar')
  pos_vocab_path = os.path.join(output_dir, 'vocab.pos.pth.tar')
  char_vocab_path = os.path.join(output_dir, 'vocab.char.pth.tar')

  if not resume:
    # build vocabularies
    # words have a min frequency of 2 to be included; others become <unk>
    # words without a Glove vector are initialized ~ N(0, 0.5) mimicking Glove

    # Note: this requires the latest torchtext development version from Github.
    # - git clone https://github.com/pytorch/text.git torchtext
    # - cd torchtext
    # - python setup.py build
    # - python setup.py install

    def unk_init(x):
      # return 0.01 * torch.randn(x)
      return torch.zeros(x)

    if glove:
      logger.info("Using Glove vectors")
      glove_vectors = GloVe(name='6B', dim=100)
      _form.build_vocab(train_dataset, min_freq=2,
                        unk_init=unk_init,
                        vectors=glove_vectors)
      n_unks = 0
      unk_set = set()
      # for now, set UNK words manually
      # (torchtext does not seem to support it yet)
      for i, token in enumerate(_form.vocab.itos):
        if token not in glove_vectors.stoi:
          n_unks += 1
          unk_set.add(token)
          _form.vocab.vectors[i] = unk_init(emb_dim)
      # print(n_unks, unk_set)

    else:
      _form.build_vocab(train_dataset, min_freq=2)

    _pos.build_vocab(train_dataset)
    _chars.build_vocab(train_dataset)

    # save vocabularies
    torch.save(_form.vocab, form_vocab_path)
    torch.save(_pos.vocab, pos_vocab_path)
    torch.save(_chars.vocab, char_vocab_path)

  else:
    # load vocabularies
    _form.vocab = torch.load(form_vocab_path)
    _pos.vocab = torch.load(pos_vocab_path)
    _chars.vocab = torch.load(char_vocab_path)

  print("First 10 vocabulary entries, words: ", " ".join(_form.vocab.itos[:10]))
  print("First 10 vocabulary entries, pos tags: ", " ".join(_pos.vocab.itos[:10]))
  print("First 10 vocabulary entries, chars: ", " ".join(_chars.vocab.itos[:10]))

  n_words = len(_form.vocab)
  n_tags = len(_pos.vocab)
  n_chars = len(_chars.vocab)

  def batch_size_fn(new, count, sofar):
    return len(new.form) + 1 + sofar

  # iterators
  train_iter = Iterator(train_dataset, batch_size, train=True,
                        sort_within_batch=True, batch_size_fn=batch_size_fn,
                        device=device)
  dev_iter = Iterator(dev_dataset, 32, train=False, sort_within_batch=True,
                      device=device)
  test_iter = Iterator(test_dataset, 32, train=False, sort_within_batch=True,
                       device=device)

  # uncomment to see what a mini-batch looks like numerically
  # e.g. some things are being inserted dynamically (ROOT at the start of seq,
  #   padding items, maybe UNKs..)
  # batch = next(iter(train_iter))
  # print("form", batch.form)
  # print("pos", batch.pos)
  # print("deprel", batch.deprel)
  # print("head", batch.head)

  # if n_iters or eval_every are negative, we set them to that many
  # number of epochs
  iters_per_epoch = (len(train_dataset) // batch_size) + 1
  if eval_every < 0:
    logger.info("Setting eval_every to %d epoch(s) = %d iters" % (
      -1 * eval_every, -1 * eval_every * iters_per_epoch))
    eval_every = iters_per_epoch * eval_every

  if n_iters < 0:
    logger.info("Setting n_iters to %d epoch(s) = %d iters" % (
        -1 * n_iters, -1 * n_iters * iters_per_epoch))
    n_iters = -1 * n_iters * iters_per_epoch

  # load up the model
  model = Tagger(n_words=n_words, n_tags=n_tags, n_chars=n_chars,
                 form_vocab=_form.vocab, char_vocab=_chars.vocab,
                 pos_vocab=_pos.vocab, **cfg)

  # set word vectors
  if glove:
    _form.vocab.vectors = _form.vocab.vectors / torch.std(_form.vocab.vectors)
    # print(torch.std(_form.vocab.vectors))
    model.encoder.embedding.weight.data.copy_(_form.vocab.vectors)
    model.encoder.embedding.weight.requires_grad = True

  model = model.cuda() if use_cuda else model

  start_iter = 1
  best_iter = 0
  best_pos_acc = -1.
  test_pos_acc = -1.

  # optimizer and learning rate scheduler
  trainable_parameters = [p for p in model.parameters() if p.requires_grad]
  if optimizer == 'sgd':
    optimizer = torch.optim.SGD(trainable_parameters, lr=lr, momentum=momentum)
  else:
    optimizer = torch.optim.Adam(trainable_parameters,
                                 lr=lr, betas=(adam_beta1, adam_beta2))

  # learning rate schedulers
  if not plateau:
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: lr_decay ** t)
  else:
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, min_lr=1e-4)

  # load model and vocabularies if resuming
  if resume:
    if os.path.isfile(resume):
      print("=> loading checkpoint '{}'".format(resume))
      checkpoint = torch.load(resume)
      start_iter = checkpoint['iter_i']
      best_pos_acc = checkpoint['best_pos_acc']
      test_pos_acc = checkpoint['test_pos_acc']
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print("=> loaded checkpoint '{}' (iter {})"
            .format(resume, checkpoint['iter_i']))
    else:
      print("=> no checkpoint found at '{}'".format(resume))

  print_parameters(model)

  # print some stuff just for fun
  logger.info("Most common words: %s" % _form.vocab.freqs.most_common(20))
  logger.info("Word vocab size: %s" % n_words)
  logger.info("Most common XPOS-tags: %s" % _pos.vocab.freqs.most_common())
  logger.info("POS vocab size: %s" % n_tags)
  # logger.info("Most common chars: %s" % _chars.nesting_field.vocab.freqs.most_common())
  logger.info("Chars vocab size: %s" % n_chars)

  print("First training example:")
  print_example(train_dataset[0])

  print("First dev example:")
  print_example(dev_dataset[0])

  print("First test example:")
  print_example(test_dataset[0])

  logger.info("Training starts..")
  upos_var, morph_var = None, None
  for iter_i in range(start_iter, n_iters + 1):

    if not plateau and iter_i % (912344 // batch_size) == 0:
      scheduler.step()
    model.train()

    batch = next(iter(train_iter))
    form_var, lengths = batch.form

    pos_var, pos_lengths = batch.pos
    char_var, sentence_lengths, word_lengths = batch.chars
    lengths = lengths.view(-1).tolist()

    result = model(form_var=form_var, char_var=char_var,  pos_var=pos_var,
                   lengths=lengths, word_lengths=word_lengths, pos_lengths=pos_lengths)

    # rows sum to 1
    # print(torch.exp(output_graph).sum(-1))

    # print sizes
    # print(head_logits.data.cpu().size())
    targets = dict(pos=batch.pos)

    all_losses = model.get_loss(scores=result,
                                targets=targets)

    loss = all_losses['loss']

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    optimizer.zero_grad()

    if iter_i % print_every == 0:

      # get scores for this batch
      if model.tagger == "linear" or "mlp":
          pos_predictions = result['output'].max(2)[1]
      else:
          pos_predictions = result['sequence']
      predictions = dict(pos=pos_predictions)
      targets = dict(pos=batch.pos)

      pos_acc = model.get_accuracy(predictions=predictions, targets=targets)

      if not plateau:
        lr = scheduler.get_lr()[0]
      else:
        lr = [group['lr'] for group in optimizer.param_groups][0]

      fmt = "Iter %08d loss %8.4f pos-acc %5.2f lr %.5f"

      logger.info(fmt % (iter_i, loss, pos_acc, lr))

    if iter_i % eval_every == 0:

      # parse dev set and save to file for official evaluation
      dev_out_path = 'dev.iter%08d.conll' % iter_i
      dev_out_path = os.path.join(output_dir, dev_out_path)
      predict_and_save(dataset=dev_dataset, model=model,
                       dataset_path=dev_path,
                       out_path=dev_out_path)

      _dev_pos_acc = get_pos_acc(dev_path, dev_out_path, ud)

      logger.info("Evaluation dev Iter %08d "
                  "pos-acc %5.2f" % (
                    iter_i, _dev_pos_acc))

      # parse test set and save to file for official evaluation
      test_out_path = 'test.iter%08d.conll' % iter_i
      test_out_path = os.path.join(output_dir, test_out_path)
      predict_and_save(dataset=test_dataset, model=model,
                       dataset_path=test_path,
                       out_path=test_out_path)
      _test_pos_acc = get_pos_acc(test_path, test_out_path, ud)

      logger.info("Evaluation test Iter %08d "
                  "pos-acc %5.2f" % (
                    iter_i, _test_pos_acc))

      if plateau:
        scheduler.step(_dev_pos_acc)

      if _dev_pos_acc > best_pos_acc:
        best_iter = iter_i
        best_pos_acc = _dev_pos_acc
        test_pos_acc = _test_pos_acc
        is_best = True
      else:
        is_best = False

      save_checkpoint(
        output_dir,
        {
              'iter_i':     iter_i,
              'state_dict': model.state_dict(),
              'best_iter':  best_iter,
              'test_pos_acc':   test_pos_acc,
              'optimizer':  optimizer.state_dict(),
        }, False)

  logger.info("Done Training")
  logger.info("Best model Iter %08d Dev POS-acc %12.4f Test POS-acc %12.4f " % (
              best_iter, best_pos_acc, test_pos_acc))
