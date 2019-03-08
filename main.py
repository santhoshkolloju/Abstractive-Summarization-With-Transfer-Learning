import sys

if not 'texar_repo' in sys.path:
  sys.path += ['texar_repo']
import tensorflow as tf
import texar as tx
import numpy as np
from config import *
from model import *

import os


def _train_epoch(sess, epoch, step, smry_writer):
        
            
        fetches = {
            'step': global_step,
            'train_op': train_op,
            'smry': summary_merged,
            'loss': mle_loss,
        }

        while True:
            try:
              feed_dict = {
                iterator.handle: iterator.get_handle(sess, 'train'),
                tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
              }
              op = sess.run([batch],feed_dict)
              feed_dict = {
                   src_input_ids:op[0]['src_input_ids'],
                   src_segment_ids : op[0]['src_segment_ids'],
                   tgt_input_ids:op[0]['tgt_input_ids'],

                   labels:op[0]['tgt_labels'],
                   learning_rate: utils.get_lr(step, lr),
                   tx.global_mode(): tf.estimator.ModeKeys.TRAIN
                }


              fetches_ = sess.run(fetches, feed_dict=feed_dict)
              step, loss = fetches_['step'], fetches_['loss']
              if step and step % display_steps == 0:
                  logger.info('step: %d, loss: %.4f', step, loss)
                  print('step: %d, loss: %.4f' % (step, loss))
                  smry_writer.add_summary(fetches_['smry'], global_step=step)

              if step and step % checkpoint_steps == 0:
                  model_path = model_dir+"/model_"+str(step)+".ckpt"
                  logger.info('saving model to %s', model_path)
                  print('saving model to %s' % model_path)
                  saver.save(sess, model_path)
              if step and step % eval_steps == 0:
                  _eval_epoch(sess, epoch, mode='eval')
            except tf.errors.OutOfRangeError:
                break

        return step

def _eval_epoch(sess, epoch, mode):

        references, hypotheses = [], []
        bsize = test_batch_size
        fetches = {
                'inferred_ids': inferred_ids,
            }
        bno=0
        while True:
            
            #print("Temp",temp)
            try:
              print("Batch",bno)
              feed_dict = {
              iterator.handle: iterator.get_handle(sess, 'eval'),
              tx.global_mode(): tf.estimator.ModeKeys.EVAL,
              }
              op = sess.run([batch],feed_dict)
              feed_dict = {
                   src_input_ids:op[0]['src_input_ids'],
                   src_segment_ids : op[0]['src_segment_ids'],
                   tx.global_mode(): tf.estimator.ModeKeys.EVAL
              }
              fetches_ = sess.run(fetches, feed_dict=feed_dict)
              labels = op[0]['tgt_labels']
              hypotheses.extend(h.tolist() for h in fetches_['inferred_ids'])
              references.extend(r.tolist() for r in labels)
              hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
              references = utils.list_strip_eos(references, eos_token_id)
              bno = bno+1
              
            except tf.errors.OutOfRangeError:
                break


        if mode == 'eval':
            # Writes results to files to evaluate BLEU
            # For 'eval' mode, the BLEU is based on token ids (rather than
            # text tokens) and serves only as a surrogate metric to monitor
            # the training process
            fname = os.path.join(model_dir, 'tmp.eval')
            
            hypotheses = tx.utils.str_join(hypotheses)
            references = tx.utils.str_join(references)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                hypotheses, references, fname, mode='s')
            eval_bleu = bleu_wrapper(ref_fn, hyp_fn, case_sensitive=True)
            eval_bleu = 100. * eval_bleu
            logger.info('epoch: %d, eval_bleu %.4f', epoch, eval_bleu)
            print('epoch: %d, eval_bleu %.4f' % (epoch, eval_bleu))

            if eval_bleu > best_results['score']:
                logger.info('epoch: %d, best bleu: %.4f', epoch, eval_bleu)
                best_results['score'] = eval_bleu
                best_results['epoch'] = epoch
                model_path = os.path.join(model_dir, 'best-model.ckpt')
               
                logger.info('saving model to %s', model_path)
                print('saving model to %s' % model_path)
                saver.save(sess, model_path)


tx.utils.maybe_create_dir(model_dir)
logging_file= os.path.join(model_dir,"logging.txt")
logger = utils.get_logger(logging_file)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    smry_writer = tf.summary.FileWriter(model_dir, graph=sess.graph)

    if run_mode == 'train_and_evaluate':
        logger.info('Begin running with train_and_evaluate mode')

        if tf.train.latest_checkpoint(model_dir) is not None:
            logger.info('Restore latest checkpoint in %s' % model_dir)
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        
        iterator.initialize_dataset(sess)

        step = 0
        for epoch in range(epochs):
          iterator.restart_dataset(sess, 'train')
          step = _train_epoch(sess, epoch, step, smry_writer)

    

    else:
        raise ValueError('Unknown mode: {}'.format(run_mode))
