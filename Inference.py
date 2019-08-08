from flask import Flask,request,render_template
import requests 
import json
from collections import OrderedDict
import os
import numpy as np
import tensorflow as tf

app =Flask(__name__)

import sys

if not 'texar_repo' in sys.path:
  sys.path += ['texar_repo']

from config import *
from model import *
from preprocess import *


start_tokens = tf.fill([tx.utils.get_batch_size(src_input_ids)],
                       bos_token_id)
predictions = decoder(
    memory=encoder_output,
    memory_sequence_length=src_input_length,
    decoding_strategy='infer_greedy',
    beam_width=beam_width,
    alpha=alpha,
    start_tokens=start_tokens,
    end_token=eos_token_id,
    max_decoding_length=400,
    mode=tf.estimator.ModeKeys.PREDICT
)
if beam_width <= 1:
    inferred_ids = predictions[0].sample_id
else:
    # Uses the best sample by beam search
    inferred_ids = predictions['sample_id'][:, :, 0]




tokenizer = tokenization.FullTokenizer(
      vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'),
      do_lower_case=True)


sess = tf.Session()
def infer_single_example(story,actual_summary,tokenizer):
      example = {"src_txt":story,
      "tgt_txt":actual_summary
      }
      features = convert_single_example(1,example,max_seq_length_src,max_seq_length_tgt,
         tokenizer)
      feed_dict = {
      src_input_ids:np.array(features.src_input_ids).reshape(-1,1),
      src_segment_ids : np.array(features.src_segment_ids).reshape(-1,1)

      }

      references, hypotheses = [], []
      fetches = {
      'inferred_ids': inferred_ids,
      }
      fetches_ = sess.run(fetches, feed_dict=feed_dict)
      labels = np.array(features.tgt_labels).reshape(-1,1)
      hypotheses.extend(h.tolist() for h in fetches_['inferred_ids'])
      references.extend(r.tolist() for r in labels)
      hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
      references = utils.list_strip_eos(references[0], eos_token_id)
      hwords = tokenizer.convert_ids_to_tokens(hypotheses[0])
      rwords = tokenizer.convert_ids_to_tokens(references[0])

      hwords = tx.utils.str_join(hwords).replace(" ##","")
      rwords = tx.utils.str_join(rwords).replace(" ##","")
      print("Original",rwords)
      print("Generated",hwords)
      return hwords

@app.route("/results",methods=["GET","POST"])
def results():
	story = request.form['story']
	summary = request.form['summary']
	hwords = infer_single_example(story,summary,tokenizer)
	return hwords


if __name__=="__main__":
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    app.run(host="0.0.0.0",port=1118,debug=False)
    


