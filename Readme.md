<h3>Abstractive summarization using bert as encoder and transformer decoder</h3>

I have used a text generation library called Texar , Its a beautiful library with a lot of abstractions, i would say it to be 
scikit learn for text generation problems.

The main idea behind this architecture is to use the transfer learning from pretrained BERT a masked language model ,
I have replaced the Encoder part with BERT Encoder and the deocder is trained from the scratch.

One of the advantages of using Transfomer Networks is training is much faster than LSTM based models as elimanate sequential behaviour in Transformer models.

Transformer based models generate more gramatically correct  coherent sentences.


<h1> Code </h1>

<pre>

<h2>Download the Bert Models </h2>

mkdir bert_pretrained_models
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -P bert_pretrained_models/;
unzip bert_pretrained_models/uncased_L-12_H-768_A-12.zip -d bert_pretrained_models/



<h2>download the texar code and install all the python packages specified 
    in requirement.txt of texar_repo</h2>
import sys
!test -d texar_repo || git clone https://github.com/asyml/texar.git texar_repo
if not 'texar_repo' in sys.path:
  sys.path += ['texar_repo']

</h2>download the CNN Stories data set and unzip the file</h2>
https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ
tar -zxf cnn_stories.tgz

<h2>Import the modules</h2>
import os
import csv
import collections
import sys
from texar_repo.examples.bert.utils import data_utils, model_utils, tokenization
import importlib
import tensorflow as tf
import texar as tx 
from texar_repo.examples.bert import config_classifier as config_downstream
from texar_repo.texar.utils import transformer_utils
from texar_repo.examples.transformer.utils import data_utils, utils
from texar_repo.examples.transformer.bleu_tool import bleu_wrapper




<h2>preprocessing the cnn data</h2>
from preprocess import *
directory = 'cnn/stories/'
stories = load_stories(directory)
print('Loaded Stories %d' % len(stories))


f1 = open("stories.txt",'w')
f2 = open("summary.txt",'w')
for example in stories:
  example['story'] = clean_lines(example['story'].split('\n'))
  example['highlights'] = clean_lines(example['highlights'])
  f1.write(" ".join(example['story']))
  f1.write("\n")
  f2.write(" ".join(example['highlights']))
  f2.write("\n")
f1.close()
f2.close()
  
story = open("stories.txt").readlines()
summ = open("summary.txt").readlines() 
train_story = story[0:90000]
train_summ = summ[0:90000]

eval_story = story[90000:91579]
eval_summ = summ[90000:91579]


test_story = story[91579:92579]
test_summ = summ[91579:92579]


with open("train_story.txt",'w') as f:
  f.write("\n".join(train_story))
  
with open("train_summ.txt",'w') as f:
  f.write("\n".join(train_summ))
  
with open("eval_story.txt",'w') as f:
  f.write("\n".join(eval_story))
  
  
with open("eval_summ.txt",'w') as f:
  f.write("\n".join(eval_summ))
  
  
with open("test_story.txt",'w') as f:
  f.write("\n".join(test_story))
  
  
with open("test_summ.txt",'w') as f:
  f.write("\n".join(test_summ))  
  


<h2>Setup the configuration for bert and Decoder</h2>
#change the numbers bolcks and heads as required 
dcoder_config = {
    'dim': 768,
    'num_blocks': 6,
    'multihead_attention': {
        'num_heads': 8,
        'output_dim': 768
        # See documentation for more optional hyperparameters
    },
    'position_embedder_hparams': {
        'dim': 768
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': tx.modules.default_transformer_poswise_net_hparams(
        output_dim=768)
}

loss_label_confidence = 0.9

random_seed = 1234
beam_width = 5
alpha = 0.6
hidden_dim = 768


opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.997,
            'epsilon': 1e-9
        }
    }
}

<h3>learning rate used called linear learning rate warmup steps should be 1% of your 
     number of iteratons.learning rate 
     increases linearly till the warmup number of steps and then decreases.</h3>
lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (hidden_dim ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 2000,
}

<h3>using berts [CLS] and [SEP] token as beginning and end of sentence</h3>
bos_token_id =101
eos_token_id = 102

#models will be saved in this path
model_dir= "./models"
run_mode= "train_and_evaluate"
batch_size = 4
test_batch_size = 4

max_train_epoch = 20
display_steps = 10
<h3>Number of iiterations after which evaluation runs </h3>
eval_steps = 100000

max_decoding_length = 400

max_seq_length_src = 512
max_seq_length_tgt = 400



<h3>change the path pointing to your location</h3>
bert_pretrain_dir = 'bert_pretrained_models/uncased_L-12_H-768_A-12'


<h3>Create the TF record files for train eval and test data sets </h3>
from create_tf_records import *

bert_config = model_utils.transform_bert_to_texar_config(
            os.path.join(bert_pretrain_dir, 'bert_config.json'))


<h3>Bert Word Piece Tokenization </h3>
tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'),
        do_lower_case=True)

vocab_size = len(tokenizer.vocab)

processor = CNNDailymail()
train_dataset = get_dataset(processor,tokenizer,"./",max_seq_length_src,max_seq_length_tgt,4,'train',"./data")
eval_dataset = get_dataset(processor,tokenizer,"./",max_seq_length_src,max_seq_length_tgt,4,'eval',"./data")
test_dataset = get_dataset(processor,tokenizer,"./",max_seq_length_src,max_seq_length_tgt,4,'test',"./data")
<h3>Three files gets created under the folder data </h3>

<h2> Model Architecture </h2>
<h3>Placeholders </h3>
src_input_ids = tf.placeholder(tf.int64, shape=(None, None))
src_segment_ids = tf.placeholder(tf.int64, shape=(None, None))
tgt_input_ids = tf.placeholder(tf.int64, shape=(None, None))
tgt_segment_ids = tf.placeholder(tf.int64, shape=(None, None))

batch_size = tf.shape(src_input_ids)[0]

src_input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(src_input_ids, 0)),
                             axis=1)
tgt_input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(src_input_ids, 0)),
                             axis=1)

labels = tf.placeholder(tf.int64, shape=(None, None))
is_target = tf.to_float(tf.not_equal(labels, 0))

global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')

<h1>create the data set iterator </h1>
iterator = tx.data.FeedableDataIterator({
        'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset})

batch = iterator.get_next()


<h2>encoder Bert model </h2>
print("Intializing the Bert Encoder Graph")
with tf.variable_scope('bert'):
        embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.vocab_size,
            hparams=bert_config.embed)
        word_embeds = embedder(src_input_ids)

        #Creates segment embeddings for each type of tokens.
        segment_embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.type_vocab_size,
            hparams=bert_config.segment_embed)
        segment_embeds = segment_embedder(src_segment_ids)

        input_embeds = word_embeds + segment_embeds

        # The BERT model (a TransformerEncoder)
        encoder = tx.modules.TransformerEncoder(hparams=bert_config.encoder)
        encoder_output = encoder(input_embeds, src_input_length)
        
        # Builds layers for downstream classification, which is also initialized
        # with BERT pre-trained checkpoint.
        with tf.variable_scope("pooler"):
            # Uses the projection of the 1st-step hidden vector of BERT output
            # as the representation of the sentence
            bert_sent_hidden = tf.squeeze(encoder_output[:, 0:1, :], axis=1)
            bert_sent_output = tf.layers.dense(
                bert_sent_hidden, config_downstream.hidden_dim,
                activation=tf.tanh)
            output = tf.layers.dropout(
                bert_sent_output, rate=0.1, training=tx.global_mode_train())


<h3>Loads pretrained BERT model parameters</h3>
print("loading the bert pretrained weights")
init_checkpoint = os.path.join(bert_pretrain_dir, 'bert_model.ckpt')
model_utils.init_bert_checkpoint(init_checkpoint)


<h3>decoder part and mle losss</h3>
tgt_embedding = tf.concat(
    [tf.zeros(shape=[1, embedder.dim]), embedder.embedding[1:, :]], axis=0)

decoder = tx.modules.TransformerDecoder(embedding=tgt_embedding,
                             hparams=dcoder_config)
<h3>For training this takes as input BERT encoder final hidden states </h3>
outputs = decoder(
    memory=encoder_output,
    memory_sequence_length=src_input_length,
    inputs=embedder(tgt_input_ids),
    sequence_length=tgt_input_length,
    decoding_strategy='train_greedy',
    mode=tf.estimator.ModeKeys.TRAIN
)

mle_loss = transformer_utils.smoothing_cross_entropy(
        outputs.logits, labels, vocab_size, loss_label_confidence)
mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)

train_op = tx.core.get_train_op(
        mle_loss,
        learning_rate=learning_rate,
        global_step=global_step,
        hparams=opt)

tf.summary.scalar('lr', learning_rate)
tf.summary.scalar('mle_loss', mle_loss)
summary_merged = tf.summary.merge_all()

<h2>Code for Inference </h2>
#prediction 
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

<h3> Saver object for checkpointing </h3>
saver = tf.train.Saver(max_to_keep=5)
best_results = {'score': 0, 'epoch': -1}

<h3> Training the model </h3>
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
              <h1>checkpoints every 1000 iterations </h1>
              if step and step % 1000 == 0:
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
              <h3> Displaying the output of summary here we replace ## by empty space is because 
              bert by default uses word piece tokenization</h3>
              print("Output Summary is ")
              for s_toks,summ_toks in zip(references,hypotheses):
                story = tokenizer.convert_ids_to_tokens(s_toks)
                print("Story is "" ".join(story).replace(" ##",""))
                summ = tokenizer.convert_ids_to_tokens(summ_toks)
                print("Story is "" ".join(summ).replace(" ##",""))
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

<h3> Run Training </h3>
logging_file= "logging.txt"
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

        step = 5000
        for epoch in range(max_train_epoch):
          iterator.restart_dataset(sess, 'train')
          step = _train_epoch(sess, epoch, step, smry_writer)

    elif run_mode == 'test':
        logger.info('Begin running with test mode')

        logger.info('Restore latest checkpoint in %s' % model_dir)
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        _eval_epoch(sess, 0, mode='test')

    else:
        raise ValueError('Unknown mode: {}'.format(run_mode))






</pre>
