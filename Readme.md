<h3>Abstractive summarization using bert as encoder and transformer decoder</h3>

I have used a text generation library called Texar , Its a beautiful library with a lot of abstractions, i would say it to be 
scikit learn for text generation problems.

The main idea behind this architecture is to use the transfer learning from pretrained BERT a masked language model ,
I have replaced the Encoder part with BERT Encoder and the deocder is trained from the scratch.

One of the advantages of using Transfomer Networks is training is much faster than LSTM based models as elimanate sequential behaviour in Transformer models.

Transformer based models generate more gramatically correct  coherent sentences.


<h3> Code </h3>

<pre>

<h2>Download the Bert Models </h2>

mkdir bert_pretrained_models
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -P bert_pretrained_models/;
unzip bert_pretrained_models/uncased_L-12_H-768_A-12.zip -d bert_pretrained_models/



<h2>#download the texar code and install all the python packages specified 
    in requirement.txt of texar_repo</h2>
import sys
!test -d texar_repo || git clone https://github.com/asyml/texar.git texar_repo
if not 'texar_repo' in sys.path:
  sys.path += ['texar_repo']

</h2>#download the CNN Stories data set and unzip the file</h2>
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




<h2>#preprocessing the cnn data</h2>
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

<h1>#learning rate used called linear learning rate warmup steps should be 1% of your 
     number of iteratons.learning rate 
     increases linearly till the warmup number of steps and then decreases.</h1>
lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (hidden_dim ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 2000,
}

<h1>#using berts [CLS] and [SEP] token as beginning and end of sentence</h1>
bos_token_id =101
eos_token_id = 102

#models will be saved in this path
model_dir= "./models"
run_mode= "train_and_evaluate"
batch_size = 4
test_batch_size = 4

max_train_epoch = 20
display_steps = 10
<h1>Number of iiterations after which evaluation runs </h1>
eval_steps = 100000

max_decoding_length = 400

max_seq_length_src = 512
max_seq_length_tgt = 400



<h1>#change the path pointing to your location</h1>
bert_pretrain_dir = 'bert_pretrained_models/uncased_L-12_H-768_A-12'


<h3>Create the TF record files for train eval and test data sets </h3>
from create_tf_records import *

bert_config = model_utils.transform_bert_to_texar_config(
            os.path.join(bert_pretrain_dir, 'bert_config.json'))


<h1>Bert Word Piece Tokenization </h1>
tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'),
        do_lower_case=True)

vocab_size = len(tokenizer.vocab)

processor = CNNDailymail()
train_dataset = get_dataset(processor,tokenizer,"./",max_seq_length_src,max_seq_length_tgt,4,'train',"./data")
eval_dataset = get_dataset(processor,tokenizer,"./",max_seq_length_src,max_seq_length_tgt,4,'eval',"./data")
test_dataset = get_dataset(processor,tokenizer,"./",max_seq_length_src,max_seq_length_tgt,4,'test',"./data")
<h1>Three files gets created under the folder data </h1>


</pre>
