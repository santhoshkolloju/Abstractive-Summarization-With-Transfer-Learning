import texar as tx
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


#warmup steps must be 0.1% of number of iterations
lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (hidden_dim ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 10000,
}

bos_token_id =101
eos_token_id = 102

model_dir= "./models"
run_mode= "train_and_evaluate"
batch_size = 32
test_batch_size = 32

max_train_steps = 100000

display_steps = 100
eval_steps = 100000

max_decoding_length = 400

max_seq_length_src = 512
max_seq_length_tgt = 400

train_file = "data/train.tf_reccord"
eval_file = "data/eval.tf_record"

bert_pretrain_dir="./bert_uncased_model"
