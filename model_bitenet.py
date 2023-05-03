import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import math_ops
from layers import AttentionPooling, EmbeddingSharedWeights, \
    EncoderStack, Flatten, Reshape

import sys
from utils.configs import cfg

class BiteNet(object):

    def __init__(self, dataset):

        # dataset hyperparameters
        self.n_intervals = dataset.days_size
        self.n_codes = dataset.max_len_visit
        self.vocabulary_size = len(dataset.dictionary)
        self.digit3_size = len(dataset.dictionary_3digit)

        # training hyperparameters
        self.lr = 0.0001
        self.dropout_rate = cfg.dropout

        # model hyperparameter
        self.n_visits = cfg.valid_visits
        self.embedding_size = cfg.embedding_size
        self.num_hidden_layers = cfg.num_hidden_layers
        self.train = cfg.train
        self.predict_type = cfg.predict_type
        self.model = None

        # input placeholder
        self.code_inputs = keras.layers.Input(shape=(self.n_visits, self.n_codes,), dtype=tf.int32, name='train_inputs')
        self.interval_inputs = keras.layers.Input(shape=(self.n_visits,), dtype=tf.int32, name='interval_inputs')
        self.inputs_mask = math_ops.not_equal(self.code_inputs, 0)
        visit_mask = tf.reduce_sum(tf.cast(self.code_inputs, tf.int32), -1)
        self.visit_mask = tf.cast(visit_mask, tf.bool)

        # embedding placeholder
        self.embedding = EmbeddingSharedWeights(self.vocabulary_size,
                                                                self.embedding_size,name='codes_embedding')
        self.interval_embedding = EmbeddingSharedWeights(self.n_intervals, self.embedding_size,
                                                         name='interval_embedding')
        
        
        self.params = dict()
        self.params["hidden_size"] = self.embedding_size
        self.params["filter_size"] = self.embedding_size
        self.params["dropout"] = self.dropout_rate
        self.params["allow_ffn_pad"] = False
        self.params["num_hidden_layers"] = self.num_hidden_layers
        self.params["is_scale"] = False
        self.params["direction"] = 'diag'
        self.params["num_heads"] = cfg.num_heads

    def build_network(self):

         # input/mask input and reshape from 3d to 2d
        code_inputs = Flatten(2, name='code_flatten')(self.embedding(self.code_inputs))
        mask_inputs = Flatten(1, name='mask_flatten')(self.inputs_mask)

        # Vanilla Encoder and attention pooling
        hidden = EncoderStack(self.params, self.train, self.embedding_size, name='Vanilla_encoder')((code_inputs, mask_inputs))
        attention_aft = Reshape()((AttentionPooling(self.embedding_size, name='intra_attn_pool')
                       ((hidden, mask_inputs)), self.code_inputs, self.embedding_size))

        
        # position embedding layer
        embedding_position = self.interval_embedding(self.interval_inputs)
        concat_input = keras.layers.Add()([embedding_position, attention_aft])

        # we following the convention of naming it as forward/backward
        self.params["direction"] = 'forward'
        forward_encoder = EncoderStack(self.params, self.train, self.embedding_size,
                                         name='forward_encoder')((concat_input, self.visit_mask))
        forward_attn = AttentionPooling(self.embedding_size, name='forward_attn_pool')((forward_encoder, self.visit_mask))

        self.params["direction"] = 'backward'
        backward_encoder = EncoderStack(self.params, self.train, self.embedding_size,
                                         name='backward_encoder')((concat_input, self.visit_mask))
        backward_attn = AttentionPooling(self.embedding_size, name='backward_attn_pool')((backward_encoder, self.visit_mask))

        # concatenate outputs of forward and backward
        concat_attn = keras.layers.Concatenate()([forward_attn, backward_attn])
        out1 = keras.layers.Dense(self.embedding_size, 'relu')(concat_attn)
        out = keras.layers.Dropout(self.dropout_rate)(out1)

        if self.predict_type == 'dx':
            pred = keras.layers.Dense(self.digit3_size, activation='sigmoid')(out)
            self.model = keras.Model(inputs=[self.code_inputs, self.interval_inputs],
                                     outputs=pred, name='hierarchicalSA')
            self.model.compile(optimizer=keras.optimizers.Adam(0.0001),
                               loss='binary_crossentropy',
                               metrics=['accuracy'])

        elif self.predict_type == 're':
            pred = keras.layers.Dense(1, activation='sigmoid', use_bias=True)(out)
            self.model = keras.Model(inputs=[self.code_inputs, self.interval_inputs],
                                     outputs=pred, name='BiteNet')
            self.model.compile(optimizer=keras.optimizers.Adam(0.001),
                               loss=keras.losses.BinaryCrossentropy(),
                               metrics=['accuracy'])

        # two tasks
        else:
            pred_dx = keras.layers.Dense(self.digit3_size, activation='sigmoid', name='los_dx')(out)
            pred_re = keras.layers.Dense(1, activation='sigmoid', use_bias=True, name='los_re')(out)
            self.model = keras.Model(inputs=[self.code_inputs, self.interval_inputs],
                                     outputs=[pred_dx, pred_re],
                                     name='hierarchicalSA')
            # using the same setup as original paper
            self.model.compile(keras.optimizers.RMSprop(learning_rate=0.001),
                               loss={'los_dx': 'binary_crossentropy',
                                     'los_re': 'binary_crossentropy'},
                               loss_weights={'los_dx': 1., 'los_re': 0.1},
                               metrics={'los_dx': 'accuracy', 'los_re': 'accuracy'})

        print(self.model.summary())

