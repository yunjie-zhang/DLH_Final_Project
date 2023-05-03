import tensorflow as tf
from tensorflow import keras
from functools import reduce
from operator import mul
import math

class Activation(keras.layers.Layer):
    def __init__(self, output_size, activation, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.output_size = output_size
        self.flatten = Flatten(1)
        self.leaner = keras.layers.Dense(output_size, activation=activation)
        self.reconstruct = Reconstruct(1)

    def call(self, inputs):
        output = self.reconstruct([self.leaner(self.flatten(inputs)), inputs])
        return output

class AttentionPooling(keras.layers.Layer):
    def __init__(self, embedding_size, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.linear = DenseActivation(embedding_size)
        self.dense = DenseActivation(embedding_size, activation='relu')

    def call(self, inputs):
        tensor, mask = inputs
        out1 = self.linear(self.dense(tensor))
        out2 = tf.add(out1, (1 - tf.cast(tf.expand_dims(mask, -1), tf.float32)) * (-1e30))
        soft_out = tf.nn.softmax(out2, 1)  
        attn_output = tf.reduce_sum(soft_out * tensor, 1) 
        return attn_output
    
class EmbeddingSharedWeights(keras.layers.Layer): #embedding_layer
    def __init__(self, vocab_size, hidden_size, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        super().__init__(**kwargs)

    def build(self, _):
          self.shared_weights = self.add_weight(
              name="weights", shape = (self.vocab_size, self.hidden_size),
              dtype='float32', trainable=True,
              initializer=tf.random_normal_initializer(0.0, self.hidden_size ** -0.5))

    def call(self, x):
        emb = tf.gather(self.shared_weights, x)
        emb = emb * tf.expand_dims(tf.cast(tf.not_equal(x, 0), tf.float32), -1)
        output = emb * (self.hidden_size ** 0.5)
        return output

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(dtype="float32", **kwargs)

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]
        self.scale = self.add_weight(
            shape=[self.hidden_size],
            initializer=tf.ones_initializer())
        self.bias = self.add_weight(
            shape=[self.hidden_size],
            initializer=tf.zeros_initializer())
        super().build(input_shape)

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, direction, train, dropout, num_units, num_heads=10, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.direction = direction
        self.dropout = dropout
        self.train = train
        self.num_units = num_units
        self.q_linear = keras.layers.Dense(self.num_units, use_bias=False)
        self.k_linear = keras.layers.Dense(self.num_units, use_bias=False)
        self.v_linear = keras.layers.Dense(self.num_units, use_bias=False)

    def call(self, inputs):
        input_tensor, input_mask = inputs
        queries = input_tensor
        keys = input_tensor

        Q = self.q_linear(queries)
        K = self.k_linear(keys)
        V = self.v_linear(keys)

        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        key_masks = tf.sign(tf.reduce_sum(tf.abs(K_), axis=-1))
        key_masks = tf.expand_dims(key_masks, 1)
        key_masks = tf.tile(key_masks, [1, Q_.get_shape().as_list()[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        n_visits = input_tensor.get_shape()[1]
        sw_indices = tf.range(n_visits, dtype=tf.int32)
        sw_col, sw_row = tf.meshgrid(sw_indices, sw_indices)
        
        if self.direction == 'diag':
            attention_mask = tf.cast(tf.linalg.diag(- tf.ones([n_visits], tf.int32)) + 1, tf.bool)
        elif self.direction == 'forward':
            attention_mask = tf.greater(sw_row, sw_col)
        else:
            attention_mask = tf.greater(sw_col, sw_row)
        
        adder = (1.0 - tf.cast(attention_mask, outputs.dtype)) * -10000.0
        outputs += adder

        outputs = tf.nn.softmax(outputs)

        query_masks = tf.sign(tf.reduce_sum(tf.abs(Q_), axis=-1))
        query_masks = tf.expand_dims(query_masks, -1)
        query_masks = tf.tile(query_masks, [1, 1, tf.shape(K_)[1]])

        outputs = outputs * query_masks

        if self.train:
            outputs = tf.nn.dropout(outputs, rate=self.dropout)

        outputs = tf.matmul(outputs, V_)

        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)

        val_mask = tf.expand_dims(input_mask, -1)
        outputs = tf.multiply(outputs, tf.cast(val_mask, tf.float32))

        return outputs
        
class FeedForwardNetwork(keras.layers.Layer):
  def __init__(self, hidden_size, filter_size, relu_dropout, train, allow_pad, **kwargs):
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout
    self.train = train
    self.allow_pad = allow_pad

    self.filter_dense_layer = keras.layers.Dense(
        filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
    self.output_dense_layer = keras.layers.Dense(
        hidden_size, use_bias=True, name="output_layer")
    super().__init__(**kwargs)
  def call(self, input):
    x, mask = input
    output = self.filter_dense_layer(x)
    # apply dropout 
    if self.train:
      output = tf.nn.dropout(output, self.relu_dropout)
    output = self.output_dense_layer(output)
    return output


class AddResidual(tf.keras.layers.Layer):
  def __init__(self, layer):
    self.layer = layer
    self.layer_norm = LayerNormalization()
    super().__init__()

  def call(self, input):
    x, mask = input
    r = self.layer((self.layer_norm(x), mask))
    return x + r
  
class EncoderStack(keras.layers.Layer):
    def __init__(self, params, train, embedding_size, **kwargs):
        super(EncoderStack, self).__init__(**kwargs)
        self.layers = []
        self.train = train
        self.embedding_size = embedding_size
        self.output_normalization = LayerNormalization()
        for _ in range(params["num_hidden_layers"]):
            masked_encoder_layer = MultiHeadAttention(params["direction"],
                                                               train,
                                                               params["dropout"],
                                                               self.embedding_size,
                                                               params["num_heads"],
                                                               name='masked_encoder')
            feed_forward_network = FeedForwardNetwork(params["hidden_size"],
                                                                params["filter_size"],
                                                                params["dropout"],
                                                                train, params["allow_ffn_pad"],
                                                                name='feed_forward_network')

            self.layers.append([
                          AddResidual(masked_encoder_layer),
                          AddResidual(feed_forward_network)
                      ])


    def call(self, inputs):
        encoder_inputs, input_mask = inputs
        for _, layer in enumerate(self.layers):
          masked_encoder_layer = layer[0]
          feed_forward_network = layer[1]
          encoder_inputs = masked_encoder_layer((encoder_inputs, input_mask))
          encoder_inputs = feed_forward_network((encoder_inputs, input_mask))
        return self.output_normalization(encoder_inputs)
        
class PositionEncoding(keras.layers.Layer):
    def __init__(self, hidden_size, is_encoding, **kwargs):
        self.hidden_size = hidden_size
        self.is_encoding = is_encoding
        super().__init__(dtype="float32", **kwargs)

    def call(self, position, min_timescale=1.0, max_timescale=1.0e4):
        mask = tf.cast(tf.not_equal(position, 0), tf.float32)
        position = tf.cast(position, tf.float32)
        num_timescales = self.hidden_size // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale))
                                   /(tf.cast(num_timescales, tf.float32) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, -1) * tf.expand_dims(inv_timescales, 0)
        signal *= tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=-1) * tf.expand_dims(mask, -1)
        return signal
    
class Reconstruct(keras.layers.Layer):
    def __init__(self, maintain):
        super().__init__()
        self.maintain = maintain

    def call(self, inputs):
        main_tensor, reference = inputs[0], inputs[1]
        reduced_dim = self.maintain
        ref_shape = reference.get_shape().as_list()
        tensor_shape = main_tensor.get_shape().as_list()
        ref_end = len(ref_shape) - self.maintain
        tensor_begin = len(tensor_shape) - reduced_dim
        first_shape = [ref_shape[i] or tf.shape(reference)[i] for i in range(ref_end)]
        second_shape = [tensor_shape[i] or tf.shape(main_tensor)[i] for i in range(tensor_begin, len(tensor_shape))]
        target_shape = first_shape + second_shape
        output = tf.reshape(main_tensor, target_shape)
        return output

class Flatten(keras.layers.Layer):
    def __init__(self, maintain, **kwargs):
        self.maintain = maintain
        super().__init__(**kwargs)

    def call(self, inputs):
        static_shape = inputs.get_shape().as_list()
        beginning = len(static_shape) - self.maintain
        left_part = reduce(mul, [static_shape[i] or tf.shape(inputs)[i] for i in range(beginning)])
        out_shape = [left_part] + [static_shape[i] or tf.shape(inputs)[i] for i in range(beginning, len(static_shape))]
        compressed = tf.reshape(inputs, out_shape)
        return compressed

class Reshape(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        data, ref, embed_size = inputs
        batch_sz = tf.shape(ref)[0]
        visit_num = tf.shape(ref)[1]
        output = tf.reshape(data, [batch_sz, visit_num, embed_size])
        return output

class DenseActivation(keras.layers.Layer):
    def __init__(self, out_dim, activation=None):
        super().__init__()
        self.out_dim = out_dim
        self.compress = Flatten(1)
        self.linear = keras.layers.Dense(out_dim, activation=activation)
        self.recreate = Reconstruct(1)

    def call(self, inputs):
        inp = self.compress(inputs)
        inp = self.linear(inp)
        inp = self.recreate([inp, inputs])
        return inp
