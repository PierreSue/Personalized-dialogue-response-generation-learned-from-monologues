import tensorflow as tf
import numpy as np
import data_utils
import collections

#from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
#from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

#linear = core_rnn_cell_impl._linear

def encode(cell, embedding, encoder_inputs, seq_len=None, dtype=tf.float32):
    with variable_scope.variable_scope("embedding_rnn_seq2seq") as scope:
        scope.set_dtype(dtype)
        
        encoder_inputs = [tf.cast(embedding_ops.embedding_lookup(embedding, i), tf.float32) for i in encoder_inputs]
        
        return tf.nn.static_rnn(
            cell,
            encoder_inputs,
            sequence_length=seq_len,
            dtype=dtype)


def decode(cell, init_state, embedding, decoder_inputs, feature_inputs, feature_proj, maxlen, feed_prev=False, loop_function=None, dtype=tf.float32):
    with variable_scope.variable_scope("embedding_rnn_decoder") as scope:
        outputs = []
        hiddens = []
        state = init_state
        
        feature_inputs = [tf.matmul(feature_inputs[0], feature_proj)]

        if not feed_prev:
            emb_inputs = [tf.cast(embedding_ops.embedding_lookup(embedding, i), tf.float32) for i in decoder_inputs]

            for i, emb_inp in enumerate(emb_inputs):
                if i >= maxlen:
                    break
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()

                emb_inp = tf.concat([emb_inp, feature_inputs[0]],1)
                output, state = cell(emb_inp, state) #add feature to embed input, duai chi!
                outputs.append(output)
                hiddens.append(state)
            return outputs, hiddens, state
        else:
            samples = []
            i = 0
            emb_inp = tf.cast(embedding_ops.embedding_lookup(embedding, decoder_inputs[0]), tf.float32)
            prev = None
            tmp = None

            index = 0
            while(True):
                index = index + 1
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                emb_inp = tf.concat([emb_inp, feature_inputs[0]],1) # concat_feature

                output, state = cell(emb_inp, state)
                outputs.append(output)
                hiddens.append(state)
                prev = output
                with tf.variable_scope('loop', reuse=True):
                    if prev is not None:
                        tmp = loop_function(prev)

                if tmp is not None:
                    if isinstance(tmp, list):
                        emb_inp, prev_symbol = tmp
                        samples.append(prev_symbol)
                    else:
                        emb_inp = tmp
                i += 1
                if i >= maxlen:
                    break
            return outputs, samples, hiddens



#FIXME
"""
def attn_decode(cell, init_state, embedding, decoder_inputs, encoder_outputs, maxlen, feed_prev=False, loop_function=None, dtype=tf.float32):
    encoder_size = tf.shape(encoder_outputs[0])[1]
    top_states = [ array_ops.reshape(e, [-1, 1, encoder_size])
                  for e in encoder_outputs if e != tf.zeros(tf.shape(e)) ]
    attention_states = tf.concat(top_states, 1)
    attn_length = attention_states.get_shape()[1].value
    if attn_length is None:
      attn_length = tf.shape(attention_states)[1]
    attn_size = tf.shape(attention_states)[2]

    # [batch, in-height, in-width, in-channel]
    attn_hidden = array_ops.reshape(attention_states,
                                    [-1, attn_length, 1, attn_size])

    size = encoder_outputs[0].shape[1]
    # filt [height, width, in-channel, out-channel]
    filt = variable_scope.get_variable("AttnW", [1, 1, size, size])
    # conv2d (tensor, filter, stride, padding)
    hidden_features = nn_ops.conv2d(attn_hidden, filt, [1, 1, 1, 1], "SAME")

    v = variable_scope.get_variable("AttnV", [size])
    # attention with query
    def attention(query):
        with variable_scope.variable_scope("Attention"):
            y = linear(query, size, True)
            y = tf.reshape(y, [-1, 1, 1, attn_size])
            s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [2, 3])
            a = nn_ops.softmax(s) # over attn_length
            # attention-weighted vector c
            d = tf.reduce_sum(
                tf.reshape(a, [-1, attn_length, 1, 1]) * attn_hidden,
                [1, 2])
            ds = array_ops.reshape(d, [-1, attn_size])
        return ds
    # decode
    outputs = []
    state = init_state
    attns = attention(state)

    if not feed_prev:
        emb_inputs = (embedding_ops.embedding_lookup(embedding, i)
                      for i in decoder_inputs)
        for i, emb_inp in enumerate(emb_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            x = linear(emb_inp + attns, size, True)
            output, state = cell(x, state)
            if i == 0:
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(),
                        reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear(output + attns, size, True)
            outputs.append(output) 
        return outputs
    else:
        samples = []
        i = 0
        emb_inp = embedding_ops.embedding_lookup(embedding, decoder_inputs[0])
        prev = None
        tmp = None
        while(True):
            with tf.variable_scope('loop', reuse=True):
                if prev is not None:
                    tmp = loop_function(prev)
            if tmp is not None:
                if isinstance(tmp, list):
                    emb_inp, prev_symbol = tmp
                    samples.append(prev_symbol)
                else:
                    emb_inp = tmp
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            x = linear(emb_inp + attns, size, True)
            output, state = cell(x, state)
            if i == 0:
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(),
                        reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear(output + attns, size, True)
            outputs.append(output)
            if i >= maxlen:
                break
            prev = output
            i += 1
        return outputs, samples
"""
