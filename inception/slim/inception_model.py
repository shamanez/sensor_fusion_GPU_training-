
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes
layers = tf.contrib.layers 
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

FLAGS = tf.app.flags.FLAGS

def inception_v3(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=6,
                 is_training=True,
                 restore_logits=True,
                 scope='wave_sense' ,reuse=True):
  

  split_size=inputs.get_shape().as_list()[0]
  converted_inputs=tf.reshape(inputs,[split_size,20,120])
  used = tf.sign(tf.reduce_max(tf.abs(converted_inputs), reduction_indices=2)) #fibd how many time step with value without zeros 
  len_seq=tf.reduce_sum(used, reduction_indices=1)
  len_seq=tf.cast(len_seq, tf.int64)
  end_points = {}
  with tf.name_scope(scope, 'inception_v3', [inputs]):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='VALID'):
        with tf.name_scope("Conv_Encorder"):
        # 299 x 299 x 3
          inputs=tf.reshape(inputs,[split_size,20,10,12,1])
         
          end_points['conv0'] = ops.conv2d(inputs, 16, [3, 3], stride=2,
                                         scope='conv0')


        #acc_conv1 = layers.convolution2d(inputs, 16, kernel_size=[1, 3,3],
            #stride=[1, 2,2], padding='VALID', activation_fn=None, data_format='NHWC', scope='acc_conv1')
        #print(acc_conv1)
        # 149 x 149 x 32
          end_points['conv1'] = ops.conv2d(end_points['conv0'], 32, [3, 3],
                                         scope='conv1')
        # 147 x 147 x 32
          end_points['conv2'] = ops.conv2d(end_points['conv1'], 64, [3, 3],
                                        padding='SAME', scope='conv2')


        # 147 x 147 x 64
        #end_points['pool1'] = ops.max_pool(end_points['conv2'], [3, 3],
        #                                   stride=2, scope='pool1')
        # 73 x 73 x 64
          end_points['conv3'] = ops.conv2d(end_points['conv2'], 64, [1, 1],
                                       scope='conv3' , padding='SAME')

          net=end_points['conv3']

         


          net = ops.avg_pool(net, [2,3], padding='VALID', scope='pool')

          end_points['average_pool1'] = net


          time_vec=tf.reshape(end_points['average_pool1'],[split_size,20,-1])

          end_points['time_vec'] = time_vec

        with tf.name_scope("seq2seq"):
      
          encoder_hidden_units=128
          encoder_cell = LSTMCell(encoder_hidden_units) 
         


          ((encoder_fw_outputs,encoder_bw_outputs),(encoder_fw_final_state,encoder_bw_final_state)) =   (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,cell_bw=encoder_cell,
                                    inputs=time_vec,
                                    sequence_length=len_seq,
                                    dtype=tf.float32, time_major=False)) 
          encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

          encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

          encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

          encoder_final_state = LSTMStateTuple(c=encoder_final_state_c,h=encoder_final_state_h)

          

        logits = layers.fully_connected(encoder_final_state.h, 6, activation_fn=None, scope='output')


      return logits , end_points
        # 73 x 73 x 80.
       # end_points['conv4'] = ops.conv2d(end_points['conv3'], 192, [3, 3],
       #                                  scope='conv4')
        # 71 x 71 x 192.
        #end_points['pool2'] = ops.max_pool(end_points['conv4'], [3, 3],
       #                                    stride=2, scope='pool2')
        # 35 x 35 x 192.
'''
        net = end_points['pool2']

        with tf.variable_scope('Flatterning_1'):
          shape = net.get_shape()
          net = ops.avg_pool(net, shape[1:3], padding='VALID', scope='pool')
          end_points['average_pool1'] = net

      

        with tf.variable_scope('LSTM'):
              gru_cell1 = tf.contrib.rnn.GRUCell(INTER_DIM)
              if train:
                gru_cell1 = tf.contrib.rnn.DropoutWrapper(gru_cell1, output_keep_prob=0.5)

              gru_cell2 = tf.contrib.rnn.GRUCell(INTER_DIM)
              if train:
                gru_cell2 = tf.contrib.rnn.DropoutWrapper(gru_cell2, output_keep_prob=0.5)

              cell = tf.contrib.rnn.MultiRNNCell([gru_cell1, gru_cell2])
              init_state = cell.zero_state(BATCH_SIZE, tf.float32)

              cell_output, final_stateTuple = tf.nn.dynamic_rnn(cell, sensor_conv_out, sequence_length=length, initial_state=init_state, time_major=False)

              sum_cell_out = tf.reduce_sum(cell_output*mask, axis=1, keep_dims=False)
              avg_cell_out = sum_cell_out/avgNum



              end_points['LSTM'] = avg_cell_out
              net=avg_cell_out
        # Final pooling and prediction
        with tf.variable_scope('logits'):
          #shape = net.get_shape()
          #net = ops.avg_pool(net, shape[1:3], padding='VALID', scope='pool')
          # 1 x 1 x 2048
          net = ops.dropout(net, dropout_keep_prob, scope='dropout')
          net = ops.flatten(net, scope='flatten')
          # 2048
          logits = ops.fc(net, num_classes, activation=None, scope='logits',
                          restore=restore_logits)
          # 1000
          end_points['logits'] = logits
          end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
      return logits, end_points


def inception_v3_parameters(weight_decay=0.00004, stddev=0.1,
                            batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
  """Yields the scope with the default parameters for inception_v3.

  Args:
    weight_decay: the weight decay for weights variables.
    stddev: standard deviation of the truncated guassian weight distribution.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Yields:
    a arg_scope with the parameters needed for inception_v3.
  """
  # Set weight_decay for weights in Conv and FC layers.
  with scopes.arg_scope([ops.conv2d, ops.fc],
                        weight_decay=weight_decay):
    # Set stddev, activation and parameters for batch_norm.
    with scopes.arg_scope([ops.conv2d],
                          stddev=stddev,
                          activation=tf.nn.relu,
                          batch_norm_params={
                              'decay': batch_norm_decay,
                              'epsilon': batch_norm_epsilon}) as arg_scope:
      yield arg_scope
'''