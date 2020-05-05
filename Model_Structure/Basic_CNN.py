from COMP7015_Mini_Project_1.tool.Network import *

def basic_cnn(inputs,
              num_classes=1000,
              is_training=True,
              reuse=None,
              scope='basic_cnn',
              dtype=tf.float32,
              weight_decay=0.0005,
              dropout=0.5,
              ):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'scale': True,
        'is_training': is_training,
    }

    print('using basic cnn network')
    with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=dtype),
                            biases_initializer=tf.constant_initializer(0., dtype=dtype),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            ):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d,
                                 slim.batch_norm, slim.dropout], outputs_collections='end_points'):
                with tf.variable_scope('layer1'):
                    net = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer2'):
                    net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer3'):
                    net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer4'):
                    net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
                net = slim.flatten(net)

                with tf.variable_scope('layer5'):
                    net = slim.fully_connected(inputs=net, num_outputs=64, activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                with tf.variable_scope('layer6'):
                    net = slim.fully_connected(inputs=net, num_outputs=num_classes, activation_fn=None, normalizer_fn=None)

                end_points = slim.utils.convert_collection_to_dict('end_points')

                return net, end_points

def basic_mini_cnn(inputs,
                   num_classes=1000,
                   is_training=True,
                   reuse=None,
                   scope='basic_cnn',
                   dtype=tf.float32,
                   weight_decay=0.0005,
                   dropout=0.3,
                   ):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'scale': True,
        'is_training': is_training,
    }

    print('using basic mini cnn network')
    with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=dtype),
                            biases_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=dtype),
                            weights_regularizer=slim.l2_regularizer(weight_decay)
                            ):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d, slim.batch_norm], outputs_collections='end_points'):
                with tf.variable_scope('layer1'):
                    net = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    # net = ops.batch_norm_layer(net, scope='batch_norm', is_training=is_training)
                    net = tf.nn.relu(net)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer2'):
                    net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    # net = ops.batch_norm_layer(net, scope='batch_norm', is_training=is_training)
                    net = tf.nn.relu(net)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer3'):
                    net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    # net = ops.batch_norm_layer(net, scope='batch_norm', is_training=is_training)
                    net = tf.nn.relu(net)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer4'):
                    net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    # net = ops.batch_norm_layer(net, scope='batch_norm', is_training=is_training)
                    net = tf.nn.relu(net)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
                net = slim.flatten(net)

                with tf.variable_scope('layer5'):
                    net = slim.fully_connected(inputs=net, num_outputs=64, activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    # net = ops.batch_norm_layer(net, scope='batch_norm', is_training=is_training)
                    net = tf.nn.relu(net)

                with tf.variable_scope('layer6'):
                    net = slim.fully_connected(inputs=net, num_outputs=num_classes, activation_fn=None, normalizer_fn=None)

                end_points = slim.utils.convert_collection_to_dict('end_points')

                return net, end_points

class basic_cnn_network(NetWork):
    def get_network(self):
        return basic_cnn

class mini_cnn_network(NetWork):
    def get_network(self):
        return basic_mini_cnn
