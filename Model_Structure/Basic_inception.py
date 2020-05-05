from COMP7015_Mini_Project_1.tool.Network import *
import math

def depth(depth, max_depth=None, mini_depth=16):
    if max_depth:
        return int(min(max(depth, mini_depth), max_depth))
    else:
        return int(max(depth, mini_depth))

def inception_block(net, scope, output_depth, is_training):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'scale': True,
        'is_training': is_training,
    }
    # ratio = output_depth/96.
    with tf.variable_scope(scope):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 32, [3, 3], scope='Conv2d_0a_3x3', activation_fn=None, padding='SAME', stride=1)
            branch_0 = slim.batch_norm(branch_0, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
            branch_0 = tf.nn.relu(branch_0)

        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 32, [3, 3], scope='Conv2d_0a_3x3',activation_fn=None, padding='SAME', stride=1)
            branch_1 = slim.batch_norm(branch_1, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
            branch_1 = tf.nn.relu(branch_1)

            branch_1 = slim.conv2d(branch_1, 32, [3, 3], scope='Conv2d_0b_3x3',activation_fn=None, padding='SAME', stride=1)
            branch_1 = slim.batch_norm(branch_1, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
            branch_1 = tf.nn.relu(branch_1)

        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [3, 3], scope='Conv2d_0a_3x3', activation_fn=None, padding='SAME', stride=1)
            branch_2 = slim.batch_norm(branch_2, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
            branch_2 = tf.nn.relu(branch_2)

            branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3', activation_fn=None, padding='SAME',stride=1)
            branch_2 = slim.batch_norm(branch_2, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
            branch_2 = tf.nn.relu(branch_2)

            branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0c_3x3', activation_fn=None, padding='SAME', stride=1)
            branch_2 = slim.batch_norm(branch_2, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
            branch_2 = tf.nn.relu(branch_2)

        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
    return net

def inception_block_mini(net, scope, output_depth, is_training):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'scale': True,
        'is_training': is_training,
    }
    with tf.variable_scope(scope):
        with tf.variable_scope('Branch_0'):
            # 这里命名由于疏忽导致错误
            branch_0 = slim.conv2d(net, 32, [3, 3], scope='Conv2d_0a_1x1',activation_fn=None, padding='SAME', stride=1)
            branch_0 = slim.batch_norm(branch_0, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
            branch_0 = tf.nn.relu(branch_0)

        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 32, [3, 3], scope='Conv2d_0a_3x3', activation_fn=None, padding='SAME', stride=1)
            branch_1 = slim.batch_norm(branch_1, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
            branch_1 = tf.nn.relu(branch_1)

            branch_1 = slim.conv2d(branch_1, 32, [3, 3], scope='Conv2d_0b_3x3', activation_fn=None, padding='SAME',stride=1)
            branch_1 = slim.batch_norm(branch_1, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
            branch_1 = tf.nn.relu(branch_1)

        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_3x3', activation_fn=None, padding='SAME', stride=1)
            branch_2 = slim.batch_norm(branch_2, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
            branch_2 = tf.nn.relu(branch_2)

        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
    return net

def basic_inception(inputs,
                    num_classes=1000,
                    is_training=True,
                    reuse=None,
                    scope='basic_inception',
                    dtype=tf.float32,
                    weight_decay=0.0005,
                    dropout=0.5,
                    ):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'scale': True,
        'is_training':is_training,
    }

    print('using basic inception network')
    with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=dtype),
                            biases_initializer=tf.constant_initializer(0., dtype=dtype),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            ):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d, slim.batch_norm], outputs_collections='end_points'):
                with tf.variable_scope('layer1'):
                    net = inception_block_mini(inputs, scope='block1', output_depth=32, is_training=is_training)
                    net = inception_block_mini(net, scope='block2', output_depth=32, is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer2'):
                    net = inception_block_mini(net, scope='block1', output_depth=64, is_training=is_training)
                    net = inception_block_mini(net, scope='block2', output_depth=64, is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer3'):
                    net = inception_block_mini(net, scope='block1', output_depth=128, is_training=is_training)
                    net = inception_block_mini(net, scope='block2', output_depth=128, is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer4'):
                    net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[3, 3], stride=1, padding='SAME',activation_fn=None)
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

def basic_inception_improved_v1(inputs,
                    num_classes=1000,
                    is_training=True,
                    reuse=None,
                    scope='basic_inception',
                    dtype=tf.float32,
                    weight_decay=0.0005,
                    dropout=0.5,
                    ):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'scale': True,
        'is_training':is_training,
    }

    print('using basic inception improve v1 network')
    with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=dtype),
                            biases_initializer=tf.constant_initializer(0., dtype=dtype),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            ):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d, slim.batch_norm], outputs_collections='end_points'):
                with tf.variable_scope('layer1'):
                    net = inception_block(inputs, scope='block1', output_depth=32, is_training=is_training)
                    net = inception_block(net, scope='block2', output_depth=32, is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer2'):
                    net = inception_block(net, scope='block1', output_depth=64, is_training=is_training)
                    net = inception_block(net, scope='block2', output_depth=64, is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer3'):
                    net = inception_block(net, scope='block1', output_depth=128, is_training=is_training)
                    net = inception_block(net, scope='block2', output_depth=128, is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer4'):
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

def basic_inception_mini(inputs,
                         num_classes=1000,
                         is_training=True,
                         reuse=None,
                         scope='basic_inception',
                         dtype=tf.float32,
                         weight_decay=0.0005,
                         dropout=0.5,
                         ):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'scale': True,
        'is_training':is_training,
    }

    print('using basic mini inception network')
    with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=dtype),
                            biases_initializer=tf.constant_initializer(0., dtype=dtype),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            ):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d, slim.batch_norm], outputs_collections='end_points'):
                
                net = inception_block_mini(inputs, scope='layer1', output_depth=32, is_training=is_training)
                net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                net = inception_block_mini(net, scope='layer2', output_depth=64, is_training=is_training)
                net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                net = inception_block_mini(net, scope='layer3', output_depth=128, is_training=is_training)
                net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)
                
                with tf.variable_scope('layer4'):
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

class basic_inception_network(NetWork):
    def get_network(self):
        return basic_inception

class mini_inception_network(NetWork):
    def get_network(self):
        return basic_inception_mini

class basic_inception_network_improved_v1(NetWork):
    def get_network(self):
        return basic_inception_improved_v1
