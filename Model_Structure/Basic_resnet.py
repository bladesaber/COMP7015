import tensorflow.contrib.slim as slim
from COMP7015_Mini_Project_1.tool.Network import *

def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

def res_block_e(net, depth, stride, scope, is_training):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'scale': True,
        'is_training': is_training,
    }

    with tf.variable_scope(scope, 'bottleneck_v1', [net]) as sc:
        depth_in = slim.utils.last_dimension(net.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(net, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(net, depth, [1, 1], stride=stride, activation_fn=None, scope='shortcut')
            shortcut = slim.batch_norm(shortcut, scale=True, is_training=is_training, decay=batch_norm_params['decay'])

        with tf.variable_scope('layer1'):
            residual = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
            residual = tf.nn.relu(residual)
            residual = slim.conv2d(residual, depth, [3, 3], stride=1, activation_fn=None, padding='SAME')

        with tf.variable_scope('layer2'):
            residua2 = slim.batch_norm(residual, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
            residua2 = tf.nn.relu(residua2)
            residua2 = slim.conv2d(residua2, depth, [3, 3], stride=1, activation_fn=None, padding='SAME')

        output = tf.nn.relu(shortcut + residua2)

        return output

def res_block_mini_e(net, depth, stride, scope, is_training):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'scale': True,
        'is_training': is_training,
    }

    with tf.variable_scope(scope, 'bottleneck_v1', [net]) as sc:
        depth_in = slim.utils.last_dimension(net.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(net, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(net, depth, [1, 1], stride=stride, activation_fn=None, scope='shortcut')
            shortcut = slim.batch_norm(shortcut, scale=True, is_training=is_training, decay=batch_norm_params['decay'])

        residual = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
        residual = tf.nn.relu(residual)
        residual = slim.conv2d(residual, depth, [3, 3], stride=1, scope='conv1', activation_fn=None, padding='SAME')

        output = tf.nn.relu(shortcut + residual)

        return output

def res_block_mini(net, depth, stride, scope, is_training):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'scale': True,
        'is_training': is_training,
    }

    with tf.variable_scope(scope, 'bottleneck_v1', [net]) as sc:
        depth_in = slim.utils.last_dimension(net.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(net, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(net, depth, [1, 1], stride=stride, activation_fn=None, scope='shortcut')
            shortcut = slim.batch_norm(shortcut, scale=True, is_training=is_training, decay=batch_norm_params['decay'])

        residual = slim.conv2d(net, depth, [3, 3], stride=1, scope='conv1', activation_fn=None, padding='SAME')
        residual = slim.batch_norm(residual, scale=True, is_training=is_training, decay=batch_norm_params['decay'])

        output = tf.nn.relu(shortcut + residual)

        return output

def res_block_mini_adjust(net, depth, stride, scope, is_training):
    batch_norm_params = {
        'decay': 0.95,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'scale': True,
        'is_training': is_training,
    }

    with tf.variable_scope(scope, 'bottleneck_v1', [net]) as sc:
        depth_in = slim.utils.last_dimension(net.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(net, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(net, depth, [1, 1], stride=stride, activation_fn=None, scope='shortcut')
        shortcut = slim.batch_norm(shortcut, scale=True, is_training=is_training, decay=batch_norm_params['decay'])

        residual = slim.conv2d(net, depth, [3, 3], stride=1, scope='conv1', activation_fn=None, padding='SAME')
        residual = slim.batch_norm(residual, scale=True, is_training=is_training, decay=batch_norm_params['decay'])

        output = tf.nn.relu(shortcut + residual)

        return output

def basic_resnet(inputs,
                 num_classes=1000,
                 is_training=True,
                 reuse=None,
                 scope='basic_resnet',
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

    print('using basic res network')
    with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=dtype),
                            biases_initializer=tf.constant_initializer(0., dtype=dtype),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            ):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d, slim.batch_norm], outputs_collections='end_points'):
                with tf.variable_scope('layer1'):
                    net = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=[3, 3], stride=1, padding='SAME',activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer2'):
                    net = res_block_mini_e(net, depth=64, stride=1, scope='resblock1', is_training=is_training)
                    net = res_block_mini_e(net, depth=64, stride=1, scope='resblock2', is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer3'):
                    net = res_block_mini_e(net, depth=128, stride=1, scope='resblock1', is_training=is_training)
                    net = res_block_mini_e(net, depth=128, stride=1, scope='resblock2', is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer4'):
                    net = res_block_mini_e(net, depth=256, stride=1, scope='resblock1', is_training=is_training)
                    net = res_block_mini_e(net, depth=256, stride=1, scope='resblock2', is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
                net = slim.flatten(net)

                with tf.variable_scope('layer6'):
                    net = slim.fully_connected(inputs=net, num_outputs=64, activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                with tf.variable_scope('layer7'):
                    net = slim.fully_connected(inputs=net, num_outputs=num_classes, activation_fn=None, normalizer_fn=None)

                end_points = slim.utils.convert_collection_to_dict('end_points')

                return net, end_points

def basic_resnet_mini(inputs,
                      num_classes=1000,
                      is_training=True,
                      reuse=None,
                      scope='basic_resnet',
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

    print('using basic res network')
    with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=dtype),
                            biases_initializer=tf.constant_initializer(0., dtype=dtype),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            ):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d, slim.batch_norm], outputs_collections='end_points'):
                with tf.variable_scope('layer1'):
                    net = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer2'):
                    net = res_block_mini(net, depth=64, stride=1, scope='resnet1', is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer3'):
                    net = res_block_mini(net, depth=128, stride=1, scope='resnet2', is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer4'):
                    net = res_block_mini(net, depth=256, stride=1, scope='resnet3', is_training=is_training)
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

def basic_resnet_improve_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 reuse=None,
                 scope='basic_resnet',
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

    print('using basic res network')
    with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=dtype),
                            biases_initializer=tf.constant_initializer(0., dtype=dtype),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            ):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d, slim.batch_norm], outputs_collections='end_points'):
                with tf.variable_scope('layer1'):
                    net = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=[3, 3], stride=1, padding='SAME',activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer2'):
                    net = res_block_mini_e(net, depth=48, stride=1, scope='resblock1', is_training=is_training)
                    net = res_block_mini_e(net, depth=64, stride=1, scope='resblock2', is_training=is_training)
                    net = res_block_mini_e(net, depth=80, stride=1, scope='resblock3', is_training=is_training)
                    net = res_block_mini_e(net, depth=96, stride=1, scope='resblock4', is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer3'):
                    net = res_block_mini_e(net, depth=112, stride=1, scope='resblock1', is_training=is_training)
                    net = res_block_mini_e(net, depth=128, stride=1, scope='resblock2', is_training=is_training)
                    net = res_block_mini_e(net, depth=160, stride=1, scope='resblock3', is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer4'):
                    net = res_block_mini_e(net, depth=176, stride=1, scope='resblock1', is_training=is_training)
                    net = res_block_mini_e(net, depth=192, stride=1, scope='resblock2', is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
                net = slim.flatten(net)

                with tf.variable_scope('layer6'):
                    net = slim.fully_connected(inputs=net, num_outputs=64, activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                with tf.variable_scope('layer7'):
                    net = slim.fully_connected(inputs=net, num_outputs=num_classes, activation_fn=None, normalizer_fn=None)

                end_points = slim.utils.convert_collection_to_dict('end_points')

                return net, end_points

def basic_resnet_improve_v2(inputs,
                 num_classes=1000,
                 is_training=True,
                 reuse=None,
                 scope='basic_resnet',
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

    print('using basic res network')
    with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=dtype),
                            biases_initializer=tf.constant_initializer(0., dtype=dtype),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            ):
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d, slim.batch_norm], outputs_collections='end_points'):
                with tf.variable_scope('layer1'):
                    net = slim.conv2d(inputs=inputs, num_outputs=32, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=[3, 3], stride=1, padding='SAME',activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer2'):
                    net = res_block_e(net, depth=48, stride=1, scope='resblock1', is_training=is_training)
                    net = res_block_e(net, depth=64, stride=1, scope='resblock2', is_training=is_training)
                    net = res_block_e(net, depth=80, stride=1, scope='resblock3', is_training=is_training)
                    net = res_block_e(net, depth=96, stride=1, scope='resblock4', is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer3'):
                    net = res_block_e(net, depth=112, stride=1, scope='resblock1', is_training=is_training)
                    net = res_block_e(net, depth=128, stride=1, scope='resblock2', is_training=is_training)
                    net = res_block_e(net, depth=160, stride=1, scope='resblock3', is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                with tf.variable_scope('layer4'):
                    net = res_block_e(net, depth=176, stride=1, scope='resblock1', is_training=is_training)
                    net = res_block_e(net, depth=192, stride=1, scope='resblock2', is_training=is_training)
                    net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

                net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
                net = slim.flatten(net)

                with tf.variable_scope('layer6'):
                    net = slim.fully_connected(inputs=net, num_outputs=64, activation_fn=None)
                    net = slim.batch_norm(net, scale=True, is_training=is_training, decay=batch_norm_params['decay'])
                    net = tf.nn.relu(net)

                with tf.variable_scope('layer7'):
                    net = slim.fully_connected(inputs=net, num_outputs=num_classes, activation_fn=None, normalizer_fn=None)

                end_points = slim.utils.convert_collection_to_dict('end_points')

                return net, end_points

class basic_res_network(NetWork):
    def get_network(self):
        return basic_resnet

class mini_res_network(NetWork):
    def get_network(self):
        return basic_resnet_mini

class basic_res_network_improved_v1(NetWork):
    def get_network(self):
        return basic_resnet_improve_v1
