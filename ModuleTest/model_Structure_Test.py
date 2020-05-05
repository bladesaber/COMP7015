import tensorflow as tf
from COMP7015_Mini_Project_1.train import nets_factory

labels_offset = 0
weight_decay = 0.00004

learning_rate = 0.01
num_epochs_per_decay = 2.0
learning_rate_decay_type = 'exponential'
learning_rate_decay_factor = 0.94
end_learning_rate = 0.0001

batch_size = 32

# ----------------------------------------------------------------------------------------------------------------------
def model_Structure_Test(model_name, num_classes):

    is_trainning_placeholder = tf.placeholder(dtype=tf.bool, name='is_trainning_bool')

    network_fn, network = nets_factory.get_network_fn(model_name, weight_decay=weight_decay,
                                                      num_classes=num_classes,
                                                      is_training=is_trainning_placeholder)

    images = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='inputs')
    labels = tf.placeholder(dtype=tf.uint8, shape=[None], name='labels')

    # logits, endpoints = network_fn(images,
    #                                num_classes=num_classes,
    #                                is_training=is_trainning_placeholder,
    #                                dtype=tf.float32)

    logits, endpoints = network_fn(images)

    # Print endpoint
    # for key in end_points:
    #     print(key, end_points[key])

    # target_labels = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
    # logits_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=target_labels))
    # l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) / batch_size
    # total_loss = l2_loss + logits_loss

    # write the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("D:\GoogleDrive\Colab Notebooks\HKBU_AI_Classs\COMP7015_Mini_Project\Log\TensorBoard", sess.graph)

    # tensorboard --logdir="log"

if __name__ == '__main__':
    model_Structure_Test('mini_cnn', num_classes=10)
