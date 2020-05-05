from COMP7015_Mini_Project_1.train import nets_factory
from COMP7015_Mini_Project_1.Data import Data_Loader
import tensorflow as tf
import os
import numpy as np

#if person computer
# project_dir = 'D:\GoogleDrive\Colab Notebooks\HKBU_AI_Classs\COMP7015_Mini_Project/Log'
# if colab
project_dir = 'drive/My Drive/Colab Notebooks/HKBU_AI_Classs/COMP7015_Mini_Project/Log/ckpt_dir'

num_preprocessing_threads = 4
split_name = 'train'
use_grayscale = False
num_readers = 4

def get_config():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    return config

def standard(images, shape_num):
    means, var = tf.nn.moments(images, axes=[1,2,3], keep_dims=True)
    std = tf.maximum(tf.sqrt(var), 1.0/tf.sqrt(shape_num))
    images = (images - means)/std
    return images

def run(model_name, use_preprocess, use_zcore, label_smoothing=0.05, special=''):
    dataLoader = Data_Loader.DataLoader_Cifar10(batch_size=32, use_preprocess=use_preprocess)

    images = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='images_input')
    labels = tf.placeholder(dtype=tf.uint8, shape=[None])
    is_trainning_placeholder = tf.placeholder(dtype=tf.bool, name='is_trainning_bool')

    network_fn, network = nets_factory.get_network_fn(model_name, weight_decay=0.0001,
                                             num_classes=dataLoader.get_class_num(), is_training=is_trainning_placeholder)

    if use_zcore:
        images = standard(images, shape_num=32.*32.*3.)
    logits, endpoints = network_fn(images, num_classes=dataLoader.get_class_num(),
                                   is_training=is_trainning_placeholder, dtype=tf.float32,
                                   dropout=0.5)

    target_labels = tf.one_hot(labels, depth=dataLoader.get_class_num(), dtype=tf.float32)
    target_labels = (1. - label_smoothing) * target_labels + label_smoothing / dataLoader.get_class_num()
    total_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=target_labels))

    predict = tf.argmax(tf.nn.softmax(logits, axis=1), axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(target_labels, axis=1)), tf.float32))

    # ------------------------------------------------------------------------------------------------
    variables_to_train = tf.trainable_variables()

    g_var_list = tf.global_variables()
    bn_moving_vars = [g for g in g_var_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_var_list if 'moving_variance' in g.name]
    var_to_save = variables_to_train + bn_moving_vars

    # -----------------------    evaluation
    test_accuracy_record, test_cost_record = [], []

    saver = tf.train.Saver(var_list=var_to_save)

    model_file = model_name+special
    if use_preprocess:
        model_file = model_file+'_augu'
    if use_zcore:
        model_file = model_file+'_zcore'

    test_images, test_labels = dataLoader.get_test_batch(10000)
    each_batch = 1000

    with tf.Session(config=get_config()) as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, save_path=os.path.join(project_dir, '%s.ckpt'%model_file))

        for i in range(int(10000/each_batch)):
            test_acc, test_err = sess.run(fetches=[accuracy, total_loss],
                                feed_dict={
                                    images: test_images[i*each_batch: (i+1)*each_batch],
                                    labels: test_labels[i*each_batch: (i+1)*each_batch],
                                    is_trainning_placeholder:False})

            test_accuracy_record.append(test_acc)
            test_cost_record.append(test_err)

    print('accuracy mean:', np.mean(np.array(test_accuracy_record)))
    print('cost mean:', np.mean(test_cost_record))

if __name__ == '__main__':
    pass
