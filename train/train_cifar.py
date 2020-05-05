from COMP7015_Mini_Project_1.train import nets_factory
from COMP7015_Mini_Project_1.Data import Data_Loader
import time
import tensorflow as tf
import math
import numpy as np
from COMP7015_Mini_Project_1.tool import utils
import os

#if person computer
# project_dir = 'D:\GoogleDrive\Colab Notebooks\HKBU_AI_Classs\COMP7015_Mini_Project/Log'
# if colab
project_dir = 'drive/My Drive/Colab Notebooks/HKBU_AI_Classs/COMP7015_Mini_Project/Log'

num_preprocessing_threads = 4
split_name = 'train'
labels_offset = 0
weight_decay = 0.0001
use_grayscale = False

num_readers = 4
batch_size = 64

# without augument
# mini_learn_rate = 0.000001
# with augument
mini_learn_rate = 1e-7

presist_period = 20

def get_config():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    return config

def move_avg(record, data):
    if len(record)==0:
        record.append(data)
    else:
        value = record[-1]*0.98 + data*0.02
        record.append(value)
    return record[-1]

def standard(images, shape_num):
    means, var = tf.nn.moments(images, axes=[1,2,3], keep_dims=True)
    std = tf.maximum(tf.sqrt(var), 1.0/tf.sqrt(shape_num))
    images = (images - means)/std
    return images

def run(model_name, epoch, use_preprocess, use_zcore, is_save, is_restore, label_smoothing=0.05, special=''):
    dataLoader = Data_Loader.DataLoader_Cifar10(batch_size=batch_size, use_preprocess=use_preprocess)

    images = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='images_input')
    labels = tf.placeholder(dtype=tf.uint8, shape=[None])
    is_trainning_placeholder = tf.placeholder(dtype=tf.bool, name='is_trainning_bool')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learn_rate')

    network_fn, network = nets_factory.get_network_fn(model_name, weight_decay=weight_decay,
                                             num_classes=dataLoader.get_class_num(), is_training=is_trainning_placeholder)

    if use_zcore:
        images = standard(images, shape_num=32.*32.*3.)
    logits, endpoints = network_fn(images, num_classes=dataLoader.get_class_num(),
                                   is_training=is_trainning_placeholder, dtype=tf.float32,
                                   dropout=0.5)

    # -------------------------------------------  summaries trigger
    # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # for end_point in endpoints:
    #     x = endpoints[end_point]
    #     summaries.add(tf.summary.histogram(end_point+'_distribution', x))
    #     summaries.add(tf.summary.scalar(end_point+'_sparsity', tf.nn.zero_fraction(x)))
    # ----------------------------------------------------------------------------------------

    target_labels = tf.one_hot(labels, depth=dataLoader.get_class_num(), dtype=tf.float32)
    target_labels = (1.-label_smoothing)*target_labels + label_smoothing/dataLoader.get_class_num()

    predict = tf.argmax(tf.nn.softmax(logits, axis=1), axis=1)

    logits_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=target_labels))
    # logits_loss = tf.reduce_sum(tf.abs(tf.nn.softmax(logits, axis=1) - target_labels))
    l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) / batch_size
    total_loss = l2_loss + logits_loss

    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(target_labels, axis=1)), tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # ------------------------------------------------------------------------------------------------
    variables_to_train = tf.trainable_variables()

    g_var_list = tf.global_variables()
    bn_moving_vars = [g for g in g_var_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_var_list if 'moving_variance' in g.name]
    var_to_save = variables_to_train + bn_moving_vars

    # -------------------------------------------  summaries trigger
    # for variable in slim.get_model_variables():
    #     summaries.add(tf.summary.histogram(variable.op.name, variable))
    # summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # ---------------------------------------------------------------------------
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gradients = optimizer.compute_gradients(loss=total_loss, var_list=variables_to_train)

        # gradient clip
        # for i, (g, v) in enumerate(gradients):
        #     if g is not None:
        #         gradients[i] = (tf.clip_by_norm(g, 5.), v)
        #         # gradients[i] = (tf.clip_by_value(g, 0.001, 5.0), v)

        grad_updates = optimizer.apply_gradients(gradients)

        # claculate_grads = tf.gradients(total_loss, variables_to_train)

    # -----------------------    evaluation
    train_accuracy_record, time_record, train_cost_record = [], [], []
    test_accuracy_record, test_cost_record = [], []
    learn_rate_record = []

    batch_num_per_epoch = math.ceil(dataLoader.train_num/batch_size)
    print('train step: %d'%(epoch*batch_num_per_epoch))

    if is_save or is_restore:
        saver = tf.train.Saver(var_list=var_to_save)

    learn_rate = 0.001
    test_images, test_labels = dataLoader.get_test_batch(2048)

    # ------------------------ paras
    print('model count: ',utils.paras_count(variables_to_train))
    # mini cnn: 406602
    # mini inception: 386538
    # mini resnet: 450058
    # cnn: 1503466 不使用avg_pool
    # cnn_com: 1191402 使用avg_pool
    # resnet: 1235754
    # resnet adjust: 1235754 with bn in all resnet block shortcut
    # resnet e: 1235306 use e resnet structure
    # resnet improve v1 reduce: 1278698 change layers in order to reduce paras
    # inception: 1181034
    # inception improve v1:1181034

    model_file = model_name+special
    if use_preprocess:
        model_file = model_file+'_augu'
    if use_zcore:
        model_file = model_file+'_zcore'

    period = 0

    with tf.Session(config=get_config()) as sess:
        sess.run(tf.global_variables_initializer())

        # --------------------------------------------------------------------------------------------------------------
        # train_writer = tf.summary.FileWriter('D:/HKBU_AI_Classs/COMP7015_Mini_Project/PersonNetwork/Log', sess.graph)
        # images_batch, labels_batch = dataLoader.get_batch(batch_size)
        #
        # for k in range(4):
        #     _, summury_operation, err = sess.run(fetches=[grad_updates, summary_op, total_loss],
        #                                     feed_dict={
        #                                         images: images_batch,
        #                                         labels: labels_batch
        #                                     })
        #     train_writer.add_summary(summury_operation, k)
        #
        #     # grads, logits_err, l2_err, endpoints_dict, predict_value, grads_compute = sess.run(
        #     #     fetches=[claculate_grads, logits_loss, l2_loss, endpoints, predict, gradients],
        #     #     feed_dict={
        #     #         images: images_batch,
        #     #         labels: labels_batch
        #     #     })
        #     #
        #     # for m in range(len(grads_compute)):
        #     #     for n in range(len(grads_compute[m])):
        #     #         value = grads_compute[m][n]
        #     #         name = gradients[m][n].name
        #     #         print(name, ' shape:', value.shape, ' mean:', np.mean(value),
        #     #               ' max:', np.max(value), ' min:', np.min(value))
        #
        #     # print('logits loss:', logits_err, ' l2 loss:', l2_err, ' predice:', predict_value)
        #     # for i in range(len(grads)):
        #     #     g = grads[i]
        #     #     print('index: ', claculate_grads[i].name, ' shape:', g.shape,
        #     #           ' grad mean:', np.mean(g), ' grad std:', np.std(g),
        #     #           ' grad max:', np.max(g), ' grad min:', np.min(g))
        #
        #     # for i in endpoints_dict.keys():
        #     #     print(i, ' shape:', endpoints_dict[i].shape, ' max:', np.max(endpoints_dict[i]), ' min:',
        #     #           np.min(endpoints_dict[i]),
        #     #           ' mean:', np.mean(endpoints_dict[i]))
        #
        #     print('---------------------------------------')
        # --------------------------------------------------------------------------------------------------------------

        if is_restore:
            saver.restore(sess, save_path=os.path.join(project_dir, '%s.ckpt'%model_file))

        t = time.time()
        for i in range(epoch*batch_num_per_epoch):

            images_batch, labels_batch = dataLoader.get_batch()
            _, acc, err, predict_value = sess.run(fetches=[grad_updates, accuracy, total_loss, predict],
                     feed_dict={
                         images: images_batch,
                         labels: labels_batch,
                         is_trainning_placeholder:True,
                         learning_rate: learn_rate
                     })
            cost_time = time.time()-t
            print('eposch: %d'%i, ' accuracy: ',acc,' loss: ', err, ' time: ', cost_time)

            last_acc = move_avg(train_accuracy_record, acc)
            last_err = move_avg(train_cost_record, err)
            time_record.append(cost_time)

            if last_acc>0.95:
                print('train acc has completed')
                break

            if (i+1)%20==0:
                test_acc, test_err = sess.run(fetches=[accuracy, total_loss],
                                    feed_dict={
                                        images: test_images,
                                        labels: test_labels,
                                        is_trainning_placeholder:False})
                learn_rate_record.append(learn_rate)

                last_test_acc = move_avg(test_accuracy_record, test_acc)
                last_test_err = move_avg(test_cost_record, test_err)

                period += 1
                if len(test_cost_record)>1:
                    if test_cost_record[-2]<test_cost_record[-1] and test_acc>0.5 and period>presist_period:
                        if learn_rate > mini_learn_rate:
                            learn_rate = learn_rate * 0.1
                            period = 0
                        else:
                            print('mini learn rate has complete')
                            break

                    print('test accuracy: ', test_acc, 'test cost: ', test_err,
                            ' cost decrease:', (test_cost_record[-2] - test_cost_record[-1]),
                            ' learn rate:', learn_rate)
                else:
                    test_cost_record.append(test_err)

                # if learn_rate < mini_learn_rate:
                #     print('break')
                #     break

            if (i+1)%1000==0 and is_save:
                saver.save(sess, save_path=os.path.join(project_dir+'/ckpt_dir', '%s.ckpt'%model_file), write_state=False, write_meta_graph=False)
        if is_save:
            saver.save(sess, save_path=os.path.join(project_dir+'/ckpt_dir', '%s.ckpt'%model_file), write_state=False, write_meta_graph=False)

    save_dict = {
        'train_acc':np.array(train_accuracy_record),
        'time': np.array(time_record),
        'train_cost': np.array(train_cost_record),
        'test_acc': np.array(test_accuracy_record),
        'test_cost': np.array(test_cost_record),
        'learn_rate':np.array(learn_rate_record),
    }
    np.save(os.path.join(project_dir+'/npy_dir', '%s.npy'%model_file), save_dict)

if __name__ == '__main__':
    run('mini_cnn', epoch=5, use_preprocess=True, use_zcore=True, is_save=True, is_restore=False)
    # run('mini_inception', epoch=5, use_preprocess=True, use_zcore=False)
    # run('mini_resnet', epoch=4, use_preprocess=True, use_zcore=True)
