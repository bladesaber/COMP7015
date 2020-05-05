from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import numpy as np
from scipy import stats

class Ckpt_reader:
    def __init__(self, checkpoint_path):
        self.reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        self.var_to_shape_map = self.reader.get_variable_to_shape_map()

    def keys(self):
        return self.var_to_shape_map.keys()

    def get_tensor_from_ckpt(self, tensor_name):
        return self.reader.get_tensor(tensor_name)

def get_tensor_by_graph(graph, tensor_name):
    return graph.get_tensor_by_name(tensor_name)

def adjust_ckpt(checkpoint_path):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)

def paras_count(t_vars):
    return np.sum([np.prod(v.get_shape().as_list()) for v in t_vars])

def t_test(data1, data2):
    _, levene_p = stats.levene(data1, data2)

    equal_var = False
    if levene_p>0.05:
        equal_var = True

    if levene_p>0.05:
        _, p_value = stats.ttest_ind(data1, data2, equal_var=True)
    else:
        _, p_value = stats.ttest_ind(data1, data2, equal_var=False)

    equal_mean = False
    if p_value>0.05:
        equal_mean = True

    return (p_value, equal_mean), (levene_p, equal_var)

def norm_test(data):
    _, p = stats.normaltest(data)
    is_norm = False
    if p>0.05:
        is_norm = True
    return (p, is_norm)

def anove_test(data1, data2):
    (_, p_value) = stats.f_oneway(*[data1, data2])
    equal_mean = False
    if p_value>0.05:
        equal_mean = True
    return (p_value, equal_mean)

if __name__ == '__main__':
    reader = Ckpt_reader(checkpoint_path='D:/pretrain/inception_v3.ckpt')
    for key in reader.keys():
        print(key)
