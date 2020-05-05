import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import tensorflow.contrib.slim as slim

class NetWork:
    def load_npy(self, data_path, sess, t_vars, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for var in t_vars:
            try:
                name = var.name.replace(':0','')
                sess.run(var.assign(data_dict[name]))
                print('%s load finish'%var.name)
            except ValueError:
                if not ignore_missing:
                    raise

    def save_npy(self, checkpoint_path, save_path):
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        param = {}

        for key in var_to_shape_map:
            print("tensor_name", key)
            v = reader.get_tensor(key)
            v = v.astype(np.float16)
            param[key] = v
        np.save(save_path, param)

    def get_init_fn(self, checkpoint_exclude_scopes = []):
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

        variables_to_restore = []
        for var in slim.get_model_variables():
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
            else:
                variables_to_restore.append(var)

        # init_op= slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, '%s.ckpt'%model_name), variables_to_restore)
        restorer = tf.train.Saver(variables_to_restore)
        return restorer

if __name__ == '__main__':
    # net = NetWork()
    # net.save_npy(checkpoint_path='D:/HKBU_AI_Classs/COMP7015_Mini_Project/Pretrain/vgg_16.ckpt',
    #              save_path='D:/HKBU_AI_Classs/COMP7015_Mini_Project/Pretrain/vgg_16.npy')

    # ckpt_map = np.load('D:\HKBU_AI_Classs\COMP7015_Mini_Project\Pretrain/vgg_16.npy').item()
    # for key in ckpt_map.keys():
    #     print(key, ckpt_map[key].shape, ckpt_map[key].dtype)

    pass