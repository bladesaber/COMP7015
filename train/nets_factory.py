import functools
import tensorflow.contrib.slim as slim
from COMP7015_Mini_Project_1.Model_Structure.Vgg_16 import vgg_arg_scope, vgg_16_network
from COMP7015_Mini_Project_1.Model_Structure.ResnetV1_50 import res_network, resnet_arg_scope
from COMP7015_Mini_Project_1.Model_Structure.InceptionV3 import inception_v3_arg_scope, inception_network
from COMP7015_Mini_Project_1.Model_Structure.Basic_CNN import basic_cnn_network, mini_cnn_network
from COMP7015_Mini_Project_1.Model_Structure.Basic_inception import basic_inception_network, mini_inception_network
from COMP7015_Mini_Project_1.Model_Structure.Basic_resnet import basic_res_network, mini_res_network
from COMP7015_Mini_Project_1.Model_Structure.Basic_resnet import basic_res_network_improved_v1
from COMP7015_Mini_Project_1.Model_Structure.Basic_inception import basic_inception_network_improved_v1

# networks_map = {'vgg_16': Vgg_16.vgg_16,
#                 'inception_v3': InceptionV3.inception_v3,
#                 'resnet_v1_50': ResnetV1_50.resnet_v1_50,}

# arg_scopes_map = {'vgg_16': Vgg_16.vgg_arg_scope,
#                   'inception_v3': InceptionV3.inception_v3_arg_scope,
#                   'resnet_v1_50': ResnetV1_50.resnet_arg_scope,}

networks_map = {'vgg_16': vgg_16_network,
                'inception_v3': inception_network,
                'resnet_v1_50': res_network,
                'cnn': basic_cnn_network,
                'inception': basic_inception_network,
                'resnet': basic_res_network,
                'mini_cnn': mini_cnn_network,
                'mini_inception': mini_inception_network,
                'mini_resnet': mini_res_network,
                'resnet_improved_v1':basic_res_network_improved_v1,
                'inception_improved_v1': basic_inception_network_improved_v1,
                }

arg_scopes_map = {'vgg_16': vgg_arg_scope,
                  'inception_v3': inception_v3_arg_scope,
                  'resnet_v1_50': resnet_arg_scope,
                  'cnn': None,
                  'inception': None,
                  'resnet': None,
                  'mini_cnn': None,
                  'mini_inception': None,
                  'mini_resnet': None,
                  'resnet_improved_v1':None,
                  'inception_improved_v1':None,
                  }

def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)

    network = networks_map[name]()
    func = network.get_network()

    @functools.wraps(func)
    def network_fn(images, **kwargs):
        if arg_scopes_map[name]:
            arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
            with slim.arg_scope(arg_scope):
                return func(images, num_classes=num_classes, is_training=is_training, **kwargs)
        else:
            return func(images, **kwargs)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn, network
