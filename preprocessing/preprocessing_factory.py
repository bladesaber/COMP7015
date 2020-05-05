from COMP7015_Mini_Project_1.preprocessing import vgg_preprocessing, inception_preprocessing


def get_preprocessing(name, is_training=False, use_grayscale=False):
    preprocessing_fn_map = {
        'inception_v3': inception_preprocessing,
        'resnet_v1_50': vgg_preprocessing,
        'vgg_16': vgg_preprocessing,
    }

    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)

    def preprocessing_fn(image, output_height, output_width, **kwargs):
        return preprocessing_fn_map[name].preprocess_image(
            image,
            output_height,
            output_width,
            is_training=is_training,
            use_grayscale=use_grayscale,
            **kwargs)

    return preprocessing_fn
