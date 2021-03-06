import tensorflow as tf

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 300

# ---------------------------------------------------------------------------------------
def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes."""
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])

    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images."""
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(tf.equal(image_rank, 3),
                                ['Wrong rank for tensor  %s [expected] [actual]',
                                 image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])

    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    with tf.control_dependencies(asserts):
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])

    offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width, crop_height, crop_width) for image in image_list]

def _central_crop(image_list, crop_height, crop_width):
    """Performs central crops of the given image list."""
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width, crop_height, crop_width))
    return outputs

# --------------------------------------------------------------------------------
def _smallest_size_at_least(height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`."""
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width), lambda: smallest_side / width, lambda: smallest_side / height)
    new_height = tf.to_int32(tf.rint(height * scale))
    new_width = tf.to_int32(tf.rint(width * scale))
    return new_height, new_width

def _aspect_preserving_resize(image, smallest_side):
    """Resize images preserving the original aspect ratio."""
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image

def _normail_image(image):
    mean, var = tf.nn.moments(image, axes=[2], keep_dims=True)
    image = (image - mean) / tf.sqrt(var)
    return image

# ---------------------------------------------------------------------------------
def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX,
                         use_grayscale=False,
                         dtype=tf.float16):

    resize_side = tf.random_uniform([], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)

    image = _aspect_preserving_resize(image, resize_side)
    # image = _random_crop([image], output_height, output_width)[0]
    image = _central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.cast(image, dtype=dtype)

    # if use_grayscale:
    #     image = tf.image.rgb_to_grayscale(image)
    # image = tf.image.random_flip_left_right(image)

    return image/255.

def preprocess_for_eval(image,
                        output_height,
                        output_width,
                        resize_side,
                        use_grayscale=False,
                        dtype=tf.float16):
    image = _aspect_preserving_resize(image, resize_side)
    image = _central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.cast(image, dtype=dtype)

    # if use_grayscale:
    #     image = tf.image.rgb_to_grayscale(image)

    return image/255.

# --------------------------------------------------------------------------------------
def preprocess_image(image,
                     output_height,
                     output_width,
                     is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX,
                     use_grayscale=False,
                     dtype=tf.float16):
    if is_training:
        return preprocess_for_train(image, output_height, output_width,
                                    resize_side_min, resize_side_max,
                                    use_grayscale,
                                    dtype=dtype)
    else:
        return preprocess_for_eval(image, output_height, output_width,
                                   resize_side_min, use_grayscale,
                                   dtype=dtype)