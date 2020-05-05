import numpy as np
import scipy.misc as misc
import os
import random
import math
import pickle
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

flower_data_dir = 'D:/DataSet/flower_dataset/flower_photos/'

# if person computer
# cifar10_data_dir = 'D:/DataSet/cifar10/cifar-10-batches-py/'
# if colab
cifar10_data_dir = 'drive/My Drive/Colab Notebooks/HKBU_AI_Classs/data'

def _get_filenames_and_classes(flower_root):
    directories = []
    class_names = []
    for filename in os.listdir(flower_root):
        path = os.path.join(flower_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

class DataLoader_Flower:
    def __init__(self,
                 limit_load=None,
                 mini_side=164,
                 output_size=128,
                 dtype=np.float32,
                 test_ratio=0.3,):
        self.dtype = dtype
        self.photo_filenames, class_names = _get_filenames_and_classes(flower_data_dir)

        if limit_load:
            self.photo_filenames = self.photo_filenames[:limit_load]

        self.class_names_to_ids = dict(zip(class_names, range(len(class_names))))

        print('train set: %d' % len(self.photo_filenames))

        random.seed(0)
        random.shuffle(self.photo_filenames)

        self.data_idx = 0
        self.generate_dataset(mini_side, output_size, test_ratio=test_ratio)

    def get_class_num(self):
        return len(self.class_names_to_ids)

    def resize_image(self, image, mini_side):
        width, heigh, channal = image.shape
        if width > heigh:
            ratio = mini_side / heigh
            output_width, output_heigh = math.ceil(width * ratio), math.ceil(heigh * ratio)
        else:
            ratio = mini_side / width
            output_width, output_heigh = math.ceil(width * ratio), math.ceil(heigh * ratio)

        image = misc.imresize(image, size=[output_width, output_heigh])
        return image

    def central_crop(self, image, output_size):
        width, heigh, channal = image.shape
        width_beta = math.ceil((width - output_size) / 2)
        heigh_beta = math.ceil((heigh - output_size) / 2)
        return image[width_beta: width_beta + output_size, heigh_beta:heigh_beta + output_size, :]

    def generate_dataset(self, mini_side, output_size, test_ratio):
        self.train_images = []
        self.train_labels = []
        i = 0
        for path in self.photo_filenames:
            image = misc.imread(path)
            image = self.resize_image(image, mini_side)
            image = self.central_crop(image, output_size=output_size)
            image = image / 255.
            image = image.astype(self.dtype)
            self.train_images.append(image)

            class_name = os.path.basename(os.path.dirname(path))
            class_id = self.class_names_to_ids[class_name]
            label = np.zeros(len(self.class_names_to_ids))
            label[class_id] = 1.
            label = label.astype(self.dtype)
            self.train_labels.append(label)

            i += 1
            if (i + 1) % 100 == 0:
                print('%d finish' % i)

        num = len(self.train_images)
        self.train_images = np.array(self.train_images[:math.ceil(num*(1.-test_ratio))])
        self.train_labels = np.array(self.train_labels[:math.ceil(num*(1.-test_ratio))])
        self.test_images = np.array(self.train_images[-math.ceil(num*test_ratio):])
        self.test_labels = np.array(self.train_images[-math.ceil(num*test_ratio):])

        self.idx = np.arange(0, len(self.train_images))
        random.shuffle(self.idx)

    def get_batch(self, batch_size):
        if self.data_idx + batch_size > len(self.train_images):
            self.data_idx = 0
            random.shuffle(self.idx)

        images_batch = self.train_images[self.idx[self.data_idx: self.data_idx + batch_size]]
        labels_batch = self.train_labels[self.idx[self.data_idx: self.data_idx + batch_size]]

        self.data_idx += batch_size
        return images_batch, labels_batch

    def get_test(self, batch_size):
        test_idx = np.arange(0, len(self.test_labels))
        random.shuffle(test_idx)

        images_batch = self.test_images[test_idx[:batch_size]]
        labels_batch = self.test_labels[test_idx[:batch_size]]

        return images_batch, labels_batch

class DataLoader_Cifar10:
    def __init__(self, batch_size, use_preprocess, dtype=np.float32):
        self.dtype = dtype
        self.use_preprocess = use_preprocess
        self.mean, self.std = 0.473363, 0.25156
        self.batch_size = batch_size

        if self.use_preprocess:
            self.preprocess_load()
        else:
            self.valid_load()

    def valid_load(self):
        self.train_images = []
        self.train_labels = []
        train_path_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        for file in train_path_names:
            path = os.path.join(cifar10_data_dir, file)
            dict = unpickle(path)
            self.train_images.extend(dict[b'data'])
            self.train_labels.extend(dict[b'labels'])

        test_dict = unpickle(os.path.join(cifar10_data_dir, 'test_batch'))
        self.test_images = []
        self.test_labels = []
        self.test_images.extend(test_dict[b'data'])
        self.test_labels.extend(test_dict[b'labels'])

        self.train_images = np.array(self.train_images)
        self.train_labels = np.array(self.train_labels)
        self.test_images = np.array(self.test_images)
        self.test_labels = np.array(self.test_labels)

        self.train_images = self.train_images.reshape([-1, 3, 32, 32])
        self.test_images = self.test_images.reshape([-1, 3, 32, 32])
        self.train_images = np.transpose(self.train_images, [0, 2, 3, 1])
        self.test_images = np.transpose(self.test_images, [0, 2, 3, 1])
        self.train_images = self.train_images / 255.
        self.test_images = self.test_images / 255.

        print('train set: %d' % len(self.train_images))
        print('test set: %d' % len(self.test_images))
        self.train_num = len(self.train_images)

        self.data_idx = 0
        self.idx = np.arange(0, len(self.train_labels))
        random.shuffle(self.idx)

    def preprocess_load(self):
        train_images = []
        train_labels = []
        train_path_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        for file in train_path_names:
            path = os.path.join(cifar10_data_dir, file)
            dict = unpickle(path)
            train_images.extend(dict[b'data'])
            train_labels.extend(dict[b'labels'])

        test_dict = unpickle(os.path.join(cifar10_data_dir, 'test_batch'))
        self.test_images = []
        self.test_labels = []
        self.test_images.extend(test_dict[b'data'])
        self.test_labels.extend(test_dict[b'labels'])

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        self.test_images = np.array(self.test_images)
        self.test_labels = np.array(self.test_labels)

        train_images = train_images.reshape([-1, 3, 32, 32])
        self.test_images = self.test_images.reshape([-1, 3, 32, 32])
        train_images = np.transpose(train_images, [0, 2, 3, 1])
        self.test_images = np.transpose(self.test_images, [0, 2, 3, 1])
        self.test_images = self.test_images / 255.

        print('train set: %d' % len(train_images))
        print('test set: %d' % len(self.test_images))
        self.train_num = len(train_images)

        self.augumentation(train_images, train_labels)
        del train_images
        del train_labels

    def get_class_num(self):
        return 10

    def z_score_preprocess(self, images):
        # mean = np.mean(self.train_images)
        # std = np.std(self.train_images)
        return (images - self.mean)/self.std

    def get_batch(self):
        if self.use_preprocess:
            images_batch, labels_batch = self.loader.next()
        else:
            if self.data_idx + self.batch_size > len(self.idx):
                self.data_idx = 0
                random.shuffle(self.idx)

            images_batch = self.train_images[self.idx[self.data_idx: self.data_idx + self.batch_size]]
            labels_batch = self.train_labels[self.idx[self.data_idx: self.data_idx + self.batch_size]]

            self.data_idx += self.batch_size

        return images_batch.astype(self.dtype), labels_batch.astype(np.uint8)

    def get_test_batch(self, batch_size):
        images_batch = self.test_images[:batch_size]
        labels_batch = self.test_labels[:batch_size]

        return images_batch.astype(self.dtype), labels_batch.astype(np.uint8)

    def augumentation(self, train_images, train_labels):
        datagen = ImageDataGenerator(
            # featurewise_center=False,
            # samplewise_center=False,
            # featurewise_std_normalization=True,
            # samplewise_std_normalization=False,

            rotation_range=25,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.8, 1.2],
            # shear_range=0.0,
            zoom_range=[0.8, 1.2],
            horizontal_flip=True,
            # vertical_flip=False,
            rescale=1/255.,
        )

        self.loader = datagen.flow(x=train_images, y=train_labels, batch_size=self.batch_size, shuffle=True)

def run():
    # loader = DataLoader_Flower(limit_load=None, output_size=96, mini_side=int(96.*1.3))
    loader = DataLoader_Cifar10(batch_size=32, use_preprocess=True)
    images_batch, labels_batch = loader.get_test_batch(32)

    print(labels_batch.shape)
    print(labels_batch)
    for i in images_batch:
        # print(i)
        print(i.shape)
        i = (i - np.min(i))/(np.max(i)-np.min(i))
        plt.imshow(i)
        plt.show()

    # i = (images_batch[0] - np.min(images_batch[0])) / (np.max(images_batch[0]) - np.min(images_batch[0]))
    # plt.imshow(i)
    # plt.show()

if __name__ == '__main__':
    run()
