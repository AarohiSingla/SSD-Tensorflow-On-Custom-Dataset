import os
import sys
import random

import numpy as np
import tensorflow as tf

#import xml.etree.elementtree as et
import xml.etree.ElementTree as et

#There are only two definitions of my tags, depending on my own pictures
VOC_LABELS = {
    'none': (0, 'Background'),
    'headphone': (1, 'headphone'),
    'earphone': (2, 'earphone'),

}

#A folder for pictures and labels
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

#Random seed
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 1 #samples per file


#Generate integer, floating point, and string properties
def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


#Image processing
def _process_image(directory, name):
    # Read the image file.
    filename = directory + DIRECTORY_IMAGES + name + '.png'
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = et.parse(filename)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append (label.encode ('ascii')) #ාchange to ASCII format

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')

        ####################################################################		
        bboxes.append((max(float(bbox.find('ymin').text) / shape[0],0.0),
                   max(float(bbox.find('xmin').text) / shape[1],0.0),
                   min(float(bbox.find('ymax').text) / shape[0],1.0),
                   min(float(bbox.find('xmax').text) / shape[1],1.0)
                   ))
        ######################################################################
        # the above code in ### replaced by below commented code
        # a = float(bbox.find('ymin').text) / shape[0]
        # b = float(bbox.find('xmin').text) / shape[1]
        # a1 = float(bbox.find('ymax').text) / shape[0]
        # b1 = float(bbox.find('xmax').text) / shape[1]
        # a_e = a1 - a
        # b_e = b1 - b
        # if abs(a_e) < 1 and abs(b_e) < 1:
            # bboxes.append((a, b, a1, b1))

    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


#Conversion example
def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))
    return example


#Add to tfrecord
def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


#Name is the prefix of the converted file
def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='voc_train', shuffling=False):
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted (os.listdir (path)) #(sort)
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write ('converting image%d /% d \ n '% (i + 1, len (filenames))) #(terminal printing, similar to print)
                sys.stdout.flush() #(buffer)

                filename = filenames[i]
                img_name = filename[:-4]
                print(img_name)
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    print('\nFinished converting the Pascal VOC dataset!')


#Original dataset path, output path and output file name
dataset_dir = "D:/env_conda_1.14_for_SSD/SSD_CUSTOM/VOC2007/"   #"/voc2007/"
output_dir = "D:/env_conda_1.14_for_SSD/SSD_CUSTOM/tfrecords/"
name = "voc_2007_train"


def main(_):
    run(dataset_dir, output_dir, name)


if __name__ == '__main__':
    tf.app.run()