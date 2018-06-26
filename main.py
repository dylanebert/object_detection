from scipy.ndimage import imread
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
import enchant
import re
sys.path.append("/home/dylan/Documents/models/research/object_detection")
sys.path.append("/home/dylan/Documents/models/research")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.chdir('/home/dylan/Documents/models/research/object_detection')

DATA_PATH = '/data/flickr30k-images-raw/uncategorized/'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
CERTAINTY = .9

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

filenames = os.listdir(DATA_PATH)

TEST_IMAGE_PATHS = filenames[:10]
IMAGE_SIZE = (12, 8)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def extract_patch(filename, image, ymin, xmin, ymax, xmax, label):
    pass

#build dict of denotation graph labels
node_dict = {}
with open('/home/dylan/Documents/denotation_graph/graph/np/node.idx', 'r') as f:
    nodes = f.readlines()
    for node in nodes:
        try:
            id = re.search('^\d+', node).group(0) #get index
            caption = re.search('(?<=\t).+', node).group(0).replace('\t', '').replace('\n', '') #get caption
        except:
            continue
        if not len(caption.split(' ')) == 1:
            continue
        node_dict[id] = caption

#build dict of image labels
img_dict = {}
with open('/home/dylan/Documents/denotation_graph/graph/np/node-cap.map', 'r') as f:
    nodes = f.readlines()
    for idx, node in enumerate(nodes):
        print('Processing {0} of {1}'.format(idx + 1, len(nodes)), end='\r')
        try:
            id = re.search('^\d+', node).group(0)
            imgs = re.search('(?<=\t).+', node).group(0).split('\t')
            for i, img in enumerate(imgs):
                imgs[i] = re.search('.+(?=#)', img).group(0)
        except:
            continue
        if id in node_dict:
            for img in imgs:
                if img not in img_dict:
                    img_dict[img] = []
                if node_dict[id] not in img_dict[img]:
                    img_dict[img].append(node_dict[id])
    print('Finished processing nodes')

for idx, img in enumerate(TEST_IMAGE_PATHS):
    print('Processing {0} of {1}'.format(idx + 1, len(TEST_IMAGE_PATHS)), end='\r')
    image = Image.open(os.path.join(DATA_PATH, img))
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference(image_np, detection_graph)
    boxes = output_dict['detection_boxes']
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']
    for i in range(boxes.shape[0]):
        if scores is None or scores[i] > CERTAINTY:
            label = category_index[classes[i]]['name']
            if label in img_dict[img]:
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                extract_patch(img, image, ymin, xmin, ymax, xmax, label)
