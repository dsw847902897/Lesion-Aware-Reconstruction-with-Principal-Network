import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
from inpaint_model import InpaintCAModel
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='/home/wy/dataset/topicData/mask_dataset/random_input', type=str, help='The directory of images to be completed.')
parser.add_argument('--mask_dir', default='/home/wy/dataset/topicData/mask_dataset/random_mask', type=str, help='The directory of masks, value 255 indicates mask.')
parser.add_argument('--output_dir', default='/home/wy/dataset/topicData/mask_dataset/random_output', type=str, help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='/home/wy/py_doc/GII/generative_inpainting/model_logs/release_places2_1', type=str, help='The directory of tensorflow checkpoint.')

if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = InpaintCAModel()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    # 构建图
    input_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 6])  # 假设图像和掩码合并后的shape为[1, H, W, C]
    output = model.build_server_graph(FLAGS, input_image_placeholder)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)

    with tf.Session(config=sess_config) as sess:
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = [tf.assign(var, tf.contrib.framework.load_variable(args.checkpoint_dir, var.name)) for var in vars_list]
        sess.run(assign_ops)
        print('Model loaded.')

        for img_name in tqdm(glob(os.path.join(args.image_dir, '*.png')), desc='Processing images'):
            mask_name = os.path.join(args.mask_dir, os.path.basename(img_name).replace('input','mask'))
            image = cv2.imread(img_name)
            mask = cv2.imread(mask_name)

            assert image.shape == mask.shape

            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            
            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)

            result = sess.run(output, feed_dict={input_image_placeholder: input_image})
            
            cv2.imwrite(os.path.join(args.output_dir, os.path.basename(img_name)), result[0][:, :, ::-1])

        print('All images processed.')



