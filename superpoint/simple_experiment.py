import logging
import os
# from superpoint.models import get_model
import numpy as np
from tensorflow.python.ops.numpy_ops.np_math_ops import true_divide
import yaml
import matplotlib.pyplot as plt
from superpoint.models_V2.utils import detector_head, detector_loss, box_nms
from keras import backend as K
from tensorflow.keras import Input

# from superpoint.datasets_V2 import get_dataset
import superpoint.datasets_V2.synthetic_shapes as get_dataset
# from superpoint.models_V2 import get_model
import superpoint.models_V2.magic_point_simple as MagicPoint
# from superpoint.utils.stdout_capturing import capture_outputs
from superpoint.settings import EXPER_PATH

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
import tensorflow as tf 
from models_V2.backbones.vgg import vgg_backbone, detector_head


def grad(magic_point, model, data, **config):
  with tf.GradientTape() as tape:
    loss_value = loss(magic_point, model, data, **config)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def loss(magic_point, model, data, **config):
    y_ = {}
    # y_["logits"] = model["logits"](data["image"])
    if config['data_format'] == 'channels_first':
        image = tf.transpose(data["image"], [0, 3, 1, 2])
    else:
        image = data["image"]
    tmp = model(image)
    y_["logits"] = tmp[0]
    y_["prob"] = tmp[1]
    l = magic_point._loss(y_, data, **config)
    # scope = "loss_scope"
    # l += tf.reduce_sum(tf.Graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope))
    return l

def local_get_model(image, **config):
    # if config['data_format'] == 'channels_first':
    #     image = tf.transpose(image, [0, 3, 1, 2])
    
    model = vgg_backbone(image, **config)
    model = detector_head(image, model, **config)
    # model = model["logits"]
    
    return model

def main(config):
    magic_point = MagicPoint.MagicPointSimple()
    
    batch_size = 32
    dataset = get_dataset.SyntheticShapes(**config['data'])
    dataset = dataset.get_tf_datasets()["training"]
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    
    config = magic_point.default_config
    config['training'] = True
    config['data_format'] = 'channels_first'
    # image = K.placeholder(shape=(batch_size, None, None, 1))
    # image = Input(shape=(1,None,None), name='input',)
    image = Input(shape=(1,120,160), name='input',)
    model = local_get_model(image, **config)
    # model = magic_point._model(next(iter(dataset)), **config) # should this be uncommetned
    #dataset.batch(32).element_spec,
    
    

    # data = next(iter(dataset))
    # for key in data.keys():
    #     data[key] = tf.expand_dims(data[key], axis=0)
    # # imgplot = plt.imshow(tmp["image"])
    # # plt.show()
    # # imgplot = plt.imshow(tmp["keypoint_map"])
    # # plt.show()
    # # imgplot = plt.imshow(tmp["valid_mask"])
    # # plt.show()
    # #a = model(tf.expand_dims(tmp["image"], axis=0))
    # a = data["image"]
    # # predictions = model(a)
    # predictions2 = {}
    # predictions2["logits"] = model(a)
    # l = magic_point._loss(predictions2, data, **config)
    # print("Loss test: {}".format(l))
    # # predictions = model(tmp["image"])
    # # descriptor_loss()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # loss_value, grads = grad(magic_point, model, data, **config)

    # print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
    #                                         loss_value.numpy()))

    # optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
    #                                         loss(magic_point, model, data, **config).numpy()))
    ## Note: Rerunning this cell uses the same model variables

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    

    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches of 32
        for step, data in enumerate(dataset):
            # print(data["keypoints"].shape)
            # data = dataset.batch(64)
            # pad dimention should remove with batch
            # for key in data.keys():
            #     data[key] = tf.expand_dims(data[key], axis=0)
            # Optimize the model
            loss_value, grads = grad(magic_point, model, data, **config)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            # epoch_accuracy.update_state(data['valid_mask'], model(data["image"], training=True))
            if config['data_format'] == 'channels_first':
                image = tf.transpose(data["image"], [0, 3, 1, 2])
            else:
                image = data["image"]
            tmp = magic_point._metrics(model(image), data, **config)
            # epoch_accuracy.update_state(tmp["precision"])
            
            if(step>1000):
                break

        if(epoch%50 == 0):
            out = model(image)
            fig2, axs = plt.subplots(nrows=2, ncols=2) # two axes on figure
            axs[0,0].imshow(image[0,0])
            axs[1,0].imshow(out[1][0])
            axs[1,1].imshow(data['keypoint_map'][0])
            axs[0,1].imshow(out[2][0])
            plt.show()
        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        # train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}".format(epoch,
                                                                        epoch_loss_avg.result()))
                                                                        #epoch_accuracy.result())
            print(f"precision:{tmp['precision'].numpy()}\trecall:{tmp['recall'].numpy()}")

if __name__ == "__main__":
    with open("configs/magic-point_shapes.yaml", 'r') as f:
        config = yaml.load(f)
    main(config)