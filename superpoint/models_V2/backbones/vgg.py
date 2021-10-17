import tensorflow as tf
from tensorflow.keras import layers as tfl
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda
from tensorflow.python.ops.gen_random_ops import TruncatedNormal
from models_V2.utils import box_nms, spatial_nms


def vgg_block(filters, kernel_size, name, data_format, training=False,
              batch_normalization=True, kernel_reg=0., **params):
    # with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    #     input_shape = inputs.shape
    #     x = tf.random.normal(input_shape)
    #     x = tfl.Conv2D(filters=filters, kernel_size=kernel_size, name='conv',
    #                    kernel_regularizer=tf.keras.regularizers.l2(0.5 * (kernel_reg)),
    #                    data_format=data_format, **params)(x)
    #     if batch_normalization:
    #         x = tfl.BatchNormalization()(x) 
    #         #training=training, name='bn', fused=True,
    #         #        axis=1 if data_format == 'channels_first' else -1)(x)
    x = tfl.Conv2D(filters=filters, kernel_size=kernel_size, name=name,
                   kernel_regularizer=tf.keras.regularizers.l2(0.5 * (kernel_reg)),
                   data_format=data_format, **params)
    # if batch_normalization:
    #     x = tfl.BatchNormalization()(x) 
    return x


def vgg_backbone(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    params_pool = {'padding': 'SAME', 'data_format': config['data_format']}

    # input_shape=(1,120,160,1)
    # input_shape=(64,1,None,None)
    # x = tf.random.normal(input_shape)
    x = vgg_block(64, 3, 'conv1_1', **params_conv)(inputs)
    x = vgg_block(64, 3, 'conv1_2', **params_conv)(x)
    x = tfl.MaxPool2D(2, 2, name='pool1', **params_pool)(x)

    x = vgg_block(64, 3, 'conv2_1', **params_conv)(x)
    x = vgg_block(64, 3, 'conv2_2', **params_conv)(x)
    x = tfl.MaxPool2D(2, 2, name='pool2', **params_pool)(x)

    x = vgg_block(128, 3, 'conv3_1', **params_conv)(x)
    x = vgg_block(128, 3, 'conv3_2', **params_conv)(x)
    x = tfl.MaxPool2D(2, 2, name='pool3', **params_pool)(x)

    x = vgg_block(128, 3, 'conv4_1', **params_conv)(x)
    x = vgg_block(128, 3, 'conv4_2', **params_conv)(x)
    #// Create an arbitrary graph of layers, by connecting them
    # // via the apply() method.
    # const input = tf.input({shape: [784]});
    # const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(input);
    # const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
    # const model = tf.model({inputs: input, outputs: dense2});

    return x


def detector_head(inputs, input, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    # x = tf.keras.Sequential([
    #     input,
    #     vgg_block(256, 3, 'conv1',
    #                   activation=tf.nn.relu, **params_conv),
    #     vgg_block(1+pow(config['grid_size'], 2), 1, 'conv2',
    #                   activation=None, **params_conv),
    # ])
    x = vgg_block(256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)(input)
    x = vgg_block(1+pow(config['grid_size'], 2), 1, 'conv2',
                      activation=None, **params_conv)(x)

    prob = tfl.Softmax(axis=cindex)(x)
    # Strip the extra “no interest point” dustbin
    prob = prob[:, :-1, :, :] if cfirst else prob[:, :, :, :-1]
    prob = tf.nn.depth_to_space(
            input=prob, block_size=config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')
    prob = tf.squeeze(prob, axis=cindex)

    # prob.add(Lamda())
  

    config['nms'] = 4
    if config['nms']:
        # prob_nms = tf.map_fn(fn = lambda p: box_nms(p, config['nms'],
        #                                     min_prob=config['detection_threshold'],
        #                                     keep_top_k=config['top_k']), elems = prob, fn_output_signature=tf.int32)
        # prob_nms_layer = Lambda(lambda p: box_nms(p, config['nms'],
        #                                 min_prob=config['detection_threshold'],
        #                                 keep_top_k=config['top_k']))
        prob_nms_layer = Lambda(lambda p: spatial_nms(p, config['nms'])) # simpler because couldn't get box_nms working                        
        prob_nms = prob_nms_layer(prob)
    pred = tf.cast(tf.greater_equal(prob_nms, config['detection_threshold']), dtype=tf.int32)

    model = Model(inputs=inputs, outputs=[x, prob, pred])

    return model