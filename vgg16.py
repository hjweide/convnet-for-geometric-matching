# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

import theano
import theano.tensor as T
from lasagne.layers import batch_norm
from lasagne.layers import get_all_layers
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MergeLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import linear, rectify


class CorrelationLayer(MergeLayer):

    def __init__(self, incomings, **kwargs):
        super(CorrelationLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        Xa, Xb = inputs
        output, _ = theano.scan(fn=self.correlation,
                                outputs_info=None,
                                sequences=[Xa, Xb],
                                non_sequences=None)
        return output

    def get_output_shape_for(self, input_shapes):
        a_shape, _ = input_shapes  # both are same size
        output_shape = (
            a_shape[0], a_shape[2] * a_shape[3], a_shape[2], a_shape[3])
        return output_shape

    def correlation(self, A, B):
        Af = A.reshape((A.shape[0], A.shape[1] * A.shape[2]))
        Bf = B.reshape((B.shape[0], B.shape[1] * B.shape[2]))
        C = T.tensordot(Af.T, Bf, axes=1)

        return C.reshape((-1, A.shape[1], A.shape[2]))


def build_model(input_shape):
    net = {}
    net['inputa'] = InputLayer(input_shape)
    net['conv1_1a'] = ConvLayer(
        net['inputa'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2a'] = ConvLayer(
        net['conv1_1a'], 64, 3, pad=1, flip_filters=False)
    net['pool1a'] = PoolLayer(net['conv1_2a'], 2)
    net['conv2_1a'] = ConvLayer(
        net['pool1a'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2a'] = ConvLayer(
        net['conv2_1a'], 128, 3, pad=1, flip_filters=False)
    net['pool2a'] = PoolLayer(net['conv2_2a'], 2)
    net['conv3_1a'] = ConvLayer(
        net['pool2a'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2a'] = ConvLayer(
        net['conv3_1a'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3a'] = ConvLayer(
        net['conv3_2a'], 256, 3, pad=1, flip_filters=False)
    net['pool3a'] = PoolLayer(net['conv3_3a'], 2)
    net['conv4_1a'] = ConvLayer(
        net['pool3a'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2a'] = ConvLayer(
        net['conv4_1a'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3a'] = ConvLayer(
        net['conv4_2a'], 512, 3, pad=1, flip_filters=False)
    net['pool4a'] = PoolLayer(net['conv4_3a'], 2)
    net['norma'] = ExpressionLayer(
        net['pool4a'],
        lambda X: X / T.sqrt(T.sum(T.square(X), axis=1, keepdims=True)))

    net['inputb'] = InputLayer(input_shape)
    net['conv1_1b'] = ConvLayer(
        net['inputb'], 64, 3, pad=1, flip_filters=False,
        W=net['conv1_1a'].W, b=net['conv1_1a'].b)
    net['conv1_2b'] = ConvLayer(
        net['conv1_1b'], 64, 3, pad=1, flip_filters=False,
        W=net['conv1_2a'].W, b=net['conv1_2a'].b)
    net['pool1b'] = PoolLayer(net['conv1_2b'], 2)
    net['conv2_1b'] = ConvLayer(
        net['pool1b'], 128, 3, pad=1, flip_filters=False,
        W=net['conv2_1a'].W, b=net['conv2_1a'].b)
    net['conv2_2b'] = ConvLayer(
        net['conv2_1b'], 128, 3, pad=1, flip_filters=False,
        W=net['conv2_2a'].W, b=net['conv2_2a'].b)
    net['pool2b'] = PoolLayer(net['conv2_2b'], 2)
    net['conv3_1b'] = ConvLayer(
        net['pool2b'], 256, 3, pad=1, flip_filters=False,
        W=net['conv3_1a'].W, b=net['conv3_1a'].b)
    net['conv3_2b'] = ConvLayer(
        net['conv3_1b'], 256, 3, pad=1, flip_filters=False,
        W=net['conv3_2a'].W, b=net['conv3_2a'].b)
    net['conv3_3b'] = ConvLayer(
        net['conv3_2b'], 256, 3, pad=1, flip_filters=False,
        W=net['conv3_3a'].W, b=net['conv3_3a'].b)
    net['pool3b'] = PoolLayer(net['conv3_3b'], 2)
    net['conv4_1b'] = ConvLayer(
        net['pool3b'], 512, 3, pad=1, flip_filters=False,
        W=net['conv4_1a'].W, b=net['conv4_1a'].b)
    net['conv4_2b'] = ConvLayer(
        net['conv4_1b'], 512, 3, pad=1, flip_filters=False,
        W=net['conv4_2a'].W, b=net['conv4_2a'].b)
    net['conv4_3b'] = ConvLayer(
        net['conv4_2b'], 512, 3, pad=1, flip_filters=False,
        W=net['conv4_3a'].W, b=net['conv4_3a'].b)
    net['pool4b'] = PoolLayer(net['conv4_3b'], 2)
    net['normb'] = ExpressionLayer(
        net['pool4b'],
        lambda X: X / T.sqrt(T.sum(T.square(X), axis=1, keepdims=True)))

    net['corr'] = CorrelationLayer((net['norma'], net['normb']))
    # Need to do normalization at the output of the CorrLayer
    net['relu'] = NonlinearityLayer(
        net['corr'], name='relu', nonlinearity=rectify)
    net['norm'] = ExpressionLayer(
        net['relu'],
        lambda X: X / T.sqrt(T.sum(T.square(X), axis=1, keepdims=True)))

    net['conv1'] = batch_norm(ConvLayer(
        net['norm'], 128, 7, pad=1, flip_filters=False))
    net['conv2'] = batch_norm(ConvLayer(
        net['conv1'], 64, 5, pad=1, flip_filters=False))
    net['trans'] = DenseLayer(net['conv2'], 6, nonlinearity=linear)

    # only wish to train the matching/regression layers
    for layer in get_all_layers(net['corr']):
        for param in layer.params:
            layer.params[param].discard('trainable')

    return net


def create_corr_func():
    import numpy as np
    Xa, Xb = T.tensor4('Xa'), T.tensor4('Xb')

    def correlation(A, B):
        Ap, Bp = A.reshape((-1, 15 * 15)), B.reshape((-1, 15 * 15))
        C = T.tensordot(Ap.T, Bp, axes=1).reshape((-1, 15, 15))
        return C

    result, updates = theano.scan(fn=correlation,
                                  outputs_info=None,
                                  sequences=[Xa, Xb],
                                  non_sequences=None)
    corr_func = theano.function(
        inputs=[Xa, Xb],
        outputs=result,
    )

    X = np.random.random((32, 128, 15, 15)).astype(np.float32)
    Y = np.random.random(X.shape).astype(np.float32)

    output = corr_func(X, Y)
    print output.shape


if __name__ == '__main__':
    #create_corr_func()
    import cPickle as pickle
    import numpy as np
    from os.path import join
    from lasagne.layers import get_output, set_all_param_values
    X = np.random.random((1, 3, 227, 227)).astype(np.float32)
    layers = build_model((None, 3, 227, 227))
    pretrainfile = join('weights', 'vgg16.pkl')
    with open(pretrainfile, 'rb') as f:
        data = pickle.load(f)
    set_all_param_values(layers['pool4a'], data['param values'][0:20])
    Xa, Xb = get_output(
        [layers['norma'], layers['normb']],
        inputs={layers['inputa']: X, layers['inputb']: X},
        deterministic=False
    )

    assert np.allclose(Xa.eval(), Xb.eval()), 'outputs must match'
