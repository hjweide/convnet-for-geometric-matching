import theano
import theano.tensor as T
from lasagne.layers import get_all_params
from lasagne.layers import get_output
from lasagne.layers.special import _meshgrid
from lasagne.updates import nesterov_momentum


def create_train_func(layers):
    Xa, Xb = T.tensor4('Xa'), T.tensor4('Xb')
    Xa_batch, Xb_batch = T.tensor4('Xa_batch'), T.tensor4('Xb_batch')

    Tp = get_output(
        layers['trans'],
        inputs={
            layers['inputa']: Xa, layers['inputb']: Xb,
        }, deterministic=False,
    )

    # transforms: ground-truth, predicted
    Tg = T.fmatrix('Tg')
    Tg_batch = T.fmatrix('Tg_batch')
    theta_gt = Tg.reshape((-1, 2, 3))
    theta_pr = Tp.reshape((-1, 2, 3))

    # grids: ground-truth, predicted
    Gg = T.dot(theta_gt, _meshgrid(20, 20))
    Gp = T.dot(theta_pr, _meshgrid(20, 20))

    train_loss = T.mean(T.sqr(Gg - Gp))

    params = get_all_params(layers['trans'], trainable=True)
    updates = nesterov_momentum(train_loss, params, 1e-3, 0.9)

    corr_func = theano.function(
        inputs=[theano.In(Xa_batch), theano.In(Xb_batch), theano.In(Tg_batch)],
        outputs=[Tp, train_loss],
        updates=updates,
        givens={
            Xa: Xa_batch, Xb: Xb_batch,  # Ia, Ib
            Tg: Tg_batch,                # transform Ia --> Ib
        }
    )

    return corr_func


def create_valid_func(layers):
    Xa, Xb = T.tensor4('Xa'), T.tensor4('Xb')
    Xa_batch, Xb_batch = T.tensor4('Xa_batch'), T.tensor4('Xb_batch')

    Tp = get_output(
        layers['trans'],
        inputs={
            layers['inputa']: Xa, layers['inputb']: Xb,
        }, deterministic=True,
    )

    # transforms: ground-truth, predicted
    Tg = T.fmatrix('Tg')
    Tg_batch = T.fmatrix('Tg_batch')
    theta_gt = Tg.reshape((-1, 2, 3))
    theta_pr = Tp.reshape((-1, 2, 3))

    num_batch = Xa.shape[0]
    # grids: ground-truth, predicted
    Gg = T.dot(theta_gt, T.tile(_meshgrid(20, 20), num_batch))
    Gp = T.dot(theta_pr,  T.tile(_meshgrid(20, 20), num_batch))

    valid_loss = T.mean(T.sum(T.sqr(Gg - Gp), axis=1))

    corr_func = theano.function(
        inputs=[theano.In(Xa_batch), theano.In(Xb_batch), theano.In(Tg_batch)],
        outputs=[Tp, valid_loss],
        givens={
            Xa: Xa_batch, Xb: Xb_batch,  # Ia, Ib
            Tg: Tg_batch,                # transform Ia --> Ib
        }
    )

    return corr_func
