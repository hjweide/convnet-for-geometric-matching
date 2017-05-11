import cPickle as pickle
import numpy as np
import theano_funcs
import utils
import vgg16

from lasagne.layers import set_all_param_values, get_all_param_values
from os.path import join
from tqdm import tqdm


def train_geometric_matching():
    trans_params = {
        'rotation': (0, 0),
        'offset':   (0, 0),
        'flip':     (False, False),
        'shear':    (0., 0.),
        'stretch':  (1. / 2, 2),
    }

    print('building model')
    layers = vgg16.build_model((None, 3, 227, 227))

    # file to store the learned weights
    weightsfile = join('weights', 'weights.pickle')

    # initialize the feature extraction layers
    pretrainfile = join('weights', 'vgg16.pkl')
    print('initializing feature extraction layers from %s' % (pretrainfile))
    with open(pretrainfile, 'rb') as f:
        data = pickle.load(f)
    # weights are tied, no need to initialize a and b
    set_all_param_values(layers['pool4a'], data['param values'][0:20])

    # used to initialize from learned weights
    #with open(weightsfile, 'rb') as f:
    #    param_values = pickle.load(f)
    #set_all_param_values(layers['trans'], param_values)

    mean = data['mean value']

    max_epochs = 5000
    batch_size = 16
    sample_every = 25  # visualizes network output every n epochs
    sample_dir = join('data', 'samples')

    # set this to point to the root of Pascal VOC-2011
    voc_fpath = '/media/hdd/hendrik/datasets/pascal-2011'
    train_fpaths, valid_fpaths = utils.train_val_split(voc_fpath)

    print('compiling theano functions for training')
    train_func = theano_funcs.create_train_func(layers)
    print('compiling theano functions for validation')
    valid_func = theano_funcs.create_valid_func(layers)

    try:
        for epoch in range(1, max_epochs + 1):
            print('epoch %d' % (epoch))
            train_losses = []
            num_train_idx = (len(train_fpaths) + batch_size - 1) / batch_size
            train_iter = utils.get_batch_idx(len(train_fpaths), batch_size)
            for i, idx in tqdm(train_iter, total=num_train_idx, leave=False):
                X_crop_train, X_warp_train, M_train =\
                    utils.prepare_synth_batch(train_fpaths[idx], mean,
                                              trans_params)
                M, train_loss = train_func(X_crop_train, X_warp_train, M_train)
                train_losses.append(train_loss)
                if epoch % sample_every == 0:
                    utils.plot_samples(X_crop_train, X_warp_train, M, mean,
                                       prefix=join(sample_dir, 'train_%d' % i))
            print(' train loss = %.6f' % (np.mean(train_losses)))

            valid_losses = []
            num_valid_idx = (len(valid_fpaths) + batch_size - 1) / batch_size
            valid_iter = utils.get_batch_idx(len(valid_fpaths), batch_size)
            for i, idx in tqdm(valid_iter, total=num_valid_idx, leave=False):
                X_crop_valid, X_warp_valid, M_valid =\
                    utils.prepare_synth_batch(valid_fpaths[idx], mean,
                                              trans_params)
                M, valid_loss = valid_func(X_crop_valid, X_warp_valid, M_valid)
                valid_losses.append(valid_loss)
                if epoch % sample_every == 0:
                    utils.plot_samples(X_crop_valid, X_warp_valid, M, mean,
                                       prefix=join(sample_dir, 'valid_%d' % i))
            print(' valid loss = %.6f' % (np.mean(valid_losses)))
    except KeyboardInterrupt:
        print('caught ctrl-c, stopped training')

    print('saving weights to %s' % (weightsfile))
    weights = get_all_param_values(layers['trans'])
    with open(weightsfile, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train_geometric_matching()
