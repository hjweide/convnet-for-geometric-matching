import cPickle as pickle
import theano_funcs
import utils
import vgg16

from lasagne.layers import set_all_param_values
from tqdm import tqdm
from os.path import join


def warp_images():
    print('building model')
    layers = vgg16.build_model((None, 3, 227, 227))

    batch_size = 32
    infer_dir = join('data', 'inference')
    weightsfile = join('weights', 'weights.pickle')
    with open(weightsfile, 'rb') as f:
        param_values = pickle.load(f)
    set_all_param_values(layers['trans'], param_values)

    pretrainfile = join('weights', 'vgg16.pkl')
    with open(pretrainfile, 'rb') as f:
        data = pickle.load(f)

    mean = data['mean value']

    image_fpaths = [
        ('Cars_013b.png', 'Cars_009b.png'),
        ('060_0071.png', '060_0000.png'),
        ('246_0052.png', '246_0042.png')
    ]

    print('compiling theano functions for inference')
    num_infer_idx = (len(image_fpaths) + batch_size - 1) / batch_size
    infer_func = theano_funcs.create_infer_func(layers)
    infer_iter = utils.get_batch_idx(len(image_fpaths), batch_size)

    for i, idx in tqdm(infer_iter, total=num_infer_idx, leave=False):
        Xa, Xb = utils.prepare_batch(image_fpaths[idx], mean)
        M = infer_func(Xa, Xb)
        utils.plot_samples(Xa, Xb, M, mean,
                           prefix=join(infer_dir, 'infer_%d' % i))


if __name__ == '__main__':
    warp_images()
