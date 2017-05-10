A Lasagne and Theano implementation of the paper [Convolutional neural network architecture for geometric matching](https://arxiv.org/abs/1703.05593) by Ignacio Rocco, Relja Arandjelović, and Josef Sivic.

Download the [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar).

(Optional) Download the [Proposal Flow dataset](http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip).

**This is a work-in-progress**.  The images below were taken from the validation set after training for 300 epochs (about 17 hours on a TITAN X).
![](images/valid_8_4.png?raw=True)
![](images/valid_96_9.png?raw=True)
![](images/valid_98_10.png?raw=True)

Similarly, the images below are from the Proposal Flow dataset:
![](images/infer_0_0.png?raw=True)
![](images/infer_0_1.png?raw=True)
![](images/infer_0_2.png?raw=True)

The thin-plate-spline step has not yet been implemented.
