
# coding: utf-8

# # Very deep networks with repeating elements
# 
# As we already noticed in AlexNet, the number of layers in networks keeps on increasing. This means that it becomes extremely tedious to write code that piles on one layer after the other manually. Fortunately, programming languages have a wonderful fix for this: subroutines and loops. This way we can express networks as *code*. Just like we would use a for loop to count from 1 to 10, we'll use code to combine layers. The first network that had this structure was VGG. 

# ## VGG
# 
# We begin with the usual import ritual

# In[1]:


from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)


# In[22]:


ctx = mx.gpu(1)


# ## Load up a dataset
# 

# In[3]:


batch_size = 64

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=False, transform=transform),
                                     batch_size, shuffle=False)


# ## The VGG architecture
# 
# A key aspect of VGG was to use many convolutional blocks with relatively narrow kernels, followed by a max-pooling step and to repeat this block multiple times. What is pretty neat about the code below is that we use functions to *return* network blocks. These are then combined to larger networks (e.g. in `vgg_stack`) and this allows us to construct VGG from components. What is particularly useful here is that we can use it to reparameterize the architecture simply by changing a few lines rather than adding and removing many lines of network definitions. 

# In[14]:


from mxnet import symbol
from mxnet.gluon.nn import HybridBlock, Activation
from mxnet.base import numeric_types

def _infer_weight_shape(op_name, data_shape, kwargs):
    op = getattr(symbol, op_name)
    sym = op(symbol.var('data', shape=data_shape), **kwargs)
    return sym.infer_shape_partial()[0]

class _Conv(HybridBlock):
    """Abstract nD convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    If `use_bias` is `True`, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    Parameters
    ----------
    channels : int
        The dimensionality of the output space
        i.e. the number of output channels in the convolution.
    kernel_size : int or tuple/list of n ints
        Specifies the dimensions of the convolution window.
    strides: int or tuple/list of n ints,
        Specifies the strides of the convolution.
    padding : int or tuple/list of n ints,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation: int or tuple/list of n ints,
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two convolution
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str,
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', 'NCHW',
        'NHWC', 'NCDHW', 'NDHWC', etc. 'N', 'C', 'H', 'W', 'D' stands for
        batch, channel, height, width and depth dimensions respectively.
        Convolution is performed over 'D', 'H', and 'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer: str or `Initializer`
        Initializer for the bias vector.
    """
    def __init__(self, channels, kernel_size, strides, padding, dilation,
                 groups, layout, in_channels=0, activation=None, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros',
                 op_name='Convolution', adj=None, prefix=None, params=None):
        super(_Conv, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._channels = channels
            self._in_channels = in_channels
            if isinstance(strides, numeric_types):
                strides = (strides,)*len(kernel_size)
            if isinstance(padding, numeric_types):
                padding = (padding,)*len(kernel_size)
            if isinstance(dilation, numeric_types):
                dilation = (dilation,)*len(kernel_size)
            self._op_name = op_name
            self._kwargs = {
                'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
                'pad': padding, 'num_filter': channels, 'num_group': groups,
                'no_bias': not use_bias, 'layout': layout}
            if adj is not None:
                self._kwargs['adj'] = adj

            dshape = [0]*(len(kernel_size) + 2)
            dshape[layout.find('N')] = 1
            dshape[layout.find('C')] = in_channels
            self.wshapes = _infer_weight_shape(op_name, dshape, self._kwargs)
#             print(self.wshapes)
#             print('activation',activation)
            self.weight = self.params.get('weight', shape=self.wshapes[1],
                                          init=weight_initializer,
                                          allow_deferred_init=True)
            
            self.gamma = self.params.get('gamma', shape=self.wshapes[1],
                                          init=weight_initializer,
                                          allow_deferred_init=True)
            self.beta = self.params.get('beta', shape=self.wshapes[1],
                                          init=bias_initializer,
                                          allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=self.wshapes[2],
                                            init=bias_initializer,
                                            allow_deferred_init=True)
            else:
                self.bias = None

            if activation is not None:
                self.act = Activation(activation, prefix=activation+'_')
            else:
                self.act = None
                
    def _dropout(self, F, weight, gamma, beta, rate = -0.4):

        gamma = F.tanh(gamma) > rate
        w = weight * gamma + beta
        return w
        
    

    def hybrid_forward(self, F, x, weight, gamma, beta, bias=None):
        if bias is None:
            act = getattr(F, self._op_name)(x, self._dropout(F, weight, gamma, beta), name='fwd', **self._kwargs)
        else:
            act = getattr(F, self._op_name)(x, self._dropout(F, weight, gamma, beta), bias, name='fwd', **self._kwargs)
        if self.act is not None:
            act = self.act(act)
        return act

    def _alias(self):
        return 'conv'

    def __repr__(self):
        s = '{name}({mapping}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs['kernel'])
        if self._kwargs['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if self._kwargs['dilate'] != (1,) * len_kernel_size:
            s += ', dilation={dilate}'
        if hasattr(self, 'out_pad') and self.out_pad != (0,) * len_kernel_size:
            s += ', output_padding={out_pad}'.format(out_pad=self.out_pad)
        if self._kwargs['num_group'] != 1:
            s += ', groups={num_group}'
        if self.bias is None:
            s += ', bias=False'
        if self.act:
            s += ', {}'.format(self.act)
        s += ')'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs)
class DPConv2D(_Conv):

    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        assert layout in ('NCHW', 'NHWC'), "Only supports 'NCHW' and 'NHWC' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        super(DPConv2D, self).__init__(
            channels, kernel_size, strides, padding, dilation, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer, **kwargs)


# In[15]:


from mxnet.gluon import nn

def vgg_block(num_convs,in_channels, channels):
    out = nn.Sequential()
    for i in range(num_convs):
        if i == 0:
            inner_channels = in_channels
        else:
            in_channels = channels
            
        out.add(DPConv2D(channels=channels, kernel_size=3,
                      padding=1, activation='relu', in_channels=in_channels))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out

def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, in_channels, channels) in architecture:
        out.add(vgg_block(num_convs,in_channels, channels))
    return out

num_outputs = 10
architecture = ((1,3,64), (1,64,128), (2,128,256), (2,256,512))
net = nn.Sequential()
with net.name_scope():
    net.add(vgg_stack(architecture))
    net.add(nn.Flatten())
    net.add(nn.Dense(512, activation="relu"))
    # net.add(nn.Dropout(.5))
    net.add(nn.Dense(512, activation="relu"))
    # net.add(nn.Dropout(.5))
    net.add(nn.Dense(num_outputs))



# ## Initialize parameters

# In[17]:


net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)


# ## Optimizer

# In[18]:


trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .05})


# ## Softmax cross-entropy loss

# In[19]:


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


# ## Evaluation loop

# In[20]:


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


# ## Training loop

# In[ ]:


###########################
#  Only one epoch so tests can run quickly, increase this variable to actually run
###########################
epochs = 10
smoothing_constant = .01

for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
        
        if i > 0 and i % 200 == 0:
            print('Batch %d. Loss: %f' % (i, moving_loss))
            
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))    


# ## Next
# [Batch normalization from scratch](../chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.ipynb)

# For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
# DPConv
# Batch 200. Loss: 2.292256
# Batch 400. Loss: 1.259530
# Batch 600. Loss: 0.442372
# Batch 800. Loss: 0.217491
# Epoch 0. Loss: 0.158498515880178, Train_acc 0.962266666667, Test_acc 0.9659


# Conv
# Batch 200. Loss: 2.299246
# Batch 400. Loss: 2.223105
# Batch 600. Loss: 1.041572
# Batch 800. Loss: 0.433382
# Epoch 0. Loss: 0.2739769850900358, Train_acc 0.915633333333, Test_acc 0.9212



# Batch 200. Loss: 2.300955
# Batch 400. Loss: 2.291031
# Batch 600. Loss: 2.205365
# [Epoch 0. Loss: 2.1439804215906397, Train_acc 0.2639, Test_acc 0.2727

# Batch 200. Loss: 2.301540
# Batch 400. Loss: 2.289903
# Batch 600. Loss: 2.208009
# Epoch 0. Loss: 2.127615232444898, Train_acc 0.24312, Test_acc 0.2486



#  CONV
# Batch 200. Loss: 2.301540                                                                                                                        [211/1844]
# Batch 400. Loss: 2.289903
# Batch 600. Loss: 2.208009
# Epoch 0. Loss: 2.127615232444898, Train_acc 0.24312, Test_acc 0.2486
# Batch 200. Loss: 2.065583
# Batch 400. Loss: 2.003086
# Batch 600. Loss: 1.916065
# Epoch 1. Loss: 1.8501110877972287, Train_acc 0.15578, Test_acc 0.1546
# Batch 200. Loss: 1.783613
# Batch 400. Loss: 1.716797
# Batch 600. Loss: 1.656559
# Epoch 2. Loss: 1.6284732224685037, Train_acc 0.43144, Test_acc 0.4392
# Batch 200. Loss: 1.566576
# Batch 400. Loss: 1.531411
# Batch 600. Loss: 1.485204
# Epoch 3. Loss: 1.4781758285953546, Train_acc 0.48672, Test_acc 0.4862
# Batch 200. Loss: 1.416568
# Batch 400. Loss: 1.362378
# Batch 600. Loss: 1.327030
# Epoch 4. Loss: 1.3018849193760356, Train_acc 0.54166, Test_acc 0.5316
# Batch 200. Loss: 1.233874
# Batch 400. Loss: 1.225789
# Batch 600. Loss: 1.168954
# Epoch 5. Loss: 1.1597690263808602, Train_acc 0.5512, Test_acc 0.5304
# Batch 200. Loss: 1.102011
# Batch 400. Loss: 1.076601
# Batch 600. Loss: 1.038123
# Epoch 6. Loss: 1.0206304209069172, Train_acc 0.54238, Test_acc 0.5149
# Batch 200. Loss: 0.955668
# Batch 400. Loss: 0.955265
# Batch 600. Loss: 0.923220
# Epoch 7. Loss: 0.9004230415518, Train_acc 0.50786, Test_acc 0.4826
# Batch 200. Loss: 0.848203
# Batch 400. Loss: 0.836195
# Batch 600. Loss: 0.788386
# Epoch 8. Loss: 0.8051142071737929, Train_acc 0.67812, Test_acc 0.6316
# Batch 200. Loss: 0.725209
# Batch 400. Loss: 0.730031
# Batch 600. Loss: 0.708301
# Epoch 9. Loss: 0.6993791111181086, Train_acc 0.80828, Test_acc 0.7227