# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from visdom import Visdom
import numpy as np


# in_win = vis.images(
#     np.random.randn(batch_size, 3, 256, 256),
#     opts=dict(title='real_in')
# )
# out_win = vis.images(
#     np.random.randn(batch_size, 3, 256, 256),
#     opts=dict(title='real_out')
# )
# fake_win = vis.images(
#     np.random.randn(batch_size, 3, 256, 256),
#     opts=dict(title='fake_out')
# )
viz = Visdom()

 # image demo
viz.image(
    np.random.rand(3, 512, 256),
    opts=dict(title='Random!', caption='How random.'),
)

# grid of images
updatetextwindow = viz.images(
    np.random.randn(20, 3, 256, 256),
    opts=dict(title='Random images')
)

viz.images(
    np.random.rand(20, 3, 512, 256),
    win=updatetextwindow
)