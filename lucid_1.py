import numpy as np
import scipy.ndimage as nd
import tensorflow as tf
# seed = np.random.randint(0, 2**32 - 1)

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform


model = models.InceptionV1()
model.load_graphdef()

# param_f = lambda: param.image(128, batch=2)
# obj = objectives.channel("mixed4a_pre_relu", 492, batch=1) - objectives.channel("mixed4a_pre_relu", 492, batch=0)
# _ = render.render_vis(model, obj, param_f)

obj = objectives.channel("mixed3a_pre_relu", 250)
out_img = render.render_vis(model, obj, thresholds=(1, 32, 128, 256, 512, 1024))