import os
import numpy as np
from PIL import Image

after_img = np.array(Image.open('../logs/attack_cppn_2021_11_26_14_26_44/Samples/T0007.png').resize((256,256)))
raw_img = np.array(Image.open('/home/users/wujunde/dataset/Dataset-new-2/Test-400/0499/0499.jpg').resize((256,256)))

res = (raw_img - after_img) * 6 + np.ones([256,256,3])* 125
res = Image.fromarray(res.astype(np.uint8))
res.save('./res.jpg')