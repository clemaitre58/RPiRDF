import neurtu

from skimage.io import imread
from skimage.transform import resize

from DescGlob.fourrier1 import fourrier1


def cases():
    for size in [64, 128, 256, 512]:
        tags = {'size': (size, size)}
        img = imread('lena128x128.jpg')
        img_resized = resize(img, (size, size))
        yield neurtu.delayed(fourrier1, tags=tags)(img_resized)

df = neurtu.timeit(cases())
df['wall_time'] *= 1000
print(df)
