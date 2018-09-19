import neurtu

from skimage.io import imread
from skimage.transform import resize

from DescGlob import fourier1


def cases():
    for size in [64, 128, 256, 512]:
        tags = {'size': (size, size)}
        img = imread('lena128x128.jpg')
        img_resized = resize(img, (size, size))
        yield neurtu.delayed(fourier1, tags=tags)(img_resized)


def main():
    df = neurtu.timeit(cases())
    df['wall_time'] *= 1000
    print(df)


if __name__ == '__main__':
    main()
