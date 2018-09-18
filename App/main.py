from skimage import io
import matplotlib.pyplot as plt
# import numpy as np
import DescGlob.fourrier1 as dgl


def main():
    img = io.imread('lena128x128.jpg')
    # io.imshow(img)
    # io.show()
    # print(img.shape)
    desc_fourrier1 = dgl.fourrier1(img)
    plt.plot(desc_fourrier1)
    plt.show()


if __name__ == '__main__':
    main()
