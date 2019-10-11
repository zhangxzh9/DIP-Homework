# coding=utf-8
from PIL import Image
import numpy as np
from numpy.linalg import solve
import os,sys
import matplotlib.pyplot as plt
import matplotlib

#def scale(image,size):#size[0]:highth size[1]: width
def equalize_hist(image):
    I_array = np.array(image)
    #np.savetxt("image.txt",I_array)
    src_height , src_width = I_array.shape[0:2]
    print(src_height,src_width)
    #calculate the histogram
    hist_arry = np.zeros(256,np.uint8)
    print (hist_arry)
    for h in range(src_height):
        for w in range(src_width):
            hist_arry[I_array[h,w]] = hist_arry[I_array[h,w]] + 1
    print (hist_arry)
    hist_arry = hist_arry
    plt.hist(hist_arry, bins=256, density=1, facecolor="black", edgecolor="black", alpha=0.7)

    plt.xlabel("gray")
    plt.ylabel("freq")
    plt.title("histogram")
    plt.show()

    # #creat new picture array
    # scaled_array = np.zeros((tar_height,tar_width), np.uint8)
    # #print(scaled_array.shape)
    # for h in range(tar_height):
    #     for w in range(tar_width):
    #         #float source localation
    #         x = (w+0.5) * width_scale - 0.5
    #         y = (h+0.5) * height_scale - 0.5

    #         #int source localation
    #         x_0 = int(np.floor(x))
    #         y_0 = int(np.floor(y))
    #         x_1 = min(x_0 + 1, src_width -1)
    #         y_1 = min(y_0 + 1, src_height -1)

    #         if x_0 == x_1 :
    #             scaled_array[h,w] = int((y_1 - y) * I_array[y_0,x_0] + (y - y_0) * I_array[y_1,x_0])
    #         elif y_0 == y_1 :
    #             scaled_array[h,w] = int((x_1 - x) * I_array[y_0, x_0] + (x - x_0) * I_array[y_0, x_1])
    #         else:
    #             f_up = (x_1 - x) * I_array[y_1, x_0] + (x - x_0) * I_array[y_1, x_1]
    #             f_down = (x_1 - x) * I_array[y_0, x_0] + (x - x_0) * I_array[y_0, x_1]
    #             scaled_array[h,w] = int((y_1 - y) * f_down + (y - y_0) * f_up)
    #np.savetxt("scaled.txt",scaled_array)
    return Image.fromarray(I_array)

def quantize(image,level):
    level = level if (level >= 1 and level <= 256) else ( 1 if level < 1 else 256)
    quan_k = (level - 1) / 255
    level_dict = {}
    for n in range(level):
        level_dict[n] = round( n * (255 / (level -1)))

    I_array = np.array(image)
    src_height , src_width = I_array.shape[0:2]

    quan_array = np.zeros((src_height,src_width), np.uint8)

    for h in range(src_height):
        for w in range(src_width):
            quan_array[h,w] = level_dict[round(quan_k * I_array[h,w])]
    return Image.fromarray(quan_array)






if __name__ == "__main__":
    I = Image.open('38.png')
    #I.show()
    # Image_scaled = scale(I,(192,128))
    # Image_scaled.save('./scaled_192_168.png')
    # Image_scaled = scale(I,(96,64))
    # Image_scaled.save('./scaled_96_64.png')
    # Image_scaled = scale(I,(48,32))
    # Image_scaled.save('./scaled_48_32.png')
    # Image_scaled = scale(I,(24,16))
    # Image_scaled.save('./scaled_24_16.png')
    # Image_scaled = scale(I,(300,200))
    # Image_scaled.save('./scaled_300_200.png')
    # Image_scaled = scale(I,(450,300))
    # Image_scaled.save('./scaled_450_300.png')
    # Image_scaled = scale(I,(500,200))
    # Image_scaled.save('./scaled_500_200.png')
    Image_scaled = equalize_hist(I)
    Image_scaled.save('./quantize_128.png')

