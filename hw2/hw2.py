# coding=utf-8
from PIL import Image
import numpy as np
from numpy.linalg import solve
import os,sys
import matplotlib.pyplot as plt
import matplotlib
import math
import cv2

#def scale(image,size):#size[0]:highth size[1]: width
def equalize_hist(image):
    I_array = np.array(image)
    #np.savetxt("image.txt",I_array)
    src_height , src_width = I_array.shape[0:2]

    #calculate the scr picture histogram
    hist_arry = np.zeros(256,np.uint32)
    for h in range(src_height):
        for w in range(src_width):
            hist_arry[I_array[h,w]] += 1
    print(hist_arry[0])
    pdf_arry = np.zeros(256,np.float)
    pdf_arry = hist_arry/float(I_array.size)

    #plot the hist picture
    X = np.arange(256)
    plt.bar(X,pdf_arry,1,color="black")
    plt.xlabel("gray")
    plt.ylabel("freq")
    plt.title("histogram")
    plt.show()

    #calculate the scr picture cdf
    cdf_arry = np.zeros(256,np.float)
    cdf_arry[0] = pdf_arry[0]
    for i in range(1,256):
        cdf_arry[i] = cdf_arry[i-1] + pdf_arry[i]

    #plot the hist picture
    plt.bar(X,cdf_arry,1,color="black")
    plt.xlabel("gray")
    plt.ylabel("p(X<x)")
    plt.title("cdf")
    plt.show()

    #creat new picture by equalizing its hist (first time)
    scaled_array_1 = np.zeros((src_height,src_width), np.uint8)
    for h in range(src_height):
        for w in range(src_width):
            scaled_array_1[h,w] = np.uint8(round(255*cdf_arry[I_array[h,w]]))
    
    #calculate the picture's hist by first time hist equalizing
    hist_arry_1 = np.zeros(256,np.uint32)
    for h in range(src_height):
        for w in range(src_width):
            hist_arry_1[scaled_array_1[h,w]] += 1

    pdf_arry_1 = np.zeros(256,np.float)
    pdf_arry_1 = hist_arry_1/float(I_array.size)

    #plot the hist picture
    plt.bar(X,pdf_arry_1,1,color="black")
    plt.xlabel("gray")
    plt.ylabel("freq")
    plt.title("histogram")
    plt.show()

    return Image.fromarray(scaled_array_1)

def filter2d(image,filter,ptype):
    image_array = np.array(image)

    new_arr = filter.reshape(filter.size)
    new_arr = new_arr[::-1]
    filter = new_arr.reshape(filter.shape)

    pad = (filter.shape[0]-1) // 2
    I_array = cv2.copyMakeBorder(image_array, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    print(I_array.shape)
    y, x = I_array.shape
    y_out = y - filter.shape[0] + 1
    x_out  = x - filter.shape[0] + 1
    new_image = np.zeros((y_out, x_out),np.int32)
    if ptype == 0 :
        for i in range(y_out):
            for j in range(x_out):
                new_image[i][j] = np.sum(I_array[i:i+filter.shape[0], j:j  +filter.shape[0]]*filter)
        return Image.fromarray(new_image.astype(np.uint8))
    elif ptype == 1:
        for i in range(y_out):
            for j in range(x_out):
                new_image[i][j] = np.sum(I_array[i:i+filter.shape[0], j:j+filter.shape[0]]*filter)
                new_image[i][j] = image_array[i][j] - new_image[i][j]
                new_image[i][j] = 0 if new_image[i][j] < 0 else ( 255 if new_image[i][j] > 255 else new_image[i][j])
        return Image.fromarray(new_image.astype(np.uint8))
    else:
        mask = np.zeros((y_out, x_out),np.int32)
        smooth_image = np.zeros((y_out, x_out),np.int32)
        for i in range(y_out):
            for j in range(x_out):
                smooth_image[i][j] = np.sum(I_array[i:i+filter.shape[0], j:j+filter.shape[0]]*filter)
                
                mask[i][j] = image_array[i][j] - smooth_image[i][j]
                new_image[i][j] = image_array[i][j] + 2*mask[i][j]
                new_image[i][j] = 0 if new_image[i][j] < 0 else ( 255 if new_image[i][j] > 255 else new_image[i][j])
        return Image.fromarray(new_image.astype(np.uint8))







if __name__ == "__main__":
    I = Image.open('38.png')
    Image_equal_1 = equalize_hist(I)
    Image_equal_1.save('./38_equalize_hist_first.png')

    Image_equal_2 = equalize_hist(Image_equal_1)
    Image_equal_2.save('./38_equalize_hist_second.png')

    # filter_33 = np.ones((3,3),np.float32)/9
    # Image_filter33 = filter2d(I,filter_33,0)
    # Image_filter33.save('./38_filter33.png')

    # filter_55 = np.ones((5,5),np.float32)/25
    # Image_filter55 = filter2d(I,filter_55,0)
    # Image_filter55.save('./38_filter55.png')

    # filter_77 = np.ones((7,7),np.float32)/49
    # Image_filter77 = filter2d(I,filter_77,0)
    # Image_filter77.save('./38_filter77.png')

    # filter_33_sharpen = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    # Image_filter33 = filter2d(I,filter_33_sharpen,1)
    # Image_filter33.save('./38_filter33_sharpen.png')

    # filter_33 = np.ones((3,3),np.float32)/9
    # Image_filter33 = filter2d(I,filter_33,2)
    # Image_filter33.save('./38_filter33_highboost.png')