# coding=utf-8
from PIL import Image
import numpy as np
from numpy.linalg import solve
import os,sys 
def scale(image,size):#size[0]:highth size[1]: width
    I_array = np.array(image)
    np.savetxt("image.txt",I_array)
    tar_width , tar_height = size
    src_height , src_width = I_array.shape[0:2]
    

    height_scale = src_height / tar_height
    width_scale = src_width / tar_width

    scaled_array = np.zeros((tar_height,tar_width), np.uint8)
    print(scaled_array.shape)
    for h in range(tar_height):
        for w in range(tar_width):
            #float source localation
            x = (w+0.5) * width_scale - 0.5
            y = (h+0.5) * height_scale - 0.5

            #int source localation
            x_0 = int(np.floor(x))
            y_0 = int(np.floor(y))
            x_1 = min(x_0 + 1, src_width -1)
            y_1 = min(y_0 + 1, src_height -1)

            if x_0 == x_1 :
                scaled_array[h,w] = int((y_1 - y) * I_array[y_0,x_0] + (y - y_0) * I_array[y_1,x_0])
            elif y_0 == y_1 :      
                scaled_array[h,w] = int((x_1 - x) * I_array[y_0, x_0] + (x - x_0) * I_array[y_0, x_1])
            else:
                f_up = (x_1 - x) * I_array[y_1, x_0] + (x - x_0) * I_array[y_1, x_1]
                f_down = (x_1 - x) * I_array[y_0, x_0] + (x - x_0) * I_array[y_0, x_1]
                scaled_array[h,w] = int((y_1 - y) * f_down + (y - y_0) * f_up) 

    np.savetxt("scaled.txt",scaled_array)           
    return Image.fromarray(scaled_array)      

    # if src_width > tar_width and src_height > tar_height: #down scale
    #     for w in range(tar_width):
    #         for h in range(tar_height):
    #             corr_x = (w+0.5)/h*pic.shape[0]-0.5
    #             corr_y = (j+0.5)/w*pic.shape[1]-0.5
    # elif src_width < tar_width and src_height < tar_height : #up scale
    #     for width in range(size[0]):
    #         for heigth in range(size[1]):
    #             pass
    # else: # scale
    #     for width in range(size[0]):
    #         for heigth in range(size[1]):
    #             pass
    #return scaled_image


if __name__ == "__main__":
    I = Image.open('38.png') 
    #I.show()    
    Image_scaled = scale(I,(192,128))
    Image_scaled.save('./scaled_192_168.png')
    Image_scaled = scale(I,(96,64))
    Image_scaled.save('./scaled_96_64.png')
    Image_scaled = scale(I,(48,32))
    Image_scaled.save('./scaled_48_32.png')
    Image_scaled = scale(I,(24,16))
    Image_scaled.save('./scaled_24_16.png')
    Image_scaled = scale(I,(300,200))
    Image_scaled.save('./scaled_300_200.png')
    Image_scaled = scale(I,(450,300))
    Image_scaled.save('./scaled_450_300.png')
    Image_scaled = scale(I,(500,200))
    Image_scaled.save('./scaled_500_200.png')
