# coding=utf-8
import cv2
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt 
import math
from hw2 import *

def averequalize_hist(image):
    I_array = np.array(image)
    #print(I_array.shape[:])
    src_height , src_width , src_channal = I_array.shape[:]
    #calculate the scr picture histogram
    hist_arry = np.zeros((256,3),np.uint32)
    for h in range(src_height):
        for w in range(src_width):
            for c in range(src_channal): 
                hist_arry[I_array[h,w,c],c] += 1
    #print(hist_arry.shape)
    pdf_arry = np.zeros(256,np.float)
    #print(hist_arry.sum(axis=1))
    pdf_arry = hist_arry.sum(axis=1)/float(I_array.size)
    #print(pdf_arry)
    #calculate the scr picture cdf
    cdf_arry = np.zeros(256,np.float)
    cdf_arry[0] = pdf_arry[0]
    for i in range(1,256):
        cdf_arry[i] = cdf_arry[i-1] + pdf_arry[i]
    #creat new picture by equalizing its hist
    scaled_array = np.zeros((src_height,src_width,src_channal), np.uint8)
    for h in range(src_height):
        for w in range(src_width):
            for c in range(src_channal): 
                scaled_array[h,w,c] = np.uint8(round(255*cdf_arry[I_array[h,w,c]]))
    return scaled_array

def hsi_equalize_hist(image):
    I_array = np.array(image)
    hsi_img = RGB2HSI(I_array)
    #hsi_img = cv2.cvtColor(I_array,cv2.COLOR_BGR2HSV)
    #print(hsi_img.shape[:])
    src_height , src_width , src_channal = hsi_img.shape[:]
    Intensity = np.uint8(hsi_img[:,:,2] *255)
    I_equal = equalize_hist(Intensity)
    # #creat new picture by equalizing its hist
    scaled_img = np.zeros((src_height,src_width,src_channal))
    scaled_img[:,:,0] = hsi_img[:,:,0]
    scaled_img[:,:,1] = hsi_img[:,:,1]
    scaled_img[:,:,2] = I_equal / 255.0
    return HSI2RGB(scaled_img)
    #return cv2.cvtColor(scaled_img,cv2.COLOR_HSV2BGR)
    #return HSI2RGB(hsi_img)

def RGB2HSI(rgb_img):
    """
    这是将RGB彩色图像转化为HSI图像的函数
    :param rgm_img: RGB彩色图像
    :return: HSI图像
    """
    #保存原始图像的行列数
    row = rgb_img.shape[0]
    col = rgb_img.shape[1]
    #对原始图像进行复制
    hsi_img = np.zeros((row,col,3))
    #对图像进行通道拆分
    B,G,R = cv2.split(rgb_img)
    #把通道归一化到[0,1]
    [B,G,R] = [ i/ 255.0 for i in ([B,G,R])]
    print(B)
    H = np.zeros((row, col))    #定义H通道
    I = (R + G + B) / 3.0       #计算I通道
    S = np.zeros((row,col))      #定义S通道
    den = np.zeros((row,col))
    thetha = np.zeros((row,col))
    min_rgb = np.zeros((row,col))
    # for i in range(row):
    #     for j in range(col):
    #     den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
    #     thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)   #计算夹角
    #     h = np.zeros(col)               #定义临时数组
    #     #den>0且G>=B的元素h赋值为thetha
    #     h[B[i]<=G[i]] = thetha[B[i]<=G[i]]
    #     #den>0且G<=B的元素h赋值为thetha
    #     h[G[i]<B[i]] = 2*np.pi-thetha[G[i]<B[i]]
    #     #den<0的元素h赋值为0
    #     h[den == 0] = 0
    #     H[i] = h/(2*np.pi)      #弧度化后赋值给H通道
    for i in range(row):
        for j in range(col):
            if np.max([R[i][j],G[i][j],B[i][j]]) == np.min([B[i][j],G[i][j],R[i][j]]):
                H[i][j] =0
                S[i][j] =0
            else:
                den[i][j] = 0.5*(R[i][j]-B[i][j]+R[i][j]-G[i][j]) / np.sqrt((R[i][j]-G[i][j])**2+(R[i][j]-B[i][j])*(G[i][j]-B[i][j]))
                if den[i][j]>1:
                    den[i][j]=1
                elif den[i][j]<-1:
                    den[i][j]=-1
                else:
                    den[i][j] = den[i][j]
                thetha[i][j] = np.arccos(den[i][j])   #计算夹角
                #den>0且G>=B的元素h赋值为thetha
                if B[i][j]<=G[i][j]:
                    H[i][j] = thetha[i][j]
                #den>0且G<=B的元素h赋值为thetha
                else:
                    H[i][j] = 2*np.pi - thetha[i][j]
                #找出每组RGB值的最小值
                min_rgb[i][j] = np.min([B[i][j],G[i][j],R[i][j]])
                #计算S通道
                S[i][j] = 1-min_rgb[i][j]/I[i][j]

    #扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
    hsi_img[:,:,0] = H 
    hsi_img[:,:,1] = S
    hsi_img[:,:,2] = I
    return hsi_img

def HSI2RGB(hsi_img):
    """
    这是将HSI图像转化为RGB图像的函数
    :param hsi_img: HSI彩色图像
    :return: RGB图像
    """
    # 保存原始图像的行列数
    row = np.shape(hsi_img)[0]
    col = np.shape(hsi_img)[1]
    #对原始图像进行复制
    rgb_img = hsi_img.copy()
    #对图像进行通道拆分
    H,S,I = cv2.split(hsi_img)
    # #把通道归一化到[0,1]
    # [H,S,I] = [ i/ 255.0 for i in ([H,S,I])]

    # #H转化成角度
    # H = 2 * np.pi *H

    R = np.zeros((row, col))    #定义H通道
    G = np.zeros((row,col))       #计算I通道
    B = np.zeros((row,col))      #定义S通道
    TMP1 = np.zeros((row, col))
    TMP1 = np.where(I*(1-S)<0.0,0.0,I*(1-S))
    for i in range(row):
        for j in range(col):
            if H[i][j]>=0 and H[i][j]< 2*np.pi/3:
                B[i][j] = TMP1[i][j]
                R[i][j] = I[i][j]*(1+(S[i][j]*np.cos(H[i][j]))/np.cos(np.pi/3 - H[i][j]))
                G[i][j] = 3*I[i][j]-(B[i][j]+R[i][j])
            elif H[i][j]>= 2*np.pi/3 and H[i][j]< 4*np.pi/3:
                R[i][j] = TMP1[i][j]
                G[i][j] = I[i][j]*(1+(S[i][j]*np.cos(H[i][j]-2*np.pi/3))/np.cos(np.pi/3 - (H[i][j]-2*np.pi/3)))
                B[i][j] = 3*I[i][j]-(R[i][j]+G[i][j])
            else:
                G[i][j] = TMP1[i][j]
                B[i][j] = I[i][j]*(1+(S[i][j]*np.cos(H[i][j]-4*np.pi/3))/np.cos(np.pi/3 - (H[i][j]-4*np.pi/3)))
                R[i][j] = 3*I[i][j]-(G[i][j]+B[i][j])
    rgb_img[:,:,0] = B*255
    rgb_img[:,:,1] = G*255
    rgb_img[:,:,2] = R*255
    return rgb_img


if __name__ == "__main__":
    img = cv2.imread("./38.png",1)
    #print(type(img_gray))
    B,G,R = cv2.split(img)
    B_equal = equalize_hist(B)
    G_equal = equalize_hist(G)
    R_equal = equalize_hist(R)
    img_equal = cv2.merge([B_equal,G_equal,R_equal])
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.subplot(1,2,2)
    # plt.imshow(img_equal)
    # plt.show()
    cv2.imwrite("./38_equal.png", img_equal, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    img_averequal = averequalize_hist(img)
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.subplot(1,2,2)
    # plt.imshow(img_averequal)
    # plt.show()
    cv2.imwrite("./38_averequal.png", img_averequal, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    img_hsiequal = hsi_equalize_hist(img)
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.subplot(1,2,2)
    # plt.imshow(img_averequal)
    # plt.show()
    cv2.imwrite("./38_hsiequal.png", img_hsiequal, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])