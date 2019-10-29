import cv2
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt 
import math

def FFT2d(img,flag):
    img_fft = FFT(FFT(img).T).T
    M,N = img_fft.shape
    M = int(M/2)
    N = int(N/2)
    if flag == 0:
        return np.vstack((np.hstack((img_fft[M:,N:],img_fft[M:,:N])),np.hstack((img_fft[:M,N:],img_fft[:M,:N]))))
    else:
        return img_fft / (M*N)

def FFT(f):
    N = f.shape[1] 
    if N <= 8:
        return np.array([DFT(f[i,:]) for i in range(f.shape[0])])
    else:
        F_even = FFT(f[:,::2])
        F_odd = FFT(f[:,1::2])
        W_u2k = np.array([np.exp(-2j * np.pi * np.arange(N) / N) for i in range(f.shape[0])])
        F_u = F_even + np.multiply(W_u2k[:,:int(N/2)],F_odd)
        F_uk = F_even + np.multiply(W_u2k[:,int(N/2):],F_odd)
        return np.hstack([F_u,F_uk])

def DFT(f):
    N = f.size
    W = np.zeros((N,N),dtype=np.complex128)
    for y in range(N):
        for v in range(N):
            W[y][v] = np.exp(-1j*2*np.pi*v*y/N)
    return f.dot(W)

if __name__ == "__main__":
    img_gray = cv2.imread("38.png",0)
    h = img_gray.shape[0]
    n = 1
    k1 = 0
    while h > n :
        n = n * 2
        k1 = k1 + 1
    w = img_gray.shape[1]
    n = 1
    k2 = 0
    while  w > n:
        n = n * 2
        k2 = k2 + 1
    
    pad_h = int(math.pow(2,k1) - img_gray.shape[0]) // 2
    pad_w = int(math.pow(2,k2) - img_gray.shape[1]) // 2

    img_pad = cv2.copyMakeBorder(img_gray, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)

    fft_array = FFT2d(img_pad,0)

    fft_Conj_matrix = np.mat(fft_array).H
    fft_Conj = fft_Conj_matrix.A.T

    ifft_gray = FFT2d(fft_Conj,1)

    fft_array = abs(fft_array)
    npfft_array = abs(fftshift(fft2(img_pad)))
    ifft_gray = abs(ifft_gray)

    print('distance from numpy.fft:',np.linalg.norm(fft_array-npfft_array))
    cv2.imwrite("./38_fft.png",np.log(1+fft_array),[int(cv2.IMWRITE_PNG_COMPRESSION),9])
    cv2.imwrite("./38_ifft.png",ifft_gray,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
    plt.subplot(2,2,1)
    plt.title('gray(after padding)')
    plt.imshow(img_pad,cmap=plt.cm.gray)
    plt.subplot(2,2,2)
    plt.title('my FFT2D')
    plt.imshow(np.log(1+fft_array),cmap=plt.cm.gray)
    plt.subplot(2,2,3)
    plt.title('numpy fft2D')
    plt.imshow(np.log(1+npfft_array),cmap=plt.cm.gray)
    plt.subplot(2,2,4)
    plt.title('ifft gray')
    plt.imshow(ifft_gray,cmap=plt.cm.gray)
    plt.show()
    
