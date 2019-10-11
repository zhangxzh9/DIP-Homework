# Exercises
## 1.1 Linearity
是的，直方图均衡化不是一个线性操作。

## 1.2 Spatial Filtering
### 1.
结果图像为
$$
 \frac{1}{9}
 \left[
  \begin{matrix}
  8 & 13 & 13 & 8\\
  \\
  13 & 21 & 21 & 13\\
  \\
  13 & 21 & 21 & 13\\
  \\
  8 & 13 & 13 & 8
  \end{matrix} 
 \right]
$$

### 2.
反复使用给出滤波器的缺点为使图像变得模糊,无法辨识.

### 3.
卷积与相关操作的不同为:卷积在滑动卷积操作之前需要将滤波器进行旋转180°再进行相关操作

### 4.
1)该滤波器可以用于图像预处理任务中,可以在目标提取之前去除图像中一些琐碎细节
2)该滤波器可以连接直线或者曲线之间的缝隙
3)该滤波器可以降低噪声

## 1.3 Spatial Filtering
滤波器为：
$$
 \frac{1}{9}
 \left[
  \begin{matrix}
  1 & 1 & 1\\
  \\
  1 & 1 & 1\\
  \\
  1 & 1 & 1
  \end{matrix} 
 \right]
$$

# 2 Programming Tasks
## 2.2 Histogram Equalization 
