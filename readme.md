## 车辆检测

**实现步骤：**

* 在标记的训练图像集上执行定向梯度直方图（HOG）特征提取并训练SVM分类器
* 实施滑动窗口技术并使用训练完成的分类器搜索图像中的车辆
* 应用热力图(heatMap)过滤错误检测(false positive)
* 估计检测到的车辆的边界框

[//]: # (Image References)
[image1]: ./output/Vehicle_Nonvehicle.png
[image2]: ./output/hog.png
[image3]: ./output/Multi_window.png
[image4]: ./output/Window_example.png
[image5]: ./output/error.png
[image6]: ./output/hot.png
[image7]: ./output/Result_image.png
[video1]: ./project_video.mp4


### 项目的环境 

**软件** 
* Windows10 (64bit)
* Anaconda 4.2.0

**硬件**
* CPU: Intel(R) Core(TM) i7-4578K CPU @ 3.00GHZ
* GPU: 核心显卡
* 内存: 16GB

####数据集

以下是用于训练分类器的车辆和非车辆的数据链接。图像来自 [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html),  [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/)


#### 提取图片HOG特征,并显示样本图像
训练数据为64x64x3的RBG图片，包含车辆与非车辆图片两类，车辆图片8971张，非车辆图片8799张。
以下为车辆，非车辆图片样例：
![alt text][image1]

这个项目的第一个功能是HOG。它使用图像渐变，因此无论颜色如何，它都可以提取车辆图像的特征。

图像和HOG可视化的示例如下：

![alt text][image2]

#### 训练分类器

这里使用SVM分类器



最终训练的分类器在测试数据集得到98.7%的准确率。

#### 应用滑动窗口(sliding windows)实现车辆检测

我用滑动窗口技术找到车辆。
在图像中，如果车辆远离摄像机，则车辆的尺寸很小。 如果车辆离相机很近，则车辆的尺寸很大。 因此，我使用多尺度窗口来寻找各种尺寸的车辆。 示例图像如下。

![alt text][image3]


多尺度窗口和车辆检测结果如下。

![alt text][image4]

发现了一些错检和重复检测的现象。

![alt text][image5]

---

#### 应用热图(heatMap)过滤错误检测(false positive)

由于使用多个大小不一滑动窗口，且窗口存在重叠，单个车辆图像会被多个窗口捕捉检测。使用这个现象可以过滤错误检测。

记录一张图片上所有positive detections，使用记录的positive detections形成一个检测热图：



以下应用在测试图片得到的检测热图：
![alt text][image6]



 然后对热图进行阈值过滤,过滤错误检测,结果如下
  ![alt text][image7]
 
####结果

最终结果视频如下。
 


[结果视频](./output/project.mp4)

