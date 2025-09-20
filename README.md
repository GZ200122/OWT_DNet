# OWT -DNet: A Timely and High-Accuracy End-to-End Offshore Wind Turbine Detection Network Based on Multimodal Remote Sensing Data

ABSTRACT:
With the rapid development of the offshore wind power industry in recent years, its effects on local social, economic and ecological environments have attracted widespread attention. Therefore, a timely understanding of the development status of offshore wind power, specifically, offshore wind turbines (OWTs) is crucial for the healthy and sustainable development of the offshore wind power industry . However, existing OWT detection methods often struggle to achieve timely, high-precision end-to-end detection of OWTs. To address this, in this study, an OWT detection network (OWT-DNet) based on multimodal remote sensing data is proposed. This network integrates Sentinel-1 synthetic aperture radar (SAR) imagery and Sentinel-2 optical imagery, effectively addressing the insufficient semantic information inherent in single-modal data for OWT detection. Experiments across five global test regions demonstrate that OWT-DNet achieves detection accuracy, recall, and comprehensive evaluation metrics that exceed 99.9%. Furthermore, OWT-DNet demonstrates outstanding detection performance under complex weather conditions. Comparative and ablation experiments validate the network's superior capability in OWT detection tasks. Overall, timely, high-precision end-to-end OWT detection is achieved for the first time on the basis of multimodal remote sensing data. Furthermore, an inaugural multimodal OWT sample dataset is established, laying a solid foundation for future OWT detection research. 


一、我们提供了以下一些数据：
  1. [OWT-DNet模型训练权重](https://drive.google.com/file/d/1f8TFYgmIKAbe3txAnvjTC246K1S5wm0u/view?usp=drive_link)
  2. [部分测试影像](https://drive.google.com/file/d/1BCXfhZODQZ-9NzyDDZoFW-lI5eCB-w8W/view?usp=drive_link)（如下图所示）
  3. [全球五个测试地区的RGB影像]()和[预测结果及其矢量化数据](https://drive.google.com/file/d/1YzixTQumdinuzVy-m4oZm9KeZkazKKei/view?usp=drive_link)

二、说明

  1.测试影像说明：
    我们提供了一些影像以便测试模型的性能，当然您也可以自己制作一些影像做测试。这里选择了C测试区（中国南通市附近海域）的9个512*512大小的影像，位置分布如下图，代表了该地区OWT分布特点，每个影像有5个波段（VV、VH、B2、B3、B8）.
![image](https://github.com/GZ200122/OWT_DNet/blob/main/Location%20map%20of%20the%209%20test%20images%20at%20Image%20C%20(waters%20near%20Nantong%20City%2C%20China).jpg)

    
  3.全球五个测试地区的RGB影像说明：
    由于完整的多模态影像数量太大了，因此这里给大家提供对应的RGB影像便于查看，如果需要对应的多模态影像，可以参照论文中的表1时间点进行下载。
