# OWT -DNet: A Timely and High-Accuracy End-to-End Offshore Wind Turbine Detection Network Based on Multimodal Remote Sensing Data

ABSTRACT:
With the rapid development of the offshore wind power industry in recent years, its effects on local social, economic and ecological environments have attracted widespread attention. Therefore, a timely understanding of the development status of offshore wind power, specifically, offshore wind turbines (OWTs) is crucial for the healthy and sustainable development of the offshore wind power industry . However, existing OWT detection methods often struggle to achieve timely, high-precision end-to-end detection of OWTs. To address this, in this study, an OWT detection network (OWT-DNet) based on multimodal remote sensing data is proposed. This network integrates Sentinel-1 synthetic aperture radar (SAR) imagery and Sentinel-2 optical imagery, effectively addressing the insufficient semantic information inherent in single-modal data for OWT detection. Experiments across five global test regions demonstrate that OWT-DNet achieves detection accuracy, recall, and comprehensive evaluation metrics that exceed 99.9%. Furthermore, OWT-DNet demonstrates outstanding detection performance under complex weather conditions. Comparative and ablation experiments validate the network's superior capability in OWT detection tasks. Overall, timely, high-precision end-to-end OWT detection is achieved for the first time on the basis of multimodal remote sensing data. Furthermore, an inaugural multimodal OWT sample dataset is established, laying a solid foundation for future OWT detection research. 


## Data Availability

**Note:**

### I. We provide the following data:
1. [OWT-DNet model training weights](https://drive.google.com/file/d/1f8TFYgmIKAbe3txAnvjTC246K1S5wm0u/view?usp=drive_link)  
2. [Partial test images](https://drive.google.com/file/d/1BCXfhZODQZ-9NzyDDZoFW-lI5eCB-w8W/view?usp=drive_link) (as shown below)  
3. [RGB images of the five global test regions](https://drive.google.com/file/d/1j9VqEQgzhHJEUd0l85_Ozi2kDA23ReAR/view?usp=drive_link) and [prediction results with their vectorized data](https://drive.google.com/file/d/1YzixTQumdinuzVy-m4oZm9KeZkazKKei/view?usp=drive_link)  

---

### II. Explanations

#### 1. Test images description
We provide a set of images for evaluating the model’s performance, but you may also create your own test images.  
Here, we selected **nine (512 × 512) images** from **Test Region C** (waters near Nantong City, China), which are representative of the OWT distribution in this region.  
Each image contains **five bands (VV, VH, B2, B3, B8)**. Their spatial distribution is shown in the figure below:  

![Location map of the 9 test images at Image C (waters near Nantong City, China)](https://github.com/GZ200122/OWT_DNet/blob/main/Location%20map%20of%20the%209%20test%20images%20at%20Image%20C%20(waters%20near%20Nantong%20City%2C%20China).jpg)

---

#### 2. RGB images of the five global test regions
Since the complete set of multimodal images is too large, we provide the corresponding **RGB images** here for convenient viewing.  
If you require the **multimodal images**, please refer to **Table 1 in the paper** for the acquisition dates and download them accordingly.
