# BEDSN
Title: *Boundary-enhanced dual-stream network for semantic segmentation of high-resolution remote sensing images* [[paper]](https://www.sciencedirect.com/science/article/pii/S0924271621001842)<br>https://doi.org/10.1080/15481603.2024.2356355

X. Li, L. Xie, C. Wang, J. Miao, H. Shen and L. Zhang, “Boundary-enhanced dual-stream network for semantic segmentation of high-resolution remote sensing images,” GIScience & Remote Sensing, vol. 61, pp. 2356355, May, 2024.
<br>
<br>
***Introduction***<br>
<br>
Deep convolutional neural networks (DCNNs) have been successfully used in semantic segmentation of high-resolution remote sensing images (HRSIs). However, this task still suffers from intra- class inconsistency and boundary blur due to high intra-class heterogeneity and inter-class homogeneity, considerable scale variance, and spatial information loss in conventional DCNN- based methods. Therefore, a novel boundary-enhanced dual-stream network (BEDSN) is proposed, in which an edge detection branch stream (EDBS) with a composite loss function is introduced to compensate for boundary loss in semantic segmentation branch stream (SSBS). EDBS and SSBS are integrated by highly coupled encoder and feature extractor. A lightweight multilevel information fusion module guided by channel attention mechanism is designed to reuse intermediate boundary information effectively. For aggregating multiscale contextual information, SSBS is enhanced by multiscale feature extraction module and hybrid atrous convolution module. Extensive experiments have been tested on ISPRS Vaihingen and Potsdam datasets. Results show that BEDSN can achieve significant improvements in intra-class consistency and boundary refinement. Compared with 11 state-of-the-art methods, BEDSN exhibits higher-level performance in both quantitative and visual assessments with low model complexity. <br>
<br>
![Fig](https://user-images.githubusercontent.com/75232301/187924458-19ba5dd3-f4bc-4f25-958e-e5a7a349bd9e.jpg)<br>
<br>
<br>
***Usage***<br>
The implementation code of the proposed method consists of two parts:***Deep translation*** and ***Change detection*** <br>
First, you should run ***Deep translation*** folder. deep translation is the code of deep migration, and the input data need to cut the image into small pictures to build samples<br>
Second, ***Change detection*** floder is to use the migrated image for Change detection. The example data given here is Gloucester-SAR, but without data enhancement<br>
