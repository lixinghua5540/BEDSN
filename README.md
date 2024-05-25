# BEDSN
Title: *Boundary-enhanced dual-stream network for semantic segmentation of high-resolution remote sensing images* <br>https://doi.org/10.1080/15481603.2024.2356355

X. Li, L. Xie, C. Wang, J. Miao, H. Shen and L. Zhang, “Boundary-enhanced dual-stream network for semantic segmentation of high-resolution remote sensing images,” GIScience & Remote Sensing, vol. 61, pp. 2356355, May, 2024.
<br>
<br>
***Introduction***<br>
<br>
Deep convolutional neural networks (DCNNs) have been successfully used in semantic segmentation of high-resolution remote sensing images (HRSIs). However, this task still suffers from intra- class inconsistency and boundary blur due to high intra-class heterogeneity and inter-class homogeneity, considerable scale variance, and spatial information loss in conventional DCNN- based methods. Therefore, a novel boundary-enhanced dual-stream network (BEDSN) is proposed, in which an edge detection branch stream (EDBS) with a composite loss function is introduced to compensate for boundary loss in semantic segmentation branch stream (SSBS). EDBS and SSBS are integrated by highly coupled encoder and feature extractor. A lightweight multilevel information fusion module guided by channel attention mechanism is designed to reuse intermediate boundary information effectively. For aggregating multiscale contextual information, SSBS is enhanced by multiscale feature extraction module and hybrid atrous convolution module. Extensive experiments have been tested on ISPRS Vaihingen and Potsdam datasets. Results show that BEDSN can achieve significant improvements in intra-class consistency and boundary refinement. Compared with 11 state-of-the-art methods, BEDSN exhibits higher-level performance in both quantitative and visual assessments with low model complexity. <br>
<br>
<br>![overall](https://github.com/lixinghua5540/BEDSN/assets/75232301/a531ef9d-24f2-4878-9197-b1f464c46c62)
<br>
***Usage***<br>
The implementation code contains several versions of BEDSN and some of the comparartive methods. <br>
The training and testing codes are in the ***Train_Test*** folder, if the network belongs to conventional semantic segmentation, run ***train.py***, if the network is semantic segmentation enhanced by edge detection, run ***train_edge_combined.py***, and we provided two versions of the evaluation process including ***evaluate_operation.py*** and ***Nonboundary_Evaluation.py*** <br>
First, the data is public dataset from ISPRS 2D semantic labeling contest, and the original data can be acquired on https://www.isprs.org/education/benchmarks/UrbanSemLab/Default.aspx.. Nessesary preprocess of the data is provided in ***Data_processing***<br>
