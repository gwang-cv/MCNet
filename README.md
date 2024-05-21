# MCNet

Pytorch implementation of MCNet for IEEE/CAA JAS 2023 paper "[MCNet: Multiscale Clustering Network for Two-View Geometry Learning and Feature Matching](https://ieeexplore.ieee.org/abstract/document/10141546)", by Gang Wang and Yufei Chen.

The [pretrained models](https://drive.google.com/drive/folders/1wtcIFn7mFiw82naK3jfoD0IdS_bWMWOE?usp=drive_link) are found in the 'model' folder, including 'yfcc-sift', 'yfcc-superpoint', 'sun3d-sift', and 'sun3d-superpoint'.



If you find this project useful, please cite:

	@article{wang2023mcnet,
	  title={MCNet: Multiscale Clustering Network for Two-View Geometry Learning and Feature Matching},
	  author={Wang, Gang and Chen, Yufei},
	  journal={IEEE/CAA Journal of Automatica Sinica},
	  volume={10},
	  number={6},
	  pages={1507--1509},
	  year={2023},
	  publisher={IEEE}
	}

Requirements
-
Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.1.0). Other dependencies should be easily installed through pip or conda.


Acknowledgement
-
This code is heavily borrowed from [OANet](https://github.com/zjhthu/OANet). If you use the part of code related to data generation, testing and evaluation, you should cite this paper and follow its license.


	@article{zhang2019oanet,
	  title={Learning Two-View Correspondences and Geometry Using Order-Aware Network},
	  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
	  journal={International Conference on Computer Vision (ICCV)},
	  year={2019}
	}

