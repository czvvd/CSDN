# CSDN: Cross-modal Shape-transfer Dual-refinement Network for Point Cloud Completion

This repository contains the PyTorch implementation of the paper:

**[CSDN: Cross-modal Shape-transfer Dual-refinement Network for Point Cloud Completion](https://ieeexplore.ieee.org/abstract/document/10015045), TVCG 2023**

<!-- <br> -->
[Zhe Zhu](https://scholar.google.com/citations?user=pM4ebg0AAAAJ),  [Liangliang Nan](https://3d.bk.tudelft.nl/liangliang/index.html), [Haoran Xie](https://scholar.google.com/citations?user=O4lGUj8AAAAJ), [Honghua Chen](https://scholar.google.com/citations?user=S7yyHpAAAAAJ), [Jun Wang](https://scholar.google.com/citations?user=vFYyThwAAAAJ), [Mingqiang Wei](https://scholar.google.com/citations?user=TdrJj8MAAAAJ), [Jing Qin](https://harry-qinjing.github.io/).
<!-- <br> -->

## Abstract

> How will you repair a physical object with some missings? 
You may imagine its original shape from previously captured images, recover its overall (global) but coarse shape first, and then refine its local details. 
We are motivated to imitate the physical repair procedure to address point cloud completion.
To this end, we propose a cross-modal shape-transfer dual-refinement network (termed CSDN), a coarse-to-fine paradigm with images of full-cycle participation,  for quality point cloud completion.
CSDN mainly consists of "shape fusion" and "dual-refinement" modules to tackle the cross-modal challenge.
The first module transfers the intrinsic shape characteristics from single images to guide the geometry generation of the missing regions of point clouds, in which we propose IPAdaIN to embed the global features of both the image and the partial point cloud into completion. The second module refines the coarse output by adjusting the positions of the generated points, where the local refinement unit exploits the geometric relation between the novel and the input points by graph convolution, and the global constraint unit utilizes the input image to fine-tune the generated offset.
Different from most existing approaches, CSDN not only explores the complementary information from images but also effectively exploits cross-modal data in the whole coarse-to-fine completion procedure.
Experimental results indicate that CSDN performs favorably against twelve competitors on the cross-modal benchmark.

## Get Started

### Environment and Installation
Code has been tested with Ubuntu 20.04, GCC 9.4.0, Python 3.6, PyTorch 1.8.2, CUDA 11.1 and cuDNN 8.1.0.

Install PointNet++ and Chamfer Distance.
```
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

cd ../Chamfer3D

python setup.py install
```


### Dataset
Download the ShapeNetViPC dataset from [ViPC](https://github.com/Hydrogenion/ViPC) and specify the data path in Train.py.

### Training
```
python Train.py
```

## Citation
If you use CSDN in your research, please consider citing our paper:
```bibtex
@article{zhu2023csdn,
  title={CSDN: Cross-modal Shape-transfer Dual-refinement Network for Point Cloud Completion},
  author={Zhu, Zhe and Nan, Liangliang and Xie, Haoran and Chen, Honghua and Wang, Jun and Wei, Mingqiang and Qin, Jing},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2023},
  publisher={IEEE}
}
```


## Acknowledgement
The code is based on [ViPC](https://github.com/Hydrogenion/ViPC). Some of the code is borrowed from:
- [SpareNet](https://github.com/microsoft/SpareNet)
- [GRNet](https://github.com/hzxie/GRNet)
- [PytorchPointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)

The point clouds are visualized with [Easy3D](https://github.com/LiangliangNan/Easy3D).

We thanks the authors for their great work.

## License

This project is open sourced under MIT license.


