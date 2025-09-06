 <div align="center">

# CHaRM: Conditioned Heatmap Regression Methodology for Accurate and Fast Dental Landmarks Localization

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>
 
## Abstract   
Identifying anatomical landmarks in 3D dental models is essential for orthodontic treatment, yet manual placement is labor-intensive and requires expert knowledge. While machine learning methods have been proposed for automatic landmark detection in 3D Intraoral Scans (IOS), none provide a fully end-to-end solution that avoids costly tooth segmentation.

We present CHaRM (Conditioned Heatmap Regression Methodology), the first fully end-to-end deep learning approach for tooth landmark detection in 3D IOS. CHaRM integrates four components: a point cloud encoder, a decoder with a heatmap regression head, a teeth-presence classification head, and the novel CHaR module. The CHaR module leverages teeth-presence information to adapt to missing teeth, improving detection accuracy in complex dental cases. Unlike two-stage workflows that segment teeth before landmarking, CHaRM operates directly on IOS point clouds, reducing complexity, avoiding error propagation, and lowering computational cost. 

We evaluated CHaRM with five point cloud learning backbones on IOSLandmarks-1k, a new dataset of 1,214 annotated 3D dental models. Both the dataset and code will be publicly released  to address the scarcity of open data in orthodontics and foster reproducible research. 

CHaRM with PointMLP, named CHaRNet, achieved the best accuracy and efficiency. Compared to state-of-the-art methods (TSMDL and ALIIOS), CHaRNet reduced mean Euclidean distance error to 0.56 mm on standard dental models and 1.12 mm across all dentition types, while delivering up to 14.8× faster inference on GPU.
This end-to-end approach streamlines orthodontic workflows, enhances the precision of 3D IOS analysis, and enables efficient computer-assisted treatment planning.

## Dependencies
Insert instructions to install necessary software...

## How to run   
Insert instructions to run the project/experiments


### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```