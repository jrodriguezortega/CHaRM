# Third-party notices

CHaRM is released under the MIT license (see [LICENSE](LICENSE)). Parts of
it are adapted from the open-source projects below, which remain under their
own licenses. The full license texts are available in the linked
repositories.

| Component in this repository | Adapted from | License |
| --- | --- | --- |
| `src/charm/models/pointnext.py`, `src/charm/models/dgcnn.py`, `src/charm/models/blocks.py` | [PointNeXt / OpenPoints](https://github.com/guochengqian/PointNeXt) (Guocheng Qian et al.) | MIT |
| `src/charm/models/pointmlp.py` | [pointMLP-pytorch](https://github.com/ma-xu/pointMLP-pytorch) (Xu Ma et al.) | Apache-2.0 |
| `src/charm/models/pointtransformer.py` | [point-transformer](https://github.com/POSTECH-CVLab/point-transformer) (POSTECH-CVLab, Hengshuang Zhao) | See the upstream repository |
| `packages/pointnet2-batch/` (CUDA kernels) | [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch) (Shaoshuai Shi), as distributed with OpenPoints | MIT |
| `packages/pointops/` (CUDA kernels) | `pointops` of [point-transformer](https://github.com/POSTECH-CVLab/point-transformer) (Hengshuang Zhao) | See the upstream repository |

Changes made to the adapted code include: a single landmark network per
backbone with the `base`, `postproc` and `charm` variants, removal of unused
layers and options, pure PyTorch fallbacks for every CUDA operator, and
updates of the C++ extensions to the current PyTorch API.

The architectures are described in:

- Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in
  a Metric Space", NeurIPS 2017.
- Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", ACM
  Transactions on Graphics 2019.
- Zhao et al., "Point Transformer", ICCV 2021.
- Qian et al., "PointNeXt: Revisiting PointNet++ with Improved Training and
  Scaling Strategies", NeurIPS 2022.
- Ma et al., "Rethinking Network Design and Local Geometry in Point Cloud: A
  Simple Residual MLP Framework", ICLR 2022.
