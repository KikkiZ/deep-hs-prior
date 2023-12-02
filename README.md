# Deep Hyperspectral Prior

​	该库主要源自论文[O Sidorov, JY Hardeberg. Deep Hyperspectral Prior: Denoising, Inpainting, Super-Resolution](https://arxiv.org/abs/1902.00301) 的补充代码，实现的基础来自[original Deep Image Prior code by Dmitry Ulyanov](https://github.com/DmitryUlyanov/deep-image-prior)。该库主要修改了网络搭建，使用2D卷积处理高光谱数据，网络用于高光谱图像去噪。

​	基于上述文献所给出的代码，该库实现并大致取得了文献中所提出的性能。该库的该分支仅实现了模型在去噪领域的应用，并且删除了性能较差的3D卷积。

### Requirements

1. python 3.9
2. pytorch 2.1
3. numpy
4. scipy
5. matplotlib
6. scikit

​	对于依赖的兼容性问题，该库没有进行测试。对于更老版本的python，我们不推荐使用，但不一定无法使用。同时，代码中使用了cuda来提升模型性能，因此，当您的计算机中没有GPU加速训练时需要少量调整代码。

### Prepare the data

​	使用`data`目录下的`prepare_denoising.m`或`prepare_denoising.mlx`来处理原始数据，并将处理好的数据以`denoising.mat`的名称存储在当前目录下。

### Run the code

​	直接运行`bootstrap.py`，并附加需要调整的相关参数。（可调整的参数详见代码）
