# learn-neural

## upd 2023-9-13

### instant-ngp

简单实现了instant-ngp用于拟合图像，在使用2层隐藏层每层32节点的mlp与hash grid encode下达成了还不错的效果

还存在的问题：

1. hash grid encode哈希表访问太慢，需要优化访存
2. 原论文在计算feature gradient时直接使用float atomic add，在我的电脑（gpu 1660ti）实现时太慢（只要用了浮点原子加就只有3帧），现在的实现是直接不处理加法直接写入未相加的梯度，但仍然能收敛

![image](assets/1694540879471.png)
