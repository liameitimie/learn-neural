# learn-neural

## upd 2023-10-29

尝试在访问哈希表时，block内每个线程处理4个输入，在block内将hash index排序后再访问，访问时间有所提升，但块内排序shared访问过多，最终速度不如之前，优化失败

## upd 2023-10-16

重新组织一下代码

## upd 2023-9-13

### instant-ngp

简单实现了instant-ngp用于拟合图像，在使用2层隐藏层每层32节点的mlp与hash grid encode下达成了还不错的效果

还存在的问题：

1. hash grid encode哈希表访问太慢，需要优化访存
2. 原论文在计算feature gradient时直接使用float atomic add，在我的电脑（gpu 1660ti）实现时太慢（只要用了浮点原子加就只有3帧），现在的实现是直接不处理加法直接写入未相加的梯度，但仍然能收敛

![image](assets/1694540879471.png)
