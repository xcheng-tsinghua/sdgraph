## 定义
### 1. STK sketch
在 np.ndarray 中保存时，大小为 [n_stk, n_stk_pnt, 2]。
保存为 .txt 文件时，每行记录一个点，数据之间以英文逗号分隔，如下所示。

x, y, s

x, y, s

...

x, y, s

s = global_defs.pen_up or global_defs.pen_down
数据的行数为 n_stk * n_stk_pnt
草图质心在(0, 0)，范围在[-1, 1]^2。

STK2格式, s = 1表示该点有效， s = 0 表示该点无效，无效位置坐标初始用 (0, 0) 占位
x y s
x y s
...
x y s

### 2. S5 sketch
在 np.ndarray 中保存时，大小为 [n_pnt, 5]。

每行数据为 x, y, p1, p2, p3

其中 (p1, p2, p3) 为 one-hot vector，

p1 = 1: 该点下一个点属于另一个笔划

p2 = 1: 该点下一个点仍属于当前笔划

p3 = 1: 该点为草图最后一个点

### 2. S3 sketch
在 np.ndarray 中保存时，大小为 [n_pnt, 3]。

txt文件中每行为 x y s

不同元素之间以空格分隔，因此读取时使用函数：np.loadtxt(filename)

s = pen_down (0)：该点下一个点仍属于当前笔划

s = pen_up (1)：该点绘制完毕后抬笔

---------------------

## 环境搭建
### 1. 创建 conda 环境  
`conda create -n dp python=3.11`  
`conda activate dp`
---------------------
### 2. 安装工具包
主要工具包为pytorch，其它按要求安装

--------------------------
## 各脚本功能

### 1. global_defs.py
确定草图笔划数，每个笔划中点数，绘制时画笔抬起或放下的后缀

该文件中的数值将应用于多个脚本

------------------

### 2. Dataset
与数据集相关的存放于 data_utils 文件夹

`sketch_statis.py`:

用于统计草图数据集的数据，例如草图中的笔划数，每个草图中的点数。

`sketch_utils.py`:

读取数据集时使用的功能

`sketch_vis.py`:

草图可视化

`SketchDataset.py`:

草图数据集读取，以及草图转化

----------------------

### 3. train
`train_cls.py`
分类训练脚本

`train_diff.py`
生成训练脚本


