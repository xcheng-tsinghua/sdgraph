## 定义
### 1. std sketch
保存为 .txt 文件，每行记录一个点，数据之间以英文逗号分隔，如下所示。

x, y, s

x, y, s

...

x, y, s

草图质心在(0, 0)，范围在[-1, 1]^2。std草图中不同草图的笔划数及笔划上的点数可能不同。

### 2. unified_std sketch
保存方式与 std 草图类似，不同点在于 unified_std 草图中不同草图的笔划数及笔划上的点数相同。

SDGraph 类的输入需要是 unified_std sketch。

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


