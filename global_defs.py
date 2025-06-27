"""
用于全局定义
"""


# 每个草图中的笔划数
n_stk = 11  # 自建机械草图
# n_stk = 30  # 自建机械草图
# n_stk = 5  # quickdraw apple
# n_stk = 4  # quickdraw apple
# n_stk = 16  # quickdraw apple
# n_stk = 8  # quickdraw apple
# n_stk = 5  # Tu-Berlin
# n_stk = 16  # Tu-Berlin

# 每个笔划中的点数
# n_stk_pnt = 32  # 自建机械草图
# n_stk_pnt = 32  # quickdraw apple
n_stk_pnt = 32  # quickdraw apple
# n_stk_pnt = 32  # quickdraw apple
# n_stk_pnt = 32  # Tu-Berlin

# 笔划抬起时的后缀，该点的下一个点属于另一个笔划
# pen_up = 16  # 自建机械草图
pen_up = 0  # quickdraw

# 笔划在绘制时的后缀
# pen_down = 17  # 自建机械草图
pen_down = 1  # quickdraw

# 单个草图中的总点数
n_skh_pnt = n_stk * n_stk_pnt






# # 每个草图中的笔划数
# n_stk = None
#
# # 每个笔划中的点数
# n_stk_pnt = None
#
# # 笔划抬起时的后缀，该点的下一个点属于另一个笔划
# pen_up = None
#
# # 笔划在绘制时的后缀，该点的下一个点仍属于该笔划
# pen_down = None
#
# # 草图中的全部点数
# n_skh_pnt = None
#
# # 是否已初始化变量
# _initialized = False
#
#
# def _init_once():
#     """
#     初始化全局变量
#     :return:
#     """
#     global _initialized, n_stk, n_stk_pnt, pen_up, pen_down, n_skh_pnt
#     if not _initialized:
#         print("Initialize global_defs")
#         n_stk = 10
#         n_stk_pnt = 20
#
#         n_stk = 5  # 自建机械草图
#         # n_stk = 30  # 自建机械草图
#         # n_stk = 5  # quickdraw apple
#         # n_stk = 4  # quickdraw apple
#         # n_stk = 16  # quickdraw apple
#         # n_stk = 8  # quickdraw apple
#         # n_stk = 5  # Tu-Berlin
#         # n_stk = 16  # Tu-Berlin
#
#         # n_stk_pnt = 32  # 自建机械草图
#         # n_stk_pnt = 32  # quickdraw apple
#         n_stk_pnt = 32  # quickdraw apple
#         # n_stk_pnt = 32  # quickdraw apple
#         # n_stk_pnt = 32  # Tu-Berlin
#
#         # pen_up = 16  # 自建机械草图
#         pen_up = 0  # quickdraw
#
#         # pen_down = 17  # 自建机械草图
#         pen_down = 1  # quickdraw
#
#         # 单个草图中的总点数
#         n_skh_pnt = n_stk * n_stk_pnt
#
#         _initialized = True
#
#
# # 主动调用
# _init_once()


if __name__ == '__main__':
    print('n_stk:', n_stk)
    print('n_stk_pnt:', n_stk_pnt)
    print('pen_up:', pen_up)
    print('pen_down:', pen_down)
    print('n_skh_pnt:', n_skh_pnt)
