
# 每个草图中的笔划数
n_stk = 12  # 自建机械草图
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

