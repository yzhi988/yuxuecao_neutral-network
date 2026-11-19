"""
Example usage of plot_training_curves.py
Demonstrates how to flexibly use the plotting tool
"""

from plot_training_curves import main



# 日志文件路径
LOG_PATH = "此前运行的实验结果/Experiment1/logging.txt"
# 图片保存路径
SAVE_PATH = "实验结果/MIT_training_convergence"

# 图片标题
TITLE = "MIT Dataset Training Convergence"

# ==================== 运行绘图 ====================
if __name__ == "__main__":
    main(LOG_PATH, SAVE_PATH, TITLE)

