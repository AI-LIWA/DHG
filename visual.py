import numpy as np
import matplotlib.pyplot as plt

# 加载模型和数据
pred = np.load('pred.npy')
data = np.load('./class14/test_ntus.npy')

# 连通性
connectivity = [(1, 2), (2,1), (3, 1), (4, 3), (5, 4),(6,5),(7,1),(8,7),(9,8),(10,9),(11,1),(12,11),
                (13,12),(14,13),(15,1),(16,15),(17,16),(18,17),(19,1),(20,19),(21,20),(22,21)]

# 创建绘图窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 定义标签和汉字对应关系
label_to_chinese = {
    0: '抓',
    1: '点击',
    2: '扩大',
    3: '捏',
    4: '顺时针旋转',
    5: '逆时针旋转',
    6: '向右滑动',
    7: '向左滑动',
    8: '向上滑动',
    9: '向下滑动',
    10: '滑动X',
    11: '滑动V',
    12: '滑动+',
    13: '摇'
}

# 指定帧数范围
start_frame = 0
end_frame = 20

# 指定要预测的样本索引
index = 663#20捏 663顺时针
sample = data[index]
predicted_label = np.argmax(pred[index])

# 对指定数据段进行预测
for i in range(start_frame, end_frame):
    points = sample[i]
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    ax.scatter(X, Y, Z, c='r', marker='o')

    # 根据连通性绘制线段
    for edge in connectivity:
        start = points[edge[0] - 1]
        end = points[edge[1] - 1]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='b')

    # 设置视角
    #ax.view_init(elev=90, azim=0)
    ax.view_init(elev=-45, azim=30)

    if i == end_frame - 1:
        # 显示识别结果在最后一帧结束后
        recognized_label = predicted_label
        chinese_label = label_to_chinese[recognized_label]
        ax.text2D(0.05, 0.95, f'识别结果 {chinese_label}', transform=ax.transAxes, fontsize=12)
        plt.draw()  # 强制更新图形窗口
        plt.pause(0.5)  # 暂停一小段时间，以显示动态效果
    else:
        ax.text2D(0.05, 0.95, '等待播放完毕...', transform=ax.transAxes, fontsize=12)

        plt.draw()  # 强制更新图形窗口
        plt.pause(0.4) #0.5 # 暂停一小段时间，以显示动态效果
        ax.cla()  # 清空图像

plt.show()
