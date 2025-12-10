import matplotlib.pyplot as plt

# 假设 data1 和 data2 是您的两个数据列表
data1 = [1, 3, 2, 4, 3, 5]
data2 = [2, 2, 3, 3, 4, 4]

# 绘制第一个数据列表并保存
plt.plot(data1)
plt.title('Data 1')
plt.xlabel('Index')
plt.ylabel('Value')
plt.savefig('data1_plot.png')
plt.clf()  # 清除当前图形

# 绘制第二个数据列表并保存
plt.plot(data2)
plt.title('Data 2')
plt.xlabel('Index')
plt.ylabel('Value')
plt.savefig('data2_plot.png')
plt.clf()
