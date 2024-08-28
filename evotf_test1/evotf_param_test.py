import pickle

# 指定.pkl文件的路径
file_path = '/home/cyh/evosax/evosax/strategies/ckpt/evotf/2024_03_SNES_small.pkl'


# 打开.pkl文件，并加载数据
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 显示数据
print(data)

# 打开一个新的txt文件，准备写入
with open('/home/cyh/evosax/evotf_test1/2024_03_SNES_small.txt', 'w') as f:
    # 将屏幕上的输出写入到txt文件中
    f.write(str(data))