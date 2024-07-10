import pandas as pd
import matplotlib.pyplot as plt
import time
import glob

def get_max_memory_from_csv(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 创建一个时间轴和memory值的列表
    time_points = sorted(set(df['iteration_start']).union(df['iteration_end']))
    memory_values = [0] * len(time_points)

    # 计算每个时间点的memory值
    for i, time_point in enumerate(time_points):
        memory_values[i] = df[(df['iteration_start'] <= time_point) & (df['iteration_end'] >= time_point)]['memory'].sum()

    # 找到memory的最大值
    max_memory = max(memory_values)
    
    return max_memory

def plot_max_memory_across_csvs(folder_path):
    # 记录程序开始时间
    start_time = time.time()

    # 获取文件夹中所有CSV文件的列表
    csv_files = glob.glob(f"{folder_path}/*.csv")
    # 存储每个CSV文件的最大memory值
    max_memories = []
    file_names = []

    for csv_file in csv_files[18:]:
        max_memory = get_max_memory_from_csv(csv_file)
        max_memories.append(max_memory)
        file_names.append(csv_file.split('/')[-1])  # 只取文件名

    # 绘制每个CSV文件的最大memory值的折线图
    plt.figure(figsize=(14, 7))
    plt.plot(file_names, max_memories, marker='o', linestyle='-', color='blue', label='Max Memory Usage')
    plt.xticks(rotation=90)  # 旋转x轴标签以便更好地显示
    plt.xlabel('CSV Files')
    plt.ylabel('Max Memory')
    plt.title('Max Memory Usage Across CSV Files (TP = 1 die, prefill die = decode die = 18')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # 自动调整子图参数以给标签留出更多空间
    plt.show()

    # 记录程序结束时间
    end_time = time.time()

    # 计算并输出程序执行时间
    execution_time = end_time - start_time
    print(f"程序执行时间: {execution_time:.2f} 秒")

# 使用示例
plot_max_memory_across_csvs('your_csv_folder')