import pandas as pd
import matplotlib.pyplot as plt
import ipdb
from datetime import datetime, timedelta

def custom_date_parser(x):
    return datetime.strptime(x, '%M:%S.%f')

df = pd.read_csv('../traces/AzureLLMInferenceTrace_conv.csv', parse_dates=['arrival_timestamp'], date_format='%M:%S.%f')

# 设置时间列为索引
df.set_index('arrival_timestamp', inplace=True)

# 创建一个新的列，表示时间的总秒数（从0开始）
df['elapsed_seconds'] = (df.index - df.index[0]).total_seconds()

# 按分钟重采样
resampled_df = df.resample('S').mean()  # 按分钟重采样


# 计算滚动平均（例如7分钟的滚动平均）
rolling_mean_df = resampled_df.rolling(window=10).mean()

# 绘制原始数据和滚动平均的数据
plt.figure(figsize=(14, 10))


#plt.plot(resampled_df.index, resampled_df['prompt_size'], label=f'org - prompt_size', alpha=0.5)
plt.plot(rolling_mean_df.index, rolling_mean_df['prompt_size'], label=f'10s rolling - prompt_size', linewidth=2)
# 设置时间格式为分钟：秒
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(timedelta(seconds=x))))

# prompt_token = code_data[['token_size']]
#data = 
# bins = [100, 500, 1000, 1500, 2000, 3000, 4000, 6000, 7000, 8000]
# bin2 = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800]
# data = code_data['arrival_timestamp'] 
# #n, bins, patches = plt.hist(prompt_token, bins=bin2, color='gray')
# plt.plot(code_data['arrival_timestamp'], code_data['prompt_size'])
# 在每个bin上标出数据大小
# for i in range(len(n)):
#     plt.text(bins[i], n[i], str(n[i]))
plt.xlabel('t')
plt.ylabel('token')
plt.title('conv prompt')
plt.legend()
plt.show()
plt.savefig('code.png')