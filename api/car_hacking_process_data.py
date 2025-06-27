import time
import numpy as np
import pandas as pd
from api.car_queue import TimeSlidingWindow

# 对car_hacking Dataset数据进行预处理和特征提取
async def car_hacking_process_data(file_path, sliding_window):
    if 'txt' in file_path:
        file = pd.read_csv(file_path, header=None, sep=r'\s+')
        file = file.loc[:, [1, 3] + list(range(6, 15))]
        file = file[:-1]
        file['new'] = 'R'
        columns = range(0, 12)
        file.columns = columns
        is_attack = 0
        print(file)
    elif 'csv' in file_path:
        file = pd.read_csv(file_path)
        columns = range(0, 12)
        file.columns = columns
        is_attack = 1
    else:
        print("file_path不规范")
        return

    data = clean_data(file, is_attack)
    data = data.values
    sliding_data = TimeSlidingWindow(is_attack)
    time_series=sliding_data.get_times()

    results = []
    labels = []
    start_time = data[0][0]

    for index in range(len(data)):
        sliding_data.add_data(data[index])
        if sliding_data.is_full() and start_time + sliding_window < sliding_data.get_start_time():
            # print(1)
            result, label,times = sliding_data.get_result()
            results.extend(result)
            labels.extend(label)
            start_time = sliding_data.get_start_time()
        progress_bar(index, len(data))

    return results, labels,time_series


# 针对car_hacking Dataset的数据进行数据清洗
def clean_data(data, attack):
    # data = data.dropna(axis=0) // 该操作将有用信息全部删除了
    data = concatenate_columns(data)
    result = data[[0, 1, 2, 'new_column']].copy()
    if attack:
        col2_plus_2 = data[2] + 3
        result[3] = [data.loc[i, col] if col in data.columns else None
                     for i, col in enumerate(col2_plus_2)]
    result = result.dropna(axis=0)
    return result


def concatenate_columns(data):
    # 获取最大列数
    max_cols = data[2].astype(int) + 2
    max_col_overall = max_cols.max()

    # 创建一个空的结果Series
    result = pd.Series('', index=data.index)

    # 对每个可能的列进行向量化操作
    for col in range(3, max_col_overall + 1):
        if col in data.columns:
            # 只在需要该列的行中添加
            mask = max_cols > col - 1
            result[mask] += data[col][mask].astype(str)

    data['new_column'] = result
    return data


def progress_bar(progress, total, bar_length=50):
    percent = progress / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r|{bar}| {percent:.2%}', end='')

    # 当进度完成时换行
    if progress == total-1:
        print()


if __name__ == '__main__':
    file_path_txt = r'../Car-Hacking Dataset/normal_run_data/normal_run_data.txt'
    file_path_csv = r"..\Car-Hacking Dataset\gear_dataset.csv"
    feature, label = car_hacking_process_data(file_path_csv)
    print(len(label))
    print(label)
