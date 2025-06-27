import time
from collections import deque, defaultdict
import numpy as np
import pandas as pd


class TimeSlidingWindow:
    def __init__(self,  window_size=30):
        self.window_size = window_size
        self.data_queue = deque()
        self._is_full = False

    def add_data(self, data):
        current_time = data[0]
        # 移除过期数据（时间戳早于当前时间-窗口大小的数据）
        while self.data_queue and current_time - self.get_start_time() > self.window_size:
            self.data_queue.popleft()
            self._is_full = True
        # 添加新数据（数据+当前时间戳）
        self.data_queue.append(data)

    def get_result(self):
        t_start = self.get_start_time()
        t_end = self.get_end_time()
        result = []
        label = []
        index = 0
        while t_start <= t_end:
            temp_data = defaultdict(int)
            temp_dlc = defaultdict(int)
            number = 0
            flag = 1
            while index < self.get_window_size() and self.data_queue[index][0] < t_start + 3:
                temp_data[self.data_queue[index][1]] += 1
                temp_dlc[self.data_queue[index][3]] += 1
                if self.data_queue[index][-1] == 'T':
                    flag = 0
                index += 1
                number += 1

            if temp_data and number >= 50:
                # start_time = time.time()
                temp_dlc_1 = np.array([v for v in temp_dlc.values()])
                # print(time.time()-start_time)
                # temp_dlc = temp_dlc.groupby([0]).size()
                temp_res = [len(temp_data), np.mean(temp_dlc_1), np.std(temp_dlc_1)]
                result.append(temp_res)
                label.append(flag)

            t_start += 3
        return result, label

    def get_window_data(self):
        return [item for item in self.data_queue]

    def get_window_size(self):
        return len(self.data_queue)

    def get_start_time(self):
        return self.data_queue[0][0]

    def get_end_time(self):
        return self.data_queue[-1][0]

    def is_full(self):
        return self._is_full

    def test(self):
        print(self.data_queue[-1][0] - self.data_queue[0][0])
