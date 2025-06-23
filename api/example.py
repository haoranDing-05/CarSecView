## 本文件为伪代码

# 文件说明
# car_queue.py 车联网流量队列 基于滑动窗口思想存放车联网流量数据
# LSTM LSTM模型
# NLT_main.py likelihood_transformation

# how to use
# initit car_queue
# is_attack参数来区别是.txt还是.csv
car_queue = TimeSlidingWindow(is_attack=False)
# iniit likelihood_transformation
transfer = likelihood_transformation()
# init model 自己配一下参数
model = LSTM()


# 对于每一条车俩网流量data
car_queue.add_data(data)
result, label = sliding_data.get_result()
feature=transfer.out(result)

predict = model.predict(feature)

# ture的值应该是经过时间t后的result
loss = model.loss(predict, ture)