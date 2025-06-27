import re
import time
import torch
import asyncio
import datetime
import aiofiles  # 用于异步文件操作
import numpy as np
from torch import nn
from fastapi import FastAPI 
from api.LSTM import LSTMAutoencoder
from multiprocessing import Queue
from api.car_queue import TimeSlidingWindow
from sklearn.metrics import confusion_matrix
from api.NLT_main import likelihood_transformation
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from api.car_hacking_process_data import car_hacking_process_data
# # is_attack参数来区别是.txt还是.csv
# car_queue = TimeSlidingWindow(is_attack=False)
# # init likelihood_transformation
# transfer = likelihood_transformation()

# # 对于每一条车俩网流量data
# car_queue.add_data(data)
# result, label = sliding_data.get_result()
# feature=transfer.out(result)

# predict = model.predict(feature)

# # ture的值应该是经过时间t后的result
# loss = model.loss(predict, ture)

#阈值提前计算好对于car-hacking
# result=confusion_matrix(labels, predict_label, labels=[0, 1]).ravel()
# confusion_matrix_accumulator = np.add(confusion_matrix_accumulator, result)
# tn, fp, fn, tp = confusion_matrix_accumulator
# accuracy = (tp + tn) / (tp + tn + fp + fn)
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# gmean=np.sqrt(recall*precision)
# f1=2*precision*recall/(precision+recall)

# yield {"message": "实时混淆矩阵", "data": {"confusion_matrix": confusion_matrix_accumulator, "accuracy": accuracy, "precision": precision, "recall": recall, "gmean": gmean, "f1": f1}}

###sleep一个时间差 ai好像写了查一下
###读一次数据在传到前端时，队列从头积累数据，队列从尾取出数据，
#按照频率对数据进行处理，得到z输入模型得到z'计算loss 和预先知道的阈值对比判断一个标签'
###每次先初始化填充一些数据
###需要测试一下txt最后一个是否为R
app = FastAPI()

def init_model():
    hidden_size = 50
    input_size = 1
    model_file = 'car_hacking_model.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(input_size, hidden_size, device)
    model.load_state_dict(torch.load(f'./model/{model_file}', map_location=device))
    
    return model,device

def init_data_acc():
    while message_queue:
        data_acc.add_data(message_queue.get())
        if data_acc.is_full():
            break
     
message_queue = Queue()
model,device = init_model()
transfer=likelihood_transformation()
data_acc=TimeSlidingWindow()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#时间 ID DLC 数据1 数据2 数据3... 数据n
async def generate_detetc_result(file_path:str):#返回time,loss,label数据对
    init_data_acc()
    yield "data: {\"message\": \"连接已建立，正在加载数据...\"}\n\n"
    #处理数据的频率
    pace=5#一秒五次
    while True:
        await asyncio.sleep(1.0/pace)
        current_time=data_acc.get_start_time()
        while message_queue:
            if data_acc.is_full():
                break
            data_acc.add_data(message_queue.get())
            
        result,label=data_acc.get_result() 
        z=transfer.out(result) 
        z=torch.from_numpy(z).float()
        if z.dim() == 1:
        # 转换为 [1, feature_dim, 1]（假设feature_dim=seq_len）
            z=z.unsqueeze(0).unsqueeze(2)  # 添加batch和feature维度  
        z_hat=model(z)
        criterion=nn.MSELoss()
        loss=criterion(z_hat,z)
        if loss>0.0007:
            label_predict=0
        yield [current_time,loss,label,label_predict]
        
@app.get("/detect_attack")
async def detect_attack(attack_type:str = "正常流量"):
    """读取数据集并以流的形式返回"""
    try:
        file_path = get_file_path(attack_type)
    except ValueError as e:
        async def error_generator():
            yield f"data: {{\"error\": \"{str(e)}\"}}\\n\\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    return StreamingResponse(generate_detetc_result(file_path), media_type="text/event-stream")
            
def get_file_path(attack_type: str) -> str:
    """根据攻击类型获取对应的文件路径"""
    if attack_type == "Dos攻击":
        return "./test/DoS_dataset.csv"
    elif attack_type == "模糊攻击":
        return "./test/Fuzzy_dataset.csv"
    elif attack_type == "RPM攻击":
        return "./test/RPM_dataset.csv"
    elif attack_type == "Gear攻击":
        return "./test/gear_dataset.csv"
    elif attack_type == "正常流量":
        return "./test/normal_run_data.txt"
    else:
        raise ValueError(f"未知攻击类型: {attack_type}")

async def process_csv_line(line: str) -> list:
    """处理CSV格式的单行数据"""
    parts = line.strip().split(',')
    print(parts)
    if len(parts) >= 12:  # 确保行有足够的字段
        try:
            timestamp = float(parts[0])
            _id = parts[1]
            dlc = int(parts[2])
            data_parts = [p for p in parts[3:3+dlc+1]]
            return [timestamp, _id, dlc] + data_parts
        except ValueError:
            return None
    elif len(parts) >= 9:  # 处理字段较少的行
        try:
            timestamp = float(parts[0])
            _id = parts[1]
            dlc = int(parts[2])
            data_parts = [p for p in parts[3:3+dlc]]
            return [timestamp, _id, dlc] + data_parts
        except ValueError:
            return None
    return None

async def process_txt_line(line: str) -> list:
    line = line.strip()
    if not line:
        return []
    
    # 提取时间戳
    timestamp_match = re.search(r'Timestamp: (\d+\.\d+)', line)
    timestamp = [timestamp_match.group(1)] if timestamp_match else []
    timestamp = [float(t) for t in timestamp]
    
    # 提取ID
    id_match = re.search(r'ID: (\w+)', line)
    message_id = [id_match.group(1)] if id_match else []
    # 提取DLC
    dlc_match = re.search(r'DLC: (\d+)', line)
    dlc = [dlc_match.group(1)] if dlc_match else []
    dlc = [int(d) for d in dlc]
    # 提取数据字节
    data_match = re.search(r'DLC: \d+\s+((?:[0-9a-f]{2}\s+)+)', line)
    data_bytes = data_match.group(1).strip().split() if data_match else []
    
    label=['R']
    # 合并所有元素到一个数组
    return timestamp + message_id + dlc + data_bytes+label

async def generate(file_path: str):
    # 初始数据，让客户端知道连接已建立
    yield "data: {\"message\": \"连接已建立，正在加载数据...\"}\n\n"
    
    previous_timestamp = None
    first_data_sent = False
    
    # 根据文件类型选择处理函数
    if file_path.endswith('.csv'):
        process_line = process_csv_line
    else:
        process_line = process_txt_line
    
    # 使用异步文件操作
    async with aiofiles.open(file_path, 'r') as f:
        first_line = True
        line_count = 0
        
        async for line in f:
            # 跳过CSV文件的标题行
            if first_line and file_path.endswith('.csv'):
                first_line = False
                continue
            
            line_count += 1
            # 处理当前行
            processed_data = await process_line(line)
            
            if not processed_data:
                continue
            timestamp = processed_data[0]
            
            # 计算时间差
            if previous_timestamp is not None and first_data_sent:
                time_diff = timestamp - previous_timestamp
                # 限制最大延迟，避免过长等待
                if time_diff > 1.0:
                    await asyncio.sleep(1.0)  # 最大延迟1秒
                else:
                    await asyncio.sleep(time_diff)
            
            previous_timestamp = timestamp
            
            # 格式化时间
            current_time = datetime.datetime.fromtimestamp(time.time())
            message_queue.put([current_time.timestamp(), processed_data[1],processed_data[2],processed_data[-1]])
            current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            data_part,time_part=current_time.split(' ',1)
            #这个是label
            label=processed_data[-1]
            yield f"date_part:{current_time}\n"

            # 构建并发送数据
            # 移除"data: "前缀和JSON结构，直接输出CSV格式
            data_str = (f"{time_part},"
                        f"ID:{processed_data[1]},"
                        f"DLC:{int(processed_data[2])},"
                        f"{','.join(str(processed_data[i]) for i in range(3, int(processed_data[2]) + 3))},"
                        f"{label}\n")
            yield f"{data_str}\n"
            first_data_sent = True
    # 发送结束标记
    yield "data: {\"message\": \"数据传输完成\"}\n\n"

@app.get("/read_dataset")
async def read_dataset(attack_type: str = "Dos攻击"):
    """读取数据集并以流的形式返回"""
    try:
        file_path = get_file_path(attack_type)
    except ValueError as e:
        async def error_generator():
            yield f"data: {{\"error\": \"{str(e)}\"}}\\n\\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    return StreamingResponse(generate(file_path), media_type="text/event-stream")

@app.get("/new_api_endpoint")
async def new_api_endpoint(attack_type: str):
    print(f"Received request for attack type: {attack_type}")
    if attack_type == "Dos攻击":
        return {"message": "Dos攻击数据已接收并处理！", "data": {"type": "Dos", "status": "active"}}
    elif attack_type == "模糊攻击":
        return {"message": "模糊攻击数据已接收并处理！", "data": {"type": "Fuzzing", "status": "completed"}}
    elif attack_type == "RPM攻击":
        return {"message": "RPM攻击数据已接收并处理！", "data": {"type": "RPM", "status": "completed"}}
    elif attack_type == "Gear攻击":
        return {"message": "Gear攻击数据已接收并处理！", "data": {"type": "Gear", "status": "completed"}}
    elif attack_type == "正常流量":
        return {"message": "正常流量数据已接收并处理！", "data": {"type": "Normal", "status": "completed"}}
    else:
        return {"message": f"未知攻击类型 {attack_type} 已接收！", "data": {"type": "Unknown", "status": "N/A"}}