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
from api.car_queue import TimeSlidingWindow
from sklearn.metrics import confusion_matrix
from api.NLT_main import likelihood_transformation
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from api.car_hacking_process_data import car_hacking_process_data
#正常流量的read_detect是错的
#切换后要清空数据积累重新积累
#攻击csv的处理没有给dlc为2的加上T或R
#封面组名 校标 制作人
app = FastAPI()

def init_model():
    hidden_size = 50
    input_size = 1
    model_file = 'car_hacking_model.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(input_size, hidden_size, device)
    model.load_state_dict(torch.load(f'./model/{model_file}', map_location=device))
    model.to(device)
    model.eval()
    return model,device
     
model,device = init_model()
transfer=likelihood_transformation()
transfer.set_global_max(0.09638328545934269)
datas=TimeSlidingWindow()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#切换或者重启都不传
#时间 ID DLC 数据1 数据2 数据3... 数据n
async def generate_detect_result(file_path: str):#返回time,loss,label数据对
    async for data in add2queue(file_path):
        datas.add_data(data)
        if datas.is_full():
            current_time=datas.get_start_time()
            result,label=datas.get_result() 
            if(result):
                z=transfer.out(result) 
                z=torch.from_numpy(z).float()
                if z.dim() == 1:
                    z = z.view(1, -1, 1) 
                z=z.to(device)
                z_hat=model(z)
                #加载慢是模型计算慢 还是异步的问题 不应该有关系
                criterion=nn.MSELoss()
                loss=criterion(z_hat,z)
                label_predict=1
                if loss>0.0004979983381813239:
                    label_predict=0

                if 0 in label:
                    label=0
                else:
                    label=1
                    
                yield f"{current_time,float(loss),label,label_predict}\n"
        
#Dos攻击 模糊攻击 RPM攻击 Gear攻击 正常流量
@app.get("/detect_attack")
async def detect_attack(attack_type: str="正常流量"):
    """读取数据集并以流的形式返回"""
    try:
        file_path = get_file_path(attack_type)
    except ValueError as e:
        async def error_generator():
            yield f"data: {{\"error\": \"{str(e)}\"}}\\n\\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    return StreamingResponse(generate_detect_result(file_path), media_type="text/event-stream")
      
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
    data_match = re.search(r'DLC: \d+\s+((?:[0-9a-f]{2}\s+)+(?:[0-9a-f]{2}))', line)
    data_bytes = data_match.group(1).strip().split() if data_match else []
    
    label=['R']
    # 合并所有元素到一个数组
    return timestamp + message_id + dlc + data_bytes+label
    
async def process_csv_line_add(line: str) -> list:
    """处理CSV格式的单行数据"""
    parts = line.strip().split(',')
    if len(parts) >= 12:  # 确保行有足够的字段
        try:
            timestamp = float(parts[0])
            _id = parts[1]
            dlc = int(parts[2])
            data_parts = [p for p in parts[3:3+dlc]]
            label=parts[-1]
            data_parts_str=""
            for byte in data_parts:
                data_parts_str+=byte
            result=[timestamp, _id, dlc, data_parts_str,label]
            return result
        except ValueError:
            return None
    elif len(parts) >= 9:  # 处理字段较少的行
        try:
            timestamp = float(parts[0])
            _id = parts[1]
            dlc = int(parts[2])
            data_parts = [p for p in parts[3:3+dlc+1]]
            label=parts[-1]
            data_parts_str=""
            for byte in data_parts:
                data_parts_str+=byte
            result=timestamp + _id + dlc
            result.append(data_parts_str)
            result+=label 
            return result
        except ValueError:
            return None
    return None

async def process_txt_line_add(line: str) -> list:
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
    data_match = re.search(r'DLC: \d+\s+((?:[0-9a-f]{2}\s+)+(?:[0-9a-f]{2}))', line)
    data_bytes = data_match.group(1).strip().split() if data_match else []
    
    data_bytes_str=""
    for byte in data_bytes:
        data_bytes_str+=byte
    label=['R']
    
    
    result=timestamp + message_id + dlc 
    result.append(data_bytes_str)
    result+=label
    return result
    
async def add2queue(file_path: str):
    # 初始数据，让客户端知道连接已建立
    previous_timestamp = None
    first_data_sent = False
    
    # 根据文件类型选择处理函数
    if file_path.endswith('.csv'):
        process_line = process_csv_line_add
    else:
        process_line = process_txt_line_add
    
    # 使用异步文件操作
    async with aiofiles.open(file_path, 'r') as f:
        first_line = True
        line_count = 0
        time_diff=0
        
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
                if(time_diff<0):
                    time_diff=0
                # 限制最大延迟，避免过长等待
                if time_diff > 1.0:
                    await asyncio.sleep(1.0)  # 最大延迟1秒
                else:
                    await asyncio.sleep(time_diff)
            previous_timestamp = timestamp
            
            current_time = time.time()
            new_timestamp=current_time+time_diff
        
            #切换数据导致的不同数据集的原始时间戳的先后关系 会将datas中的队列清空 试图用当前时间戳代替原始时间戳
            yield [new_timestamp,processed_data[1],processed_data[2],processed_data[3],processed_data[-1]]
        
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