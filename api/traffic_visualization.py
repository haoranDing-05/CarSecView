from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import uvicorn
import numpy as np
import pandas as pd
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import asyncio

@app.get("/read_dataset")
async def read_dataset():
    async def generate():
        file_path = "./test/normal_run_data.txt"
        file = pd.read_csv(file_path, header=None, sep=r'\s+')
        file = file.loc[:, [1, 3] + list(range(6, 15))]
        file = file[:-1]
        columns = range(0, 11)
        file.columns = columns
        data=file.values
        index=1
        while True:
            if index >= len(data):
                index = 1
            timestamp =time.time()+data[index][0]-data[index-1][0]
            # 构建一个更清晰的数据字符串
            data_str = (f"data: {{\"data\": \"{timestamp},"
                       f"ID:{data[index][1]},"
                       f"DLC:{data[index][2]},"
                       f"{','.join(str(data[index][i]) for i in range(3,int(data[index][2]) + 3))}"
                       f"\"}}\n\n")
            index += 1
            yield data_str
            await asyncio.sleep(data[index][0]-data[index-1][0])

    return StreamingResponse(generate(), media_type="text/event-stream")
