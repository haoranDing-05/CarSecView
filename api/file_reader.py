from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import uvicorn
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
        with open('./test/normal_run_data.txt', 'r') as file:
            for line in file:
                yield f"data: {{\"data\": \"{line.strip()}\"}}\n\n"
                await asyncio.sleep(1)
    return StreamingResponse(generate(), media_type="text/event-stream")
