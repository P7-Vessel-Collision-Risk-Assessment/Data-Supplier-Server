from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json, uvicorn
from asyncio import sleep

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def waypoints_generator():
    for i in range(5000):
        data = {
            "lat": 50 + i,
            "lng": 20 - i,
        }
        data = json.dumps(data)
        yield f"event: locationUpdate\ndata: {data}\n\n"
        await sleep(1)


@app.get("/get-waypoints")
async def root():
    return StreamingResponse(waypoints_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
