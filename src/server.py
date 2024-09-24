from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get-waypoints/{mmsi}")
async def get_waypoints(mmsi: str):
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'data', 'mmsi', f'{mmsi}.feather'))
    logger.info(f"Looking for file at: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return {"error": "MMSI not found"}
    
    try:
        data = pd.read_feather(file_path)
        logger.info(f"File read successfully: {file_path}")
        waypoints = data[['mmsi','timestamp','latitude', 'longitude']].to_dict(orient='records')
        return waypoints
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return {"error": "Failed to process file"}