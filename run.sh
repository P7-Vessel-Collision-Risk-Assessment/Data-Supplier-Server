#!/bin/bash

cd sse-server
cargo build
cd ..
python main.py