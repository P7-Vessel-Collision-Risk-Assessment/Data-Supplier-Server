import ctypes

lib = ctypes.CDLL("sse-server/target/debug/libsse_server.so")

lib.start_sse_server()