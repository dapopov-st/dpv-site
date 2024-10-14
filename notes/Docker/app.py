# app.py
import os
import time

# Create a file and write some content to it
with open("output.txt", "a") as f:
    f.write(f"Hello from Docker!\nTime: {time.time()}\n")

print("File written successfully.")