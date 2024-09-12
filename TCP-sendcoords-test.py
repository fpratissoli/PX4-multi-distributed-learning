import socket
import json

def send_coordinates(coords):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 8888))
        s.sendall(json.dumps(coords).encode())

# Example usage
coords = [
    [0, 0, 20],
    [0, 0, 30],
    [0, 0, 40]
]
send_coordinates(coords)