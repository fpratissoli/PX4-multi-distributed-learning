import socket
import json
import threading
import time

def handle_client(client_socket):
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        positions = json.loads(data.decode())
        print("Received drone positions:")
        for pos in positions:
            print(f"Drone {pos['drone_id']}: Lat {pos['latitude']}, Lon {pos['longitude']}, Alt {pos['altitude']}")
    client_socket.close()

def background_operations():
    """Function that runs other operations in the background."""
    while True:
        # Simulate background tasks
        print("Performing background tasks...")
        time.sleep(3)  # Perform the task every 5 seconds
        # Add other operations here (e.g., logging, monitoring, etc.)

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 8889))
    server.listen(5)
    print("Position receiver serving on ('localhost', 8889)")

    # Start background operations in a separate thread
    background_thread = threading.Thread(target=background_operations, daemon=True)
    background_thread.start()

    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        # Create a new thread for each client connection
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()

if __name__ == "__main__":
    main()
