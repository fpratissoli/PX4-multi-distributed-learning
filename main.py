import signal
import asyncio
import json
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__name__).split("/")[0:-1])+'/PX4-multi-drone')
from multidrone_control import DroneSwarm, read_config

class TCPServer:
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.message_callback = None

    async def start(self):
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port)
        
        addr = server.sockets[0].getsockname()
        print(f'Command server serving on {addr}')

        async with server:
            await server.serve_forever()

    async def handle_client(self, reader, writer):
        while True:
            data = await reader.read(1024)
            if not data:
                break

            message = data.decode()
            coords = json.loads(message)
            print(f"Received coordinates: {coords}")

            if self.message_callback:
                await self.message_callback(coords)

        writer.close()
        await writer.wait_closed()

    def set_message_callback(self, callback):
        self.message_callback = callback

class TCPClient:
    def __init__(self, host='localhost', port=8889):
        self.host = host
        self.port = port
        self.writer = None

    async def connect(self):
        reader, self.writer = await asyncio.open_connection(self.host, self.port)
        print(f"Connected to position broadcast server at {self.host}:{self.port}")

    async def send_data(self, data):
        if self.writer:
            self.writer.write(json.dumps(data).encode())
            await self.writer.drain()
        else:
            print("Not connected to server")

    async def close(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()

async def initialize_swarm():
    config = read_config()
    swarm = DroneSwarm(config)
    await swarm.connect_swarm()
    await asyncio.sleep(2)  # Wait for the drones to connect
    await swarm.takeoff_swarm()
    await asyncio.sleep(5)  # Wait for the drones to take off
    return swarm

async def run_swarm_mission(swarm, coords):
    print(f"Running mission to coordinates: {coords}")
    await swarm.run_goto_local(coords)

async def broadcast_drone_positions(swarm, tcp_client):
    while True:
        positions = []
        for drone in swarm.alldrones:
            position = await swarm.get_local_coords(drone.id-1)
            positions.append({
                'drone_id': drone.id,
                'latitude': position['x'],
                'longitude': position['y'],
                'altitude': position['z']
            })
        await tcp_client.send_data(positions)
        print(f"Broadcasted drone positions: {positions}")
        await asyncio.sleep(2)  # Adjust the interval as needed

async def main():
    # Initialize the swarm
    swarm = await initialize_swarm()

    # Create and configure the TCP server for receiving commands
    tcp_server = TCPServer(port=8888)
    tcp_server.set_message_callback(lambda coords: run_swarm_mission(swarm, coords))

    # Create TCP client for broadcasting positions
    tcp_client = TCPClient(port=8889)
    await tcp_client.connect()

    # Start the position broadcast task
    broadcast_task = asyncio.create_task(broadcast_drone_positions(swarm, tcp_client))

    # Setup signal handlers
    loop = asyncio.get_running_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(s, loop, broadcast_task, tcp_client))
        )

    # Start the TCP server for receiving commands
    await tcp_server.start()

    # Clean up
    # broadcast_task.cancel()
    # await tcp_client.close()

async def shutdown(signal, loop, broadcast_task, tcp_client):
    print(f"Received exit signal {signal.name}...")
    broadcast_task.cancel()
    await tcp_client.close()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

if __name__ == "__main__":
    asyncio.run(main())