import asyncio
import json

async def handle_client(reader, writer):
    while True:
        data = await reader.read(1024)
        if not data:
            break
        positions = json.loads(data.decode())
        print("Received drone positions:")
        for pos in positions:
            print(f"Drone {pos['drone_id']}: Lat {pos['latitude']}, Lon {pos['longitude']}, Alt {pos['altitude']}")
    writer.close()

async def do_other_operations():
    while True:
        print("Performing other tasks...")
        await asyncio.sleep(5)  # Simulates a background task
        # You can add other logic here (e.g., logging, monitoring, etc.)

async def main():
    server = await asyncio.start_server(handle_client, 'localhost', 8889)
    addr = server.sockets[0].getsockname()
    print(f'Position receiver serving on {addr}')

    # Run server and background tasks concurrently
    await asyncio.gather(
        server.serve_forever(),
        do_other_operations()
    )

asyncio.run(main())