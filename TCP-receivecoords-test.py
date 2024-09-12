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

async def main():
    server = await asyncio.start_server(handle_client, 'localhost', 8889)
    addr = server.sockets[0].getsockname()
    print(f'Position receiver serving on {addr}')

    async with server:
        await server.serve_forever()

asyncio.run(main())