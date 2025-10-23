import asyncio
from rplidarc1 import RPLidar

async def monitor_errors():
    lidar = RPLidar("/dev/ttyUSB1", 460800)  # already opens serial and runs healthcheck
    total = errors = 0

    scan_task = asyncio.create_task(lidar.simple_scan(make_return_dict=True))

    try:
        while True:
            data = await lidar.output_queue.get()
            #print(data.keys())
            total += 1
            if 'error' in data:
                errors += 1
            if total % 500 == 0:
                print(f"Errors: {errors} / {total} ({errors/total*100:.4f}%)")
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        await lidar.stop()
        lidar.close()

asyncio.run(monitor_errors())
