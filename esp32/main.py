from machine import UART
import sys
import time

# USB serial = sys.stdin / sys.stdout
# MicroPython REPL + print() also appear here

print("MicroPython serial test starting...")
i = 0
while i < 100:
    # Send message periodically
    print(f"Are you receiving me? {i}")

    # Check if any data came from the computer
    if sys.stdin.buffer.readable():
        # Non-blocking read of any available bytes
        import select

        poll = select.poll()
        poll.register(sys.stdin, select.POLLIN)
        ready = poll.poll(10)  # 10 ms timeout

        if ready:
            incoming = sys.stdin.readline().strip()
            if incoming:
                print("ESP RECEIVED:", incoming)

    time.sleep(1)
