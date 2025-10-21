import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rplidarc1 import RPLidar
import asyncio
from collections import deque
import matplotlib.cm as cm  # Import colormap module

# --- Configuration ---
LIDAR_PORT = "/dev/ttyUSB1"
BAUDRATE = 460800
BUFFER_SIZE = 600  # You can tune this value
MAX_DISTANCE_MM = 8000

# --- Global variables ---
scan_buffer = deque(maxlen=BUFFER_SIZE)
lidar_is_running = True


def lidar_data_collector(lidar):
    global scan_buffer, lidar_is_running
    print("Lidar data collector thread started.")

    async def run_the_scan(lidar_obj):
        print("Starting Lidar scan...")
        await lidar_obj.simple_scan(make_return_dict=True)

    async def process_the_queue(queue, stop_event):
        while lidar_is_running:
            try:
                measurement_dict = await asyncio.wait_for(queue.get(), timeout=1.0)
                scan_buffer.append(measurement_dict)
            except asyncio.TimeoutError:
                continue
        print("Setting stop event for Lidar...")
        stop_event.set()

    async def main_async_loop():
        async with asyncio.TaskGroup() as tg:
            tg.create_task(run_the_scan(lidar))
            tg.create_task(process_the_queue(lidar.output_queue, lidar.stop_event))

    try:
        asyncio.run(main_async_loop())
    except ExceptionGroup as eg:
        print(f"LIDAR ERROR: The asyncio TaskGroup failed. Details:")
        for i, error in enumerate(eg.exceptions):
            print(f"  - Sub-exception {i + 1}: {error}")
            import traceback;
            traceback.print_exception(error)
    except Exception as e:
        print(f"Lidar thread encountered a non-TaskGroup error: {e}")
    finally:
        print("Resetting Lidar...")
        lidar.reset()
        print("Lidar thread finished.")

# --- MODIFIED update_plot FUNCTION ---
# It now accepts the scatter plot artist and the colormap.
def update_plot(frame, scatter_artist, colormap):
    """
    Plots a snapshot of the current points in the buffer, with color
    representing the quality (q value).
    """
    scan_data_copy = list(scan_buffer)
    if not scan_data_copy:
        # If no data, return the artist without changes
        return scatter_artist,

    # Extract all angle, distance, and quality values from the list of dictionaries
    angles_deg = [(90-d['a_deg']) for d in scan_data_copy]
    distances_mm = [d['d_mm'] for d in scan_data_copy]
    quality_values = [d['q'] for d in scan_data_copy]

    angles_rad = np.deg2rad(angles_deg)

    # Update the data for the scatter plot
    # The 'c' argument takes the color data (normalized quality)
    # The 'cmap' argument specifies the colormap to use
    scatter_artist.set_offsets(np.c_[angles_rad, distances_mm])  # Set (angle, distance) pairs
    scatter_artist.set_array(quality_values)  # Set color data for each point

    return scatter_artist,

def main():
    global lidar_is_running
    lidar = None
    try:
        print(f"Connecting to Lidar on port {LIDAR_PORT}...")
        lidar = RPLidar(LIDAR_PORT, BAUDRATE, timeout=3)
        lidar_thread = threading.Thread(target=lidar_data_collector, args=(lidar,), daemon=True)
        lidar_thread.start()

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        # --- CODE TO ADD FOR BLACK BACKGROUND ---

        # Set the background color of the plotting area
        ax.set_facecolor('black')

        # Set the background color of the figure (the area outside the plot)
        fig.set_facecolor('black')

        # Change the color of the title to be visible
        ax.title.set_color('white')

        # Change the color of the radial and angular grid lines
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

        # Change the color of the axis labels (the numbers)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        ax.set_theta_zero_location('N')
        ax.set_theta_direction('clockwise')

        # Change the color of the plot's circular border (spine)
        ax.spines['polar'].set_edgecolor('white')

        # --- END OF ADDED CODE ---

        # --- MODIFIED PLOT INITIALIZATION ---
        # Use ax.scatter() for individual point coloring.
        # Initial call creates an empty scatter plot.
        # 'c' is the color data, 'cmap' is the colormap, 'vmin/vmax' define the color range.
        # We need to explicitly pass a colormap to update_plot.

        # Define the colormap (e.g., 'jet' for blue to red)
        color_map = cm.get_cmap('jet')

        scatter_artist = ax.scatter(
            [], [],  # Initial empty data
            c=[],  # Initial empty color data
            cmap=color_map,
            vmin=0, vmax=63,  # Scale the color bar from 0 to 63 (RPLidar quality range)
            s=5  # Size of the markers
        )

        ax.set_title(f"RPLidar Scan (Quality Map)", pad=20)
        ax.set_rlim(0, MAX_DISTANCE_MM)
        ax.set_rlim(0, 3500)
        ax.grid(True)

        # Add a color bar to show the quality scale
        cbar = fig.colorbar(scatter_artist, ax=ax, orientation='vertical', pad=0.1)
        cbar.set_label('Measurement Quality (0=low, 63=high)')

        ani = animation.FuncAnimation(
            fig,
            update_plot,
            fargs=(scatter_artist, color_map),  # Pass the scatter artist and colormap
            interval=100,  # Update every half a second
            blit=False  # Blitting can be tricky with scatter plots and dynamic colors, often set to False
        )
        plt.show()

    except Exception as e:
        print(f"An error occurred in the main program: {e}")
    finally:
        print("Shutting down...")
        lidar_is_running = False
        time.sleep(1.5)


if __name__ == '__main__':
    main()
