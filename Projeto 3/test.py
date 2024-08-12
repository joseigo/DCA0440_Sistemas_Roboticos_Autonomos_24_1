import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Example grid initialization
grid_size = (10, 10)
grid = np.random.rand(*grid_size)

# Function to update the grid
def update_grid():
    global grid
    while True:
        # Simulate grid updates
        grid = np.random.rand(*grid_size)
        threading.Event().wait(0.1)  # Adjust the delay for your needs

# Function to update the plot
def update_plot(frame):
    plt.cla()  # Clear the current axes
    plt.imshow(grid, cmap='viridis')

# Start the grid updating thread
grid_update_thread = threading.Thread(target=update_grid, daemon=True)
grid_update_thread.start()

# Set up the plot
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update_plot, interval=200)  # Update interval in milliseconds

plt.show()
