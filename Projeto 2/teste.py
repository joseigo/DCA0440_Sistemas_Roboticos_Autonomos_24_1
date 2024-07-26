import numpy as np
import matplotlib.pyplot as plt

def find_lowest_value_path(grid, start, end, search_range):
    path = [start]
    current_point = start
    nPoints = 1000

    while current_point != end:
        x, y = current_point
        
        # Determine the search range boundaries
        x_min = max(0, x - search_range)
        x_max = min(grid.shape[0], x + search_range + 1)
        y_min = max(0, y - search_range)
        y_max = min(grid.shape[1], y + search_range + 1)
        
        # Extract the subgrid
        subgrid = grid[x_min:x_max, y_min:y_max]
        
        # Find the minimum value's location in the subgrid
        min_index = np.unravel_index(np.argmin(subgrid, axis=None), subgrid.shape)
        min_point = (x_min + min_index[0], y_min + min_index[1])
        
        # Update the current point
        current_point = min_point
        path.append(current_point)
        
        # If the end point is within the subgrid, stop the loop
        if (x_min <= end[0] < x_max) and (y_min <= end[1] < y_max):
            path.append(end)
            break
        nPoints += 1
        if(nPoints>=1000):
            break
    return path

# Example usage
grid = np.random.rand(500, 500)  # Example grid with random values
start = (0, 0)
end = (499, 499)
search_range = 10  # Define the search range

path = find_lowest_value_path(grid, start, end, search_range)

# Plotting the path on the grid
plt.imshow(grid, cmap='viridis', origin='lower')
path_x, path_y = zip(*path)
plt.plot(path_y, path_x, color='red', marker='o')
plt.scatter([start[1], end[1]], [start[0], end[0]], color='blue', marker='x')  # Mark start and end points
plt.title('Path from Start to End')
plt.colorbar(label='Grid Values')
plt.show()
