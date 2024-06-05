# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('Qt5Agg')

def CirclePoints(Count):
    angles = np.linspace(0, 2 * np.pi, Count, endpoint=False)
    grid_vectors = np.transpose(np.array([np.sin(angles), np.cos(angles)]))
    return grid_vectors


def is_nonzero_det(pair, grid_vectors):
    vectors = grid_vectors[list(pair)]
    det = np.linalg.det(vectors)
    return np.abs(det) > 1e-10


def add_type_colors(area, min_area, max_area):
    """
    Generates colors for a given area using a color map.
    """
    cmap = plt.get_cmap('prism')
    normalized_area = (area - min_area) / (max_area - min_area) if max_area != min_area else 0.5
    return cmap(normalized_area)


def area_of_polygon(vertices):
    """
    Calculates the area of a polygon based on its vertices.
    """
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    area = abs(area) / 2.0
    return np.round(area, decimals=10)


def add_unique_intersection_points(points, grid_indices, unique_points, line_combinations):
    """
    Helper function for adding vertices to a list of unique vertices with their corresponding vectors
    """
    for point, lineCombination in zip(points, line_combinations):
        roundedPoint = tuple(np.round(point, decimals=10))
        gridAndLineSet = {(gridIndex, line) for gridIndex, line in zip(grid_indices, lineCombination)}
        if roundedPoint not in unique_points:
            unique_points[roundedPoint] = gridAndLineSet
        else:
            unique_points[roundedPoint].update(gridAndLineSet)


def filter_convex_hull_points(points):
    """
    Takes an array of points and returns only those points that form the convex hull
    (the outer polygon) of the set.
    :param points: Array of points in the format [(x1, y1), (x2, y2), ...]
    :return: Array of points that form the convex hull.
    """
    points = np.array(points)  # Ensure the points are in a NumPy array format
    if len(points) < 3:
        return points  # A convex hull cannot be formed with fewer than 3 points

    # Calculate the convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]  # Retrieve the points that form the convex hull

    return hull_points


def generate_combinations(corner, vectors):
    all_points = [corner]
    for r in range(1, len(vectors) + 1):
        for subset in combinations(vectors, r):
            all_points.append(corner + np.sum(subset, axis=0))
    return all_points


def are_tiles_connected(tile1, tile2, tol=1e-5):
    vertices1, vertices2 = tile1[0][0], tile2[0][0]
    distances = np.linalg.norm(vertices1[:, np.newaxis, :] - vertices2[np.newaxis, :, :], axis=2)
    return np.any(distances < tol)


def update_connected_components(tiles_data):
    queue = deque([i for i, tile in enumerate(tiles_data) if tile[2]])
    visited = set(queue)
    while queue:
        current_index = queue.popleft()
        current_tile = tiles_data[current_index]
        for i, tile in enumerate(tiles_data):
            if i not in visited and are_tiles_connected(current_tile, tile):
                tile[2] = True
                visited.add(i)
                queue.append(i)
    return tiles_data


def create_tiles(grid_shifts, grid_range, connected_tiles):
    grid_count = len(grid_shifts)
    # Define the guiding vectors
    grid_vectors = CirclePoints(grid_count)
    # Create all possible pairs of indices for gridCount
    tile_type_grid_indices = list(combinations(range(grid_count), 2))
    # Filter pairs for which the determinant of the matrix, composed of the corresponding pairs of guiding vectors, is not zero
    tile_type_grid_indices = list(filter(lambda pair: is_nonzero_det(pair, grid_vectors), tile_type_grid_indices))
    tile_type_count = len(tile_type_grid_indices)
    # Generate line numbers for each grid
    grid_line_numbers = [list(range(-grid_range, grid_range + 1)) for _ in range(grid_count)]
    # List of all unique intersections and corresponding vectors
    unique_grid_line_intersection_points = {}
    for tile_type_index in range(tile_type_count):
        # Select indices for the current pair of grids
        grid_indices = np.array(tile_type_grid_indices[tile_type_index])
        # Generate all combinations of line numbers for the current pair of grids
        line_combinations = np.array(np.meshgrid(grid_line_numbers[grid_indices[0]], grid_line_numbers[grid_indices[1]])).T.reshape(-1, 2)
        # Select guiding vectors and shifts for the current pair of grids
        vectors = [grid_vectors[grid_indices[0]], grid_vectors[grid_indices[1]]]
        shifts = [grid_shifts[grid_indices[0]], grid_shifts[grid_indices[1]]]
        # Calculate intersection points
        grid_line_intersection_points = np.dot(line_combinations - shifts, np.transpose(np.linalg.inv(vectors)))
        # Add to the collection of unique points
        add_unique_intersection_points(grid_line_intersection_points, grid_indices, unique_grid_line_intersection_points, line_combinations)

    unique_intersections = np.array(list(unique_grid_line_intersection_points.keys()))
    grid_shifts_expanded = np.array(grid_shifts)[:, np.newaxis]
    tile_corner_lattice_coordinates = np.ceil(np.dot(grid_vectors, unique_intersections.T) + grid_shifts_expanded).T

    for point, grid_and_lines in unique_grid_line_intersection_points.items():
        i = np.where((unique_intersections == point).all(axis=1))[0][0]
        for grid, line in grid_and_lines:
            tile_corner_lattice_coordinates[i, grid] = line

    # Transform tile vertex coordinates to the main coordinate system.
    tile_corner_vertices = np.dot((tile_corner_lattice_coordinates - grid_shifts), grid_vectors)

    # Structure to store vertices along with corresponding gridIndices
    tile_corner_vertices_info = []
    for point, grid_and_lines in unique_grid_line_intersection_points.items():
        # Calculate the corresponding index for the vector
        i = np.where((unique_intersections == point).all(axis=1))[0][0]
        grid_indices = [grid for grid, _ in grid_and_lines]
        # Add the vertex and corresponding gridIndices to the list
        tile_corner_vertices_info.append({'corner': tile_corner_vertices[i], 'gridIndices': grid_indices})

    # Create a list of final tiles
    tiles_data = []
    # Initialize a dictionary to store area-to-color mappings
    area_to_color = {}
    min_area = float('inf')
    max_area = float('-inf')

    for corner_info in tile_corner_vertices_info:
        corner = corner_info['corner']
        grid_indices = corner_info['gridIndices']
        vectors = [grid_vectors[i] for i in grid_indices]
        tile_vertices = []
        # All possible points for tile
        dots = generate_combinations(corner, vectors)
        # Determine if it is the central tile
        connected_tile = any(np.isclose(dot, np.array([0, 0])).all() for dot in dots)
        # Determine the boundary points of the tile
        if True:
            tile_vertices.append(filter_convex_hull_points(dots))
            # Calculate the area of the tile
            area = area_of_polygon(tile_vertices[0])
            min_area = min(min_area, area)
            max_area = max(max_area, area)
            tiles_data.append([tile_vertices, area, connected_tile])

    if connected_tiles:
        tiles_data = update_connected_components(tiles_data)
    else:
        tiles_data = [(tile_vertices, area, True) for tile_vertices, area, _ in tiles_data]

    # Add color
    tiles = []
    for tile_vertices, area, connected in tiles_data:
        if connected:
            # Check for area in the dictionary and select color
            if area not in area_to_color:
                area_to_color[area] = add_type_colors(area, min_area, max_area)
            tileColor = area_to_color[area]
            tiles.append((tile_vertices, tileColor))

    return tiles


def plot_tiles(grid_shifts, grid_range, connected_tiles, plot_range_limit):
    fig, ax = plt.subplots()

    tiles = create_tiles(grid_shifts, grid_range, connected_tiles)

    for tile_vertices, tileColor in tiles:
        for vertices in tile_vertices:
            polygon = patches.Polygon(vertices, closed=True, facecolor=tileColor, edgecolor='black')
            ax.add_patch(polygon)
    ax.set_xlim(-plot_range_limit, plot_range_limit)
    ax.set_ylim(-plot_range_limit, plot_range_limit)
    ax.set_aspect('equal')

    plt.show()


grid_shifts = [0.0, 0.0, 0.0, 0.0, 0.0]  # Modify shifts and dimensionality
grid_range = 5                            # Modify the number of grid lines
plot_range_limit = 17                     # Adjust the plot range
connected_tiles = False                    # Set to True to see only connected tiles

plot_tiles(grid_shifts, grid_range, connected_tiles, plot_range_limit)


