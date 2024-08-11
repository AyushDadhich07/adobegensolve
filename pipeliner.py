import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon, LineString
import pandas as pd

def read_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)
    path_XYs = []
    for shape_id in df[0].unique():
        shape_df = df[df[0] == shape_id]
        XYs = []
        for path_id in shape_df[1].unique():
            path_df = shape_df[shape_df[1] == path_id]
            XY = path_df[[2, 3]].values
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot_shapes(path_XYs, title, symmetry_lines=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 10))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
    for i, XYs in enumerate(path_XYs):
        c = colors[i % len(colors)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    
    if symmetry_lines:
        for line in symmetry_lines:
            ax.plot(line[:, 0], line[:, 1], 'k--', linewidth=1)
    
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.show()

def detect_symmetry(points, num_axes=8):
    center = np.mean(points, axis=0)
    angles = np.linspace(0, np.pi, num_axes)
    
    symmetry_scores = []
    for angle in angles:
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        reflected_points = np.dot(points - center, rotation) * [-1, 1] + center
        score = np.sum(cdist(points, reflected_points).min(axis=1))
        symmetry_scores.append((angle, score))
    
    symmetry_scores.sort(key=lambda x: x[1])
    
    if len(symmetry_scores) > 1 and symmetry_scores[1][1] < symmetry_scores[0][1] * 1.5:
        return symmetry_scores[:2]
    else:
        return [symmetry_scores[0]]

def detect_shape(points):
    poly = Polygon(points)
    
    # Check for circle
    min_radius = np.min(cdist(points, [np.mean(points, axis=0)]))
    max_radius = np.max(cdist(points, [np.mean(points, axis=0)]))
    if max_radius - min_radius < 0.1 * max_radius:
        return "Circle"
    
    # Check for regular polygon
    n_sides = len(points)
    if n_sides >= 3:
        angles = np.diff(np.arctan2(points[:, 1] - poly.centroid.y, points[:, 0] - poly.centroid.x))
        angles = np.abs(angles)
        if np.allclose(angles, angles[0], rtol=0.1):
            if n_sides == 3:
                return "Triangle"
            elif n_sides == 4:
                return "Square" if poly.minimum_rotated_rectangle.area / poly.area > 0.95 else "Rectangle"
            elif n_sides == 5:
                return "Pentagon"
            elif n_sides == 6:
                return "Hexagon"
            else:
                return f"{n_sides}-gon"
    
    return "Unknown"

def complete_curve(points, shape_type):
    if shape_type == "Circle":
        center = np.mean(points, axis=0)
        radius = np.mean(np.linalg.norm(points - center, axis=1))
        theta = np.linspace(0, 2*np.pi, 100)
        completed_curve = np.column_stack((radius * np.cos(theta) + center[0], 
                                           radius * np.sin(theta) + center[1]))
    elif shape_type in ["Square", "Rectangle"]:
        poly = Polygon(points)
        min_rot_rect = poly.minimum_rotated_rectangle
        completed_curve = np.array(min_rot_rect.exterior.coords)
    else:
        # Use a simple interpolation for unknown shapes
        tck, u = splprep(points.T, s=0, per=True)
        completed_curve = np.column_stack(splev(np.linspace(0, 1, 100), tck))
    
    return completed_curve

def analyze_shapes(csv_path):
    path_XYs = read_csv(csv_path)
    all_symmetry_lines = []
    completed_path_XYs = []
    
    for i, XYs in enumerate(path_XYs):
        completed_XYs = []
        for j, XY in enumerate(XYs):
            print(f"Shape {i}, Path {j}:")
            
            # Detect symmetry
            symmetries = detect_symmetry(XY)
            for k, (angle, score) in enumerate(symmetries):
                print(f"  - Symmetry axis {k+1}: angle = {angle:.2f} radians, score = {score:.2f}")
                
                # Calculate symmetry line for plotting
                center = np.mean(XY, axis=0)
                direction = np.array([np.cos(angle), np.sin(angle)])
                line_start = center - direction * 100
                line_end = center + direction * 100
                all_symmetry_lines.append(np.array([line_start, line_end]))
            
            # Detect shape
            shape_type = detect_shape(XY)
            print(f"  - Detected shape: {shape_type}")
            
            # Complete curve if necessary
            if shape_type != "Unknown" and len(XY) < 100:  # Arbitrary threshold for "incompleteness"
                completed_curve = complete_curve(XY, shape_type)
                completed_XYs.append(completed_curve)
                print("  - Curve completed")
            else:
                completed_XYs.append(XY)
            
            print()
        
        completed_path_XYs.append(completed_XYs)
    
    # Plot the results
    plot_shapes(path_XYs, f"Original: {csv_path}")
    plot_shapes(completed_path_XYs, f"Completed: {csv_path}", symmetry_lines=all_symmetry_lines)
    
    return completed_path_XYs

def save_csv(path_XYs, output_path):
    data = []
    for shape_id, XYs in enumerate(path_XYs):
        for path_id, XY in enumerate(XYs):
            for point_id, (x, y) in enumerate(XY):
                data.append([shape_id, path_id, point_id, x, y])
    
    df = pd.DataFrame(data, columns=['shape_id', 'path_id', 'point_id', 'x', 'y'])
    df.to_csv(output_path, index=False)

# Main pipeline
input_csv = 'frag1.csv'
output_csv = 'frag1_solution.csv'

completed_shapes = analyze_shapes(input_csv)
save_csv(completed_shapes, output_csv)
print(f"Solution saved to {output_csv}")