import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_naca0012(num_points=100, chord=1.0):
    """
    Generates coordinates for a NACA 0012 airfoil.

    Args:
        num_points (int): The number of points to generate along the airfoil surface.  Must be >= 2.
                         Increasing this gives a smoother airfoil.  An even number is recommended.
        chord (float): The chord length of the airfoil.

    Returns:
        numpy.ndarray: A NumPy array of shape (N, 2) containing the (x, y)
                       coordinates of the airfoil. The points are ordered from
                       the trailing edge, around the upper surface, to the leading
                       edge, and then back around the lower surface to the
                       trailing edge.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    # Create linearly spaced x-coordinates from 0 to chord
    x = np.linspace(0, chord, num_points // 2)  # Generate points for one surface

    # NACA 0012 thickness distribution formula
    t = 0.12  # Maximum thickness (12% of chord)
    yt = 5 * t * chord * (0.2969 * np.sqrt(x / chord) - 0.1260 * (x / chord) -
                         0.3516 * (x / chord)**2 + 0.2843 * (x / chord)**3 -
                         0.1036 * (x / chord)**4)   #Corrected the last coefficient to -0.1036.  -0.1015 is also acceptable.


    # For a symmetrical airfoil (00xx), the camber line is y=0.
    # The upper and lower surfaces are simply +/- the thickness distribution.
    xu = x
    yu = yt
    xl = x
    yl = -yt

    # Combine upper and lower surfaces, reversing the lower surface to create a closed loop
    x_coords = np.concatenate((xu[::-1], xl[1:]))  # Reverse upper, skip first point of lower
    y_coords = np.concatenate((yu[::-1], yl[1:])) # to avoid duplicate points.

    return np.column_stack((x_coords, y_coords))

def rotate_airfoil(airfoil_coords, angle_degrees, rotation_center):
    """
    Rotates 2D airfoil coordinates around a specified center.

    Args:
        airfoil_coords: A NumPy array of shape (N, 2) representing the airfoil
                         coordinates (x, y).
        angle_degrees: The rotation angle in degrees (positive for
                       counterclockwise rotation).
        rotation_center: A list or NumPy array of shape (2,) representing the
                         center of rotation [x_center, y_center].

    Returns:
        A NumPy array of shape (N, 2) representing the rotated airfoil
        coordinates.
    """

    # Convert angle to radians
    angle_radians = np.deg2rad(angle_degrees)

    # Create rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])

    # Translate coordinates to origin (subtract rotation center)
    translated_coords = airfoil_coords - np.array(rotation_center)

    # Rotate the translated coordinates
    rotated_translated_coords = np.dot(translated_coords, rotation_matrix)

    # Translate back to the original position (add rotation center)
    rotated_coords = rotated_translated_coords + np.array(rotation_center)

    return rotated_coords


def plot_airfoil(airfoil_coords, title="Airfoil"):
    """Plots the airfoil coordinates."""
    plt.figure(figsize=(6, 4))  # Adjust figure size for better aspect ratio
    plt.plot(airfoil_coords[:, 0], airfoil_coords[:, 1])
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')  # Ensure x and y axes have the same scale
    plt.grid(True)
    plt.show()


def generate_naca4412(num_points=100, chord=1.0):
    """Creates a simple example airfoil (NACA 4412)."""
    t = 0.12  # Maximum thickness
    m = 0.04  # Maximum camber
    p = 0.4   # Location of maximum camber
    c = chord  # Chord length

    num_points = 100  # Number of points to generate
    x = np.linspace(0, c, num_points)

    # Camber line
    yc = np.where(x <= p * c,
                  (m / p**2) * (2 * p * x / c - (x / c)**2),
                  (m / (1 - p)**2) * (1 - 2 * p + 2 * p * x / c - (x / c)**2))

    # Thickness distribution
    yt = (t / 0.2) * (0.2969 * np.sqrt(x / c) - 0.1260 * (x / c) - 0.3516 * (x / c)**2 + 0.2843 * (x / c)**3 - 0.1015 * (x / c)**4) #Original Equation
    #yt = (t / 0.2) * (0.2969 * np.sqrt(x / c) - 0.1260 * (x / c) - 0.3516 * (x / c)**2 + 0.2843 * (x / c)**3 - 0.1036 * (x / c)**4) #Corrected Equation

    # Angle of the camber line
    theta = np.arctan(np.gradient(yc, x)) #The dyc/dx, we use the central method

    # Upper and lower surfaces
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Combine upper and lower surfaces, reversing the lower surface order
    x_coords = np.concatenate((xu, xl[::-1]))
    y_coords = np.concatenate((yu, yl[::-1]))

    return np.column_stack((x_coords, y_coords))


if __name__ == '__main__':
    airfoil_type = 'naca-0012'

    if airfoil_type == 'naca-0012':
        # Example usage with NACA 4412 airfoil
        airfoil_coordinates = generate_naca0012(num_points=500, chord=1.0)
    elif airfoil_type == 'naca-4412':
        # Example usage with NACA 0012 airfoil
        airfoil_coordinates = generate_naca4412(num_points=500, chord=1.0)

    rotation_center = [0.5, 0.0]
    rotation_angle = 40  # degrees
    chord_length = 8.0

    rotated_airfoil = rotate_airfoil(airfoil_coordinates, rotation_angle, rotation_center)

    output_file = airfoil_type + '_AoA-' + str(rotation_angle) + 'deg_chord-' + str(chord_length) 
    output_folder = '/aia/r016/scratch/clagemann/maia_ml/maiaGym/maiaml/development/maia_testing/stl'

    # Plot the original and rotated airfoils
    plt.figure(figsize=(8, 6))
    plt.plot(airfoil_coordinates[:, 0], airfoil_coordinates[:, 1], label='Original')
    plt.plot(rotated_airfoil[:, 0], rotated_airfoil[:, 1], label='Rotated')
    plt.scatter(rotation_center[0], rotation_center[1], color='red', marker='x', label='Rotation Center') #Mark the rotation center
    plt.title('Airfoil Rotation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_folder, output_file + '.png'))
    # plt.show()

    output_file = airfoil_type + '_AoA-' + str(rotation_angle) + 'deg_chord-' + str(chord_length) 
    output_folder = '/aia/r016/scratch/clagemann/maia_ml/maiaGym/maiaml/development/maia_testing/stl'
    np.savetxt(os.path.join(output_folder, output_file + '.stl'), rotated_airfoil * chord_length, header=str(len(rotated_airfoil)), delimiter='	', fmt='%10.18f')