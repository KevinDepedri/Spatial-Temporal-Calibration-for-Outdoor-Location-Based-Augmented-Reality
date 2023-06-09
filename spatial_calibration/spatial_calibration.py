import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def float_convertable(string):
    """Checks if a value can be converted to float/double"""
    try:
        float(string)
        return True
    except ValueError:
        return False


def get_point(lst, invert_x_z_retrieval=False, print_points=False):
    if invert_x_z_retrieval:
        coords = tuple(float(c) for c in lst[::-1])  # Here we extract column -1(column 2), then column 1, then column 0
    else:
        coords = tuple(float(c) for c in lst)  # Here we extract column 0,1,2

    if print_points:
        print(coords)
    return np.asarray(coords)


def get_points_list(row, invert_x_z_retrieval=False, print_points=False):
    """Utility function to get a certain number of points"""
    idx = 2  # Base index is two since column 1 and column 2 represents frame_number and time_stamp, we want to skip it
    points = []
    while (idx + 3) <= len(row):  # Each time that 3 columns are available (X,Y,Z) we can add a point to our list
        points.append(get_point(row[idx:idx + 3], invert_x_z_retrieval, print_points))
        idx += 3
    return np.asarray(points)


def compute_center_of_circumscribed_circle_in_xz(pa, pb, pc):
    """Finds X and Y center coordinates of a circumscribed circle based on 3 points on the circumference.
    NOTE: Ground in optitrack is parallel to the XZ plain, not XY
    See: https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates_2
    """
    D = 2 * (pa[0] * (pb[2] - pc[2]) + pb[0] * (pc[2] - pa[2]) + pc[0] * (pa[2] - pb[2]))
    U_x = ((pa[0] ** 2 + pa[2] ** 2) * (pb[2] - pc[2]) + (pb[0] ** 2 + pb[2] ** 2) * (pc[2] - pa[2]) + (
            pc[0] ** 2 + pc[2] ** 2) * (pa[2] - pb[2])) / D
    U_y = ((pa[0] ** 2 + pa[2] ** 2) * (pc[0] - pb[0]) + (pb[0] ** 2 + pb[2] ** 2) * (pa[0] - pc[0]) + (
            pc[0] ** 2 + pc[2] ** 2) * (pb[0] - pa[0])) / D
    return U_x, U_y


def plot_rigid_body(ax_, r_body, point_line_format, edge_label, vertex_label=False):
    """Plots all points of r_body"""
    for p1 in range(len(r_body)):
        for p2 in range(p1, len(r_body)):
            if not np.array_equal(p1, p2):
                # For each possible combination of points, draw the points in [X, Y, Z] and a line between them
                ax_.plot([r_body[p1][0], r_body[p2][0]],
                         [r_body[p1][1], r_body[p2][1]],
                         [r_body[p1][2], r_body[p2][2]], point_line_format, label=edge_label + f"_{p1}-{p2}")
        if vertex_label:
            ax.text(r_body[p1][0], r_body[p1][1], r_body[p1][2], f"{p1}")


def find_normal_vector(p1, p2, p3):
    """Finds normal vector to a plain defined by points p1, p2, p3"""
    v1 = p2 - p1
    v2 = p3 - p1
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm:
        normal = normal / norm
        return normal
    return normal


def find_vectors(r_body, points_for_normal=None, points_for_parallel=None, invert_x_y_computation=False):
    """Returns 3 directional vectors for a rigid body, here, normally:
       - Vector y is computed as a vector perpendicular to the plane build using the points of the rigid
       body in position "points_for_normal"
       - Vector x is computed as a vector parallel to the vector based on the points of the rigid body in position
       "points_for_parallel"
       - Vector z is computed as a vector perpendicular to the plane build by the previously computed vectors y and x

       If the flag "invert_x_y_computation" is enabled, then x and y computation are inverted, meaning that x will be
       computed as the vector perpendicular to the plane ... while y will be built as the vector parallel to the vector
       based on ...
       """
    assert len(points_for_normal) == 3, "Error, 3 points indexes are needed to build a plane and project its normal"
    assert len(points_for_parallel) == 2, "Error, 2 points indexes are needed to build a vector and get its parallel"

    if invert_x_y_computation:
        v_y = -(r_body[points_for_parallel[0]] - r_body[points_for_parallel[1]]) \
              / (np.linalg.norm(r_body[points_for_parallel[0]] - r_body[points_for_parallel[1]]))
        v_x = find_normal_vector(r_body[points_for_normal[0]], r_body[points_for_normal[1]], r_body[points_for_normal[2]])
        v_z = np.cross(v_x, v_y)
        v_z = v_z / (np.linalg.norm(v_z))
    else:
        v_y = -find_normal_vector(r_body[points_for_normal[0]], r_body[points_for_normal[1]], r_body[points_for_normal[2]])
        v_x = (r_body[points_for_parallel[0]] - r_body[points_for_parallel[1]]) \
              / (np.linalg.norm(r_body[points_for_parallel[0]] - r_body[points_for_parallel[1]]))
        v_z = np.cross(v_x, v_y)
        v_z = v_z / (np.linalg.norm(v_z))
    return v_x, v_y, v_z


def plot_vectors(ax, ref_point, vectors, scale, colour_scheme="rgb"):
    """Plots a triplet of directional vectors from a point ref_point. Scale is used to increase visibility of vectors"""
    ref_point_x = ref_point + vectors[0] * scale
    ref_point_y = ref_point + vectors[1] * scale
    ref_point_z = ref_point + vectors[2] * scale

    ax.plot([ref_point[0], ref_point_x[0]], [ref_point[1], ref_point_x[1]],
            [ref_point[2], ref_point_x[2]], colour_scheme[0] + "s:")
    ax.plot([ref_point[0], ref_point_y[0]], [ref_point[1], ref_point_y[1]],
            [ref_point[2], ref_point_y[2]], colour_scheme[1] + "s:")
    ax.plot([ref_point[0], ref_point_z[0]], [ref_point[1], ref_point_z[1]],
            [ref_point[2], ref_point_z[2]], colour_scheme[2] + "s:")


def compute_rotation_matrix_from_vectors(unit_vectors_set_1, unit_vectors_set_2):
    """Finds a transformation matrix between 2 triplets of unit vectors"""
    A = np.zeros((9, 9))
    for vector_idx, vector in enumerate(unit_vectors_set_1):
        for vector_component_idx, vector_component in enumerate(vector):
            A[vector_idx * 3][vector_component_idx] = vector_component
            A[vector_idx * 3 + 1][vector_component_idx + 3] = vector_component
            A[vector_idx * 3 + 2][vector_component_idx + 6] = vector_component
    b = np.concatenate(unit_vectors_set_2)
    t = np.linalg.solve(A, b)
    t = np.reshape(t, (3, 3))
    return t


def rotation_matrix(s1, s2):
    A = np.column_stack(s1)
    A_inv = np.linalg.inv(A)
    B = np.column_stack(s2)
    M = np.matmul(B, A_inv)
    return M


def rotation_matrix_between_systems(R_SC1, R_SC2):
    R_CS1 = np.linalg.inv(R_SC1)
    R = np.matmul(R_SC2, R_CS1)
    return R


def compute_rotation_matrix_from_quaternion(q):
    """Returns rotation matrix(orientation) from quaternions"""
    matrix = 2 * np.array([
        [q[0] ** 2 + q[1] ** 2 - 0.5, q[1] * q[2] - q[0] * q[3], q[0] * q[2] + q[1] * q[3]],
        [q[0] * q[3] + q[1] * q[2], q[0] ** 2 + q[2] ** 2 - 0.5, q[2] * q[3] - q[0] * q[1]],
        [q[1] * q[3] - q[0] * q[2], q[0] * q[1] + q[2] * q[3], q[0] ** 2 + q[3] ** 2 - 0.5]
    ])
    return matrix


plot_computed_vectors = True
plot_optitrack_vectors = False

if __name__ == "__main__":
    # TODO: Run it multple times in a loop to average out the transformation matrices for different optitrack acquisitions?
    # PART 1
    # Following code is used to compute rigid bodies orientation and the transformation matrix using optitrack positions
    print("COMPUTATIONS BASED ON VECTORS MANUALLY COMPUTED FROM POINTS")
    stonex = []
    smartphone = []
    with open("Calibration_stonex_smartphone_XYZ.csv") as csvfile:
        points_reader = csv.reader(csvfile)
        for row in points_reader:
            if row and float_convertable(row[-1]):  # If the row is composed of float (just check the last element)
                # Skip index (frame_number, time_stamp) and extract all available points
                points = get_points_list(row, invert_x_z_retrieval=False, print_points=False)
                stonex = points[2:7]  # Elements 0 and 1: position and rotation of stonex rigid body, we skip them
                print(f"Stonex points:\n{stonex}]")
                smartphone = points[9:13]  # Element 7 and 8: position and rotation of smartphone rigid body, skip them
                print(f"Smartphone points:\n{smartphone}]\n")
                break  # We assume the object to be static, so we just take the points at the first valid frame

    # Apply the mpl "TkAgg" backend, generate the figure and add a subplot
    mpl.use("TkAgg")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Get the position of the stonex barycenter on the XZ plane (plane parallel to the ground)
    stonex_x_center, stonex_z_center = compute_center_of_circumscribed_circle_in_xz(stonex[0], stonex[1], stonex[2])
    stonex_y_center = ((stonex[0][1] + stonex[1][1] + stonex[2][1]) / 3) - 0.01
    # Append the barycenter in the list of stonex points, using the format [X,Y,Z]
    stonex = np.append(stonex, np.array([[stonex_x_center,
                                          stonex_y_center,
                                          stonex_z_center]]), axis=0)

    # Define the limits over the 3 dimensions of the initial plot, then add a label on each axe
    ax.axes.set_xlim3d(left=-0.2, right=0.2)
    ax.axes.set_ylim3d(bottom=1.6, top=2)
    ax.axes.set_zlim3d(bottom=0.05, top=0.4)
    ax.axes.set_xlabel("x")
    ax.axes.set_ylabel("y")
    ax.axes.set_zlabel("z")

    # Plot all the stonex edges using black (k) circle marker (o) with solid lines (-) between them
    plot_rigid_body(ax, stonex, "ko-", "Stonex", True)

    # Define the stonex reference point, compute the orientation vectors, define their length and print them
    stonex_ref_point = stonex[5]  # For the stonex we take the center of phase as reference point
    stonex_vx, stonex_vy, stonex_vz = find_vectors(stonex, points_for_normal=(0, 1, 2),
                                                   points_for_parallel=(0, 3), invert_x_y_computation=False)
    vectors_length = np.linalg.norm(stonex[1] - stonex_ref_point)*2
    print(f"Stonex vectors:\n{np.array([stonex_vx, stonex_vy, stonex_vz]).transpose()}")
    # Finally plot the vectors with origin in the reference point of the rigid body
    if plot_computed_vectors:
        plot_vectors(ax, stonex_ref_point, [stonex_vx, stonex_vy, stonex_vz], vectors_length)

    # Plot all the smartphone edges using black (k) circle marker (o) with solid lines (-) between them
    plot_rigid_body(ax, smartphone, "ko-", "smartphone", True)

    # Define the smartphone reference point, compute the orientation vectors and print them
    smartphone_ref_point = smartphone[2]  # For the smartphone we take the main camera as reference point
    smartphone_vx, smartphone_vy, smartphone_vz = find_vectors(smartphone, points_for_normal=(0, 1, 3),
                                                               points_for_parallel=(1, 0), invert_x_y_computation=True)
    print(f"Smartphone vectors:\n{np.array([smartphone_vx, smartphone_vy, smartphone_vz]).transpose()}\n")
    # Finally plot the vectors with origin in the reference point of the rigid body
    if plot_computed_vectors:
        plot_vectors(ax, smartphone_ref_point, [smartphone_vx, smartphone_vy, smartphone_vz], vectors_length)

    # Find the rotation matrix between smartphone and stonex
    smart_to_stone_rm = compute_rotation_matrix_from_vectors([smartphone_vx, smartphone_vy, smartphone_vz],
                                                             [stonex_vx, stonex_vy, stonex_vz])
    print(f"ROTATION MATRIX [SMARTPHONE-STONEX]:\n{smart_to_stone_rm}\n")
    a = rotation_matrix([smartphone_vx, smartphone_vy, smartphone_vz], [stonex_vx, stonex_vy, stonex_vz])
    print(f"NEW ROTATION MATRIX [SMARTPHONE-STONEX]:\n{a}\n")

    # Verify the quality of the rotation matrix
    estimated_stonex_vectors = np.matmul(smart_to_stone_rm, np.array([smartphone_vx, smartphone_vy, smartphone_vz]).transpose())
    print(f"Estimated stonex vectors through rotation matrix:\n{estimated_stonex_vectors}")
    print(f"Error on estimate:\n{np.array([stonex_vx, stonex_vy, stonex_vz]).transpose() - estimated_stonex_vectors}\n")

    # Find the inverse rotation matrix between stonex and smartphone
    stone_to_smart_rm = np.linalg.inv(smart_to_stone_rm)
    print(f"INVERSE ROTATION MATRIX [STONEX-SMARTPHONE]:\n{stone_to_smart_rm}\n")

    # Verify the quality of the inverse rotation matrix
    estimated_smartphone_vectors = np.matmul(stone_to_smart_rm, np.array([stonex_vx, stonex_vy, stonex_vz]).transpose())
    print(f"Estimated smartphone vectors through inverse rotation matrix:\n{estimated_smartphone_vectors}")
    print(f"Error on estimate:\n{np.array([smartphone_vx, smartphone_vy, smartphone_vz]).transpose() - estimated_smartphone_vectors}\n")

    # Find the translation vector between smartphone and stonex and between stonex to smartphone
    smart_to_stone_tv = (stonex_ref_point - smartphone_ref_point).transpose()
    stone_to_smart_tv = (smartphone_ref_point - stonex_ref_point).transpose()
    print(f"Translation vector from smartphone to stonex:\n{smart_to_stone_tv}")
    print(f"Translation vector from stonex to smartphone:\n{stone_to_smart_tv}\n")

    # Put together rotation matrix and translation vector to build the final transformation matrix in both directions
    smart_to_stone_rotation = R.from_matrix(smart_to_stone_rm)
    smart_to_stone_rotation_quaternion = smart_to_stone_rotation.as_quat()
    smart_to_stone_rotation_euler = smart_to_stone_rotation.as_euler("xyz", degrees=True)
    smart_to_stone_tm = np.c_[smart_to_stone_rm, np.array(smart_to_stone_tv)]
    smart_to_stone_tm = np.append(smart_to_stone_tm, [np.array([0, 0, 0, 1])], axis=0)
    print(f"FINAL TRANSFORMATION MATRIX [SMARTPHONE-STONEX](rotation + translation):\n{smart_to_stone_tm}")
    print(f"FINAL XYZW QUATERNION ROTATION [SMARTPHONE-STONEX]:\n{smart_to_stone_rotation_quaternion}")
    print(f"FINAL XYZ EULER ROTATION [SMARTPHONE-STONEX]:\n{smart_to_stone_rotation_euler}\n")

    stone_to_smart_rotation = R.from_matrix(stone_to_smart_rm)
    stone_to_smart_rotation_quaternion = stone_to_smart_rotation.as_quat()
    stone_to_smart_rotation_euler = stone_to_smart_rotation.as_euler("XYZ", degrees=True)
    stone_to_smart_tm = np.c_[stone_to_smart_rm, np.array(stone_to_smart_tv)]
    stone_to_smart_tm = np.append(stone_to_smart_tm, [np.array([0, 0, 0, 1])], axis=0)
    print(f"FINAL INVERSE TRANSFORMATION MATRIX [STONEX-SMARTPHONE](rotation + translation):\n{stone_to_smart_tm}")
    print(f"FINAL INVERSE XYZW QUATERNION ROTATION [STONEX-SMARTPHONE]:\n{stone_to_smart_rotation_quaternion}")
    print(f"FINAL INVERSE XYZ EULER ROTATION [STONEX-SMARTPHONE]:\n{stone_to_smart_rotation_euler}\n\n")

    # PART 2
    # Following code is used to compare Optitrack inherited vector coordinates with the ones computed above
    print("COMPUTATIONS BASED ON OPTITRACK QUATERNIONS")
    stonex_quaternion = None
    stonex_opt_center = None
    smartphone_quaternion = None
    smartphone_opt_center = None
    with open("Calibration_stonex_smartphone_QUATERNION.csv") as csvfile2:
        reader = csv.reader(csvfile2)
        for row in reader:
            if row and float_convertable(row[-1]):
                stonex_quaternion = np.array(get_point(row[5:6] + row[2:5]))
                stonex_opt_center = np.asarray(get_point(row[6:9]))
                smartphone_quaternion = np.array(get_point(row[27:28] + row[24:27]))
                smartphone_opt_center = np.asarray(get_point(row[28:31]))
                break

    print(f"Stonex quaternion: {stonex_quaternion}")
    print(f"Stonex center by Optitrack: {stonex_opt_center}")
    stonex_matrix = compute_rotation_matrix_from_quaternion(stonex_quaternion)
    stonex_rotation = R.from_matrix(stonex_matrix)
    stonex_rotation_quaternion = stonex_rotation.as_quat()
    stonex_rotation_euler = stonex_rotation.as_euler("XYZ", degrees=True)
    print(f"Stonex rotation matrix:\n{stonex_matrix}")
    print(f"Stonex quaternion rotation:\n{stonex_rotation_quaternion}")
    print(f"Stonex euler rotation:\n{stonex_rotation_euler}\n")

    print(f"Smartphone quaternion: {smartphone_quaternion}")
    print(f"Smartphone center by Optitrack: {smartphone_opt_center}")
    smartphone_matrix = compute_rotation_matrix_from_quaternion(smartphone_quaternion)
    smartphone_rotation = R.from_matrix(smartphone_matrix)
    smartphone_rotation_quaternion = smartphone_rotation.as_quat()
    smartphone_rotation_euler = smartphone_rotation.as_euler("XYZ", degrees=True)
    print(f"Smartphone rotation matrix:\n{smartphone_matrix}")
    print(f"Smartphone quaternion rotation:\n{smartphone_rotation_quaternion}")
    print(f"Smartphone euler rotation:\n{smartphone_rotation_euler}\n")

    stone_to_smart_euler_difference = stonex_rotation_euler - smartphone_rotation_euler
    rotation = R.from_euler("xyz", stone_to_smart_euler_difference, degrees=True)
    print(f"Stonex-Smartphone rotation matrix:\n{rotation.as_matrix()}")
    print(f"Stonex-Smartphone quaternion difference:\n{rotation.as_quat()}")
    print(f"Stonex-Smartphone euler difference:\n{rotation.as_euler('xyz', degrees=True)}\n")

    rotation_matrix_between = rotation_matrix_between_systems(smartphone_matrix, stonex_matrix)
    rotation_between = R.from_matrix(rotation_matrix_between)
    print(f"Difference between rotation matrices:\n{rotation_matrix_between}")
    print(f"Difference between quaternions:\n{rotation_between.as_quat()}")
    print(f"Difference between eulers:\n{rotation_between.as_euler('xyz', degrees=True)}")


    # ax.plot(stonex_opt_center[0], stonex_opt_center[1], stonex_opt_center[2], "yo-")
    # ax.plot(smartphone_opt_center[0], smartphone_opt_center[1], smartphone_opt_center[2], "yo-")
    # identity_matrix = np.identity(3)
    # stonex_orientation = np.matmul(stonex_matrix, identity_matrix)
    # smartphone_orientation = np.matmul(smartphone_matrix, identity_matrix)
    # print(f"Matmul stonex matrix on identity:\n{stonex_orientation}")
    # print(f"Matmul smartphone matrix on identity:\n{smartphone_orientation}")

    if plot_optitrack_vectors:
        plot_vectors(ax, stonex_ref_point,
                     [stonex_matrix.transpose()[0],
                      stonex_matrix.transpose()[1],
                      stonex_matrix.transpose()[2]],
                     np.linalg.norm(stonex[1] - stonex_ref_point) * 2,
                     colour_scheme="cmy")

        plot_vectors(ax, smartphone_ref_point,
                     [smartphone_matrix.transpose()[0],
                      smartphone_matrix.transpose()[1],
                      smartphone_matrix.transpose()[2]],
                     np.linalg.norm(stonex[1] - stonex_ref_point) * 2,
                     colour_scheme="cmy")

    if plot_computed_vectors or plot_optitrack_vectors:
        ax.view_init(elev=-20, azim=0, roll=90)
        plt.show()
