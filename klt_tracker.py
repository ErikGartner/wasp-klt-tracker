import numpy as np
from scipy import ndimage, interpolate

from matplotlib.pyplot import imshow, imread
import matplotlib.pyplot as plt
from PIL import Image


def differentiate(img_arr):
    X_FILTER = np.array([[3.0, 0.0, -3.0], [10.0, 0.0, 10.0], [3.0, 0.0, -3.0]])
    Y_FILTER = X_FILTER.T

    x_diff = ndimage.convolve(img_arr, X_FILTER)
    y_diff = ndimage.convolve(img_arr, Y_FILTER)

    #x_diff, y_diff = np.gradient(img_arr)

    return (x_diff, y_diff)


def estimate_Z(x_diff, y_diff):
    x_diff = x_diff.flatten()
    y_diff = y_diff.flatten()

    grads_T = np.array([x_diff, y_diff])
    Z = np.dot(grads_T, grads_T.T)
    return Z


def estimate_e(I, J, Ix, Iy):
    D = (J - I).flatten()
    Ix = Ix.flatten()
    Iy = Iy.flatten()

    grads_T = np.array([Ix, Iy])

    return -1 * np.dot(grads_T, D)


def interpolate_region(img_array):
    """
    Creates an bilinear interpolation of the image area sent in.
    Evaluate using spline(x, y)
    """
    x_arr = np.arange(0, img_array.shape[1])
    y_arr = np.arange(0, img_array.shape[0])
    interp = interpolate.interp2d(x_arr, y_arr, img_array, kind='linear')
    #spline = interpolate.RectBivariateSpline(x_arr, y_arr, img_array, kx=1, ky=1)
    return interp


def calc_d(I, J, x, y, win_size, max_iter, min_disp):

    it = 0
    d_tot = np.array([0., 0.]).T

    # The window to evaluate
    win_x = np.arange(x, x + win_size[0], dtype=np.float32)
    win_y = np.arange(y, y + win_size[1], dtype=np.float32)

    template = I[x:x + win_size[0], y: y + win_size[1]]

    # Find image gradient in I
    Ix, Iy = differentiate(template)

    Z = estimate_Z(Ix, Iy)
    Zinv = np.linalg.inv(Z)

    # Create interpolated versions the new frame
    J_inter = interpolate_region(J)

    while it < max_iter:
        it += 1

        # Get the current window
        J_win = J_inter(win_x + d_tot[0], win_y + d_tot[1])

        e = estimate_e(template, J_win, Ix, Iy)
        d = np.dot(Zinv, e)

        d_tot = d_tot + d

        if np.hypot(d[0], d[1]) <= min_disp:
            # Check if converged
            return d_tot

    return d_tot


def calc_klt(old_image, new_image, input_points, win_size=(21, 21), max_iter=30, min_disp=0.01):

    output_points = []
    for (x, y) in input_points:
        d = calc_d(old_image, new_image, x, y, win_size, max_iter, min_disp)
        output_points.append((x + d[0], y + d[1]))
    return output_points
