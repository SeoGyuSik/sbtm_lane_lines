import numpy as np
from practice import left_line, right_line
import cv2


def measure_curvature_real(left_fit, right_fit):
    # Conversions in x and y from pixels space to meters
    my = 30 / 720  # meters per pixel in y dimension
    mx = 3.7 / 700  # meters per pixel in x dimension
    ploty = np.linspace(0, 719, 720)
    left_fit_cc = np.array([0, 0, 0], dtype='float')
    right_fit_cc = np.array([0, 0, 0], dtype='float')
    left_fit_cc[0] = left_fit[0]*(mx/my**2)
    left_fit_cc[1] = left_fit[1]*(mx/my)
    left_fit_cc[2] = left_fit[2]*mx
    right_fit_cc[0] = right_fit[0] * (mx / my ** 2)
    right_fit_cc[1] = right_fit[1] * (mx / my)
    right_fit_cc[2] = right_fit[2] * mx
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Implement the calculation of R_curve (radius of curvature)
    left_curverad = (1 + (2 * left_fit_cc[0] * y_eval + left_fit_cc[1]) ** 2) ** (3 / 2) / np.abs(
        2 * left_fit_cc[0])
    right_curverad = (1 + (2 * right_fit_cc[0] * y_eval + right_fit_cc[1]) ** 2) ** (3 / 2) / np.abs(
        2 * right_fit_cc[0])

    # Measure the vehicle's distance from the center and its direction
    leftx_base = left_fit[0] * (719 ** 2) + left_fit[1] * 719 + left_fit[2]
    rightx_base = right_fit[0] * (719 ** 2) + right_fit[1] * 719 + right_fit[2]

    return left_curverad, right_curverad, leftx_base, rightx_base


def show_measurements(result):
    left_line.best_radius_of_curvature, right_line.best_radius_of_curvature, left_line.best_line_base_pos\
        , right_line.best_line_base_pos = measure_curvature_real(left_line.best_fit, right_line.best_fit)
    diff = (1280 / 2 - (left_line.best_line_base_pos + right_line.best_line_base_pos) / 2) * 3.7 / 700
    if diff >= 0:
        direction = 'right'
    else:
        direction = 'left'
    cv2.putText(result, "Radius of Curvature = " + str(
        int((left_line.best_radius_of_curvature + right_line.best_radius_of_curvature) / 2)) + "(m)", (30, 30),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    cv2.putText(result, 'Vehicle is ' + "%0.2f" % float(diff) + 'm ' + direction + ' of center', (30, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    return result






