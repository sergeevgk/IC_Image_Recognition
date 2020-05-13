# Модуль для загрузки шаблонов и сопоставления изображений "границ" с шаблонами
import cv2 as cv
import numpy as np


def load_templates(path):
    templates = []
    return templates


def split_separate_objs(image):
    objects = []
    return objects


def fit_template(image, template):
    d = 0
    return d


def find_best_template(image):
    templates = load_templates('data/templates')
    fit_vals = []
    for t in templates:
        fit_vals.append(fit_template(image, t))


def solve_problem(N, basis_figures, figure_corners):
    template_id = -1
    min_var = np.Inf
    best_orientation = []
    best_template = []
    mean = 0

    for i in range(N):
        basis_figure_points = basis_figures[i]
        template = []
        for j in range(0, len(basis_figure_points) - 1, 2):
            template.append((basis_figure_points[j], basis_figure_points[j+1]))

        if len(template) != len(figure_corners):
            continue

        m = len(template)

        for k in range(m):
            figure_sides = []
            template_sides = []
            for j in range(m - 1):
                figure_sides.append(dist(figure_corners[j], figure_corners[j + 1]))
                template_sides.append(dist(template[j], template[j + 1]))
            figure_sides.append(dist(figure_corners[0], figure_corners[m - 1]))
            template_sides.append(dist(template[0], template[m - 1]))

            scale_ratio = np.array(figure_sides) / np.array(template_sides)
            if min_var > np.var(scale_ratio):
                template_id = i
                best_template = template
                min_var = np.var(scale_ratio)
                mean = np.mean(scale_ratio)
                best_orientation = figure_corners.copy()

            figure_corners = rotate_list(figure_corners, 1)

    scale = int(mean)

    rotation, shift = find_angle_and_shift(best_template, best_orientation, scale)

    return template_id, scale, rotation, int(shift[1]), int(shift[0])

def find_figure_corners(img):
    res = img.astype(np.uint8)
    ret, thresh = cv.threshold(res, 0.5, 1, 0)
    _, contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for x in range(res.shape[0]):
        for y in range(res.shape[1]):
            res[x, y] = 0
    cv.drawContours(res, contours, -1, 1)
    #plt.imshow(res, cmap='gray')
    #plt.show()
    lines = cv.HoughLinesP(res, 0.5, np.pi / 360, 10, minLineLength=5, maxLineGap=5)

    line_params = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        theta = np.arctan(-1/k)
        rho = b * np.sin(theta)
        line_params.append((rho, theta, x1, y1, x2, y2))
        #cv.line(res, (x1, y1), (x2, y2), (255, 255, 255), 2)
        #plt.imshow(res, cmap='gray')
        #plt.show()
    line_params = sorted(line_params, key = lambda x: (x[0], x[1]))

    lines_unique = []
    merged_line = []

    for i in range(len(line_params) - 1):
        rho1 = line_params[i][0]
        theta1 = line_params[i][1]
        rho2 = line_params[i+1][0]
        theta2 = line_params[i+1][1]
        x11 = line_params[i][2]
        y11 = line_params[i][3]
        x12 = line_params[i][4]
        y12 = line_params[i][5]
        x21 = line_params[i + 1][2]
        y21 = line_params[i + 1][3]
        x22 = line_params[i + 1][4]
        y22 = line_params[i + 1][5]
        if len(merged_line) == 0:
            merged_line = [x11, y11, x12, y12]

        if is_params_equals(rho1, theta1, rho2, theta2):
            x1m, y1m, x2m, y2m = merged_line
            x_min = np.min([x11, x12, x21, x22, x1m, x2m])
            x_max = np.max([x11, x12, x21, x22, x1m, x2m])
            y_min = np.min([y11, y12, y21, y22, y1m, y2m])
            y_max = np.max([y11, y12, y21, y22, y1m, y2m])
            if theta1 > 0:
                merged_line = [x_max, y_min, x_min, y_max]
            else:
                merged_line = [x_max, y_max, x_min, y_min]
        else:
            lines_unique.append(merged_line)
            merged_line = []

    if len(merged_line) != 0:
        lines_unique.append(merged_line)
    else:
        lines_unique.append(list(line_params[-1][2:6]))

    curr_point = (lines_unique[0][2], lines_unique[0][3])
    curr_line_id = 0
    corners = []
    visited = []

    while len(corners) != len(lines_unique):
        min_dist = np.Inf
        for i in range(len(lines_unique)):
            if i == curr_line_id:
                continue
            line = lines_unique[i]
            p1 = (line[0], line[1])
            p2 = (line[2], line[3])
            dist1 = dist(p1, curr_point)
            dist2 = dist(p2, curr_point)

            if p1 not in visited and dist1 < min_dist:
                min_dist = dist1
                next_point_id = 1
                curr_line_id = i
            if p2 not in visited and dist2 < min_dist:
                min_dist = dist2
                next_point_id = 0
                curr_line_id = i

        next_point = (lines_unique[curr_line_id][next_point_id * 2],
                      lines_unique[curr_line_id][next_point_id * 2 + 1])

        next_point_neighbour = (lines_unique[curr_line_id][(next_point_id - 1) * 2],
                                lines_unique[curr_line_id][(next_point_id - 1) * 2 + 1])

        corners.append((int((curr_point[1] + next_point_neighbour[1]) / 2),
                        int((curr_point[0] + next_point_neighbour[0]) / 2)))
        visited.append(next_point)
        curr_point = next_point

    return corners


def find_angle_and_shift(basis, fig, scale):
    best_alpha = 0
    best_shift = (0, 0)
    min_err = np.Inf

    for alpha in range(-180, 180, 1):
        for i in range(len(fig)):
            figure = fig.copy()

            transformed = scale_figure(basis, scale)
            transformed = rotate_figure(transformed, np.deg2rad(alpha))

            shift = (figure[i][0] - transformed[0][0],
                     figure[i][1] - transformed[0][1])

            transformed = shift_figure(transformed, shift)
            err = compare_figures(figure, transformed)

            if err < min_err:
                min_err = err
                best_alpha = alpha
                best_shift = shift

    return best_alpha, best_shift


# src - binary img with contours
def find_shapes_point_list(src):
    im = src.copy()
    shape_list = []

    q = []
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            if im[x, y] != 0:
                shape = [(x, y)]
                q.append((x, y))
                while len(q) != 0:
                    elem = q.pop()
                    for u in range(elem[0] - 1, elem[0] + 2):
                        for v in range(elem[1] - 1, elem[1] + 2):
                            if im[u, v] != 0:
                                im[u, v] = 0
                                if not ((u, v) in q):
                                    shape.append((u, v))
                                    q.append((u, v))
                shape_list.append(shape)

    return shape_list