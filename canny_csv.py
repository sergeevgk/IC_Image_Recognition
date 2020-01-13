import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# BGR color format
color_dict = {'blue': (255, 0, 0), 'green': (0, 255, 0),
              'red': (0, 0, 255), 'yellow': (0, 255, 255),
              'white': (255, 255, 255), 'black': (0, 0, 0),
              'magenta': (255, 0, 255), 'orange': (0, 128, 255)}


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def view_image(image, name_of_window="image"):
    cv.namedWindow(name_of_window, cv.WINDOW_NORMAL)
    cv.imshow(name_of_window, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def lapl_sobels(img, k_l=3, k_s=3):
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    l_im = cv.GaussianBlur(img, (3, 3), 0)
    g_im = cv.cvtColor(l_im, cv.COLOR_RGB2GRAY)
    laplacian = cv.Laplacian(g_im, cv.CV_64F, ksize=k_l)
    sobelx = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=k_s)
    sobely = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=k_s)

    plt.subplot(2, 2, 1), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()

    abs_dst_l = cv.convertScaleAbs(laplacian)
    abs_dst_s_x = cv.convertScaleAbs(sobelx)
    abs_dst_s_y = cv.convertScaleAbs(sobely)
    plt.figure()
    plt.subplot(2, 2, 1), plt.imshow(abs_dst_l, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(abs_dst_s_x, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(abs_dst_s_y, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()


class circle_params:
    def __init__(self, d, dist, p1, p2, r, R):
        self.d = d
        self.dist = dist
        self.p1 = p1
        self.p2 = p2
        self.r = r
        self.R = R


hough_circle_color = color_dict['green']
hough_circles_params = [circle_params(2, 200, 1, 30, 18, 23), circle_params(2, 200, 1, 30, 18, 23),
                        circle_params(2, 130, 1, 34, 3, 12), circle_params(2, 135, 1, 70, 20, 70),
                        circle_params(2, 135, 1, 30, 20, 70), circle_params(2, 95, 1, 32, 11, 24)]


def hough_circles(src, index):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # gray = src.copy()
    # gray = cv.medianBlur(gray, 5)
    # cv.imshow("gray", gray)
    c = hough_circles_params[index]
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, c.d, c.dist,
                              param1=c.p1, param2=c.p2,
                              minRadius=c.r, maxRadius=c.R)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, hough_circle_color, 1)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, hough_circle_color, 1)
            print(radius)
    # cv.namedWindow('detected circles', cv.WINDOW_NORMAL)
    # cv.resizeWindow('detected circles', 1240, 1080)
    cv.imshow("detected circles", src)


class line_params:
    def __init__(self, min_len, max_dist_from_other_lines, voices_num, derivation_from_horizontal_and_vertical_lines,
                 color):
        self.min_len = min_len
        self.dist = max_dist_from_other_lines
        self.thresh = voices_num
        self.derivation = derivation_from_horizontal_and_vertical_lines
        self.color = color


lines_params = [line_params(12, 2, 60, np.pi / 36, (0, 255, 0)),
                line_params(16, 3.5, 15, np.pi / 36, (0, 0, 255)),
                line_params(6, 2, 50, np.pi / 5, (0, 0, 255)),
                line_params(7, 1, 100, np.pi / 2, (0, 255, 0)),
                line_params(7, 1, 100, np.pi / 2, (0, 255, 0)),
                line_params(4, 1.1, 60, np.pi / 3, (0, 255, 0)),
                line_params(4, 1.1, 60, np.pi / 3, (0, 255, 0))]


def hough_lines(src, low, high):
    ratio = 1.5
    kernel_size = 3
    low_threshold = 0
    cv.imshow("dst", src)
    params = lines_params[threshold_color_index]
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, low, high)
    mask = thresh != 0
    res = src * (mask[:, :, None].astype(src.dtype))
    gray_image = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    cv.imshow("res", gray_image)
    img_blur = cv.blur(gray_image, (5, 5))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = res * (mask[:, :, None].astype(res.dtype))
    cv.imshow("dst", dst)
    minLineLength = params.min_len
    maxLineGap = params.dist
    # lines = cv.HoughLines(detected_edges, 1, np.pi / 180, 200, minLineLength, maxLineGap)
    # for i in range(len(lines)):
    #     for rho, theta in lines[i]:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         x1 = int(x0 + 100 * (-b))
    #         y1 = int(y0 + 100 * (a))
    #         x2 = int(x0 - 100 * (-b))
    #         y2 = int(y0 - 100 * (a))
    #         cv.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)
    lines = cv.HoughLinesP(detected_edges, 1, np.pi / 180, params.thresh, minLineLength=minLineLength,
                           maxLineGap=maxLineGap)
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            angle = angle_between((np.abs(x1 - x2), np.abs(y1 - y2)), (0, 1))
            if np.abs(angle) <= params.derivation or \
                    (np.pi / 2 + params.derivation >= np.abs(angle) >= np.pi / 2 -
                     params.derivation):
                cv.line(dst, (x1, y1), (x2, y2), params.color, 1)
    cv.imshow("detected lines", dst)


threshold_color_index = 4
thresholds = [[(97, 250, 0), (115, 255, 255)],  # blue
              [(25, 70, 118), (86, 255, 255)],  # green
              [(16, 130, 88), (39, 255, 255)],  # yellow
              [(0, 130, 195), (16, 255, 255)],  # red&orange
              [(0, 130, 195), (13, 255, 255)],  # red
              [(0, 74, 200), (17, 206, 255)],  # pink
              [(0, 1, 251), (255, 69, 255)],  # white
              [(0, 2, 250), (111, 2, 255)]]  # white


def findContour(img):
    # img0 = img.copy()
    # img1 = img.copy()
    # gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # hsv_min = np.array((0, 0, 255), np.uint8)
    # hsv_max = np.array((255, 255, 255), np.uint8)
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # binary = gray.copy()
    # # преобразуем одноканальное изображение в бинарное
    # cv.inRange(gray, 30, 250, binary)
    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # thresh = cv.inRange(hsv, hsv_min, hsv_max)
    #
    # contours, hierarchy = cv.findContours(binary.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img0, contours, -1, (0, 0, 0), 2, cv.LINE_AA, hierarchy, 1)
    # # print (hierarchy)
    # contours1, hierarchy1 = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # # print (hierarchy1)
    # for cnt in contours1:
    #     rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
    #     box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
    #     box = np.int0(box)  # округление координат
    #     center = (int(rect[0][0]), int(rect[0][1]))
    #     area = int(rect[1][0] * rect[1][1])  # вычисление площади
    #     if (area > 500):
    #         cv.drawContours(img1, [box], 0, (255, 0, 200), 2)  # рисуем прямоугольник
    #         cv.circle(img1, center, 5, (255, 0, 200), 2)  # рисуем маленький кружок в центре прямоугольника
    # cv.drawContours(img1, contours1, -1, (0, 0, 0), 2, cv.LINE_AA, hierarchy1, 1)
    # cv.imshow('contours', img0)
    # cv.imshow('contours1', img1)
    # cv.waitKey()
    # cv.destroyAllWindows()
    ratio = 1.5
    kernel_size = 5
    low_threshold = 0
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(gray_image, (5, 5))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = img * (mask[:, :, None].astype(img.dtype))

    low = thresholds[threshold_color_index][0]
    high = thresholds[threshold_color_index][1]

    hsv = cv.cvtColor(dst, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, low, high)
    mask = thresh != 0
    res = dst * (mask[:, :, None].astype(dst.dtype))
    # cv.imshow('mask result', res)
    # hough_circles(img)
    hough_circles(res, 5)
    cv.namedWindow('jpg', cv.WINDOW_NORMAL)
    cv.resizeWindow('jpg', 640, 480)
    cv.imshow("jpg", dst)
    # hough_lines(img, low, high)
    cv.waitKey(0)


def CannyThreshold(val):
    window_name = "Edge Map"
    # ---------------------
    minLineLength = 3
    maxLineGap = 50
    # ----------------------
    ratio = 1.5
    kernel_size = 5
    low_threshold = val
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(gray_image, (5, 5))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = img * (mask[:, :, None].astype(img.dtype))
    cv.imshow(window_name, dst)
    cv.imshow("8bit", detected_edges)


def color_trackbars():
    def nothing(*arg):
        pass

    cv.namedWindow("result")  # создаем главное окно
    cv.namedWindow("settings")  # создаем окно настроек

    cv.createTrackbar('h1', 'settings', 0, 255, nothing)
    cv.createTrackbar('s1', 'settings', 0, 255, nothing)
    cv.createTrackbar('v1', 'settings', 0, 255, nothing)
    cv.createTrackbar('h2', 'settings', 255, 255, nothing)
    cv.createTrackbar('s2', 'settings', 255, 255, nothing)
    cv.createTrackbar('v2', 'settings', 255, 255, nothing)
    crange = [0, 0, 0, 0, 0, 0]
    ratio = 1.5
    kernel_size = 5
    low_threshold = 0
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(gray_image, (5, 5))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = img * (mask[:, :, None].astype(img.dtype))
    while True:
        hsv = cv.cvtColor(dst, cv.COLOR_BGR2HSV)

        # считываем значения бегунков
        h1 = cv.getTrackbarPos('h1', 'settings')
        s1 = cv.getTrackbarPos('s1', 'settings')
        v1 = cv.getTrackbarPos('v1', 'settings')
        h2 = cv.getTrackbarPos('h2', 'settings')
        s2 = cv.getTrackbarPos('s2', 'settings')
        v2 = cv.getTrackbarPos('v2', 'settings')

        # формируем начальный и конечный цвет фильтра
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)

        # накладываем фильтр на кадр в модели HSV
        thresh = cv.inRange(hsv, h_min, h_max)

        cv.imshow('result', thresh)
        mask = thresh != 0
        img0 = dst * (mask[:, :, None].astype(dst.dtype))
        cv.imshow('result1', img0)
        ch = cv.waitKey(5)
        if ch == 27:
            break
    cv.destroyAllWindows()


def main():
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, threshold_image = cv.threshold(gray_image, 127, 255, 0)

    lapl_sobels(img, 5, 3)

    max_lowThreshold = 100
    window_name = 'Edge Map'
    title_trackbar = 'Min Threshold:'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)
    CannyThreshold(0)
    cv.waitKey()


# img = cv.imread("./IK_images/img_thermal_1575383371149.jpg")
img = cv.imread("./IK_images/img_thermal_1575383333609.jpg")
# img = cv.imread("./IK_images/sudoku.png")

# main()
# color_trackbars()
findContour(img)
