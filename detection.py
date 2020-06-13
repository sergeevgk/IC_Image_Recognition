import cv2 as cv
import numpy as np
from skimage.morphology import *
from skimage import color
import matplotlib.pyplot as plt
from utility import get_area, get_center, get_length, get_width_height


def detect_hottest_parts(image, name):
    fig, ax = plt.subplots(3, 2, figsize=(30, 20))
    gray_im = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # gray_im = 255 - gray_im
    k = 0
    all_contours = []
    level_imgs = {'lvl': int, 'image': []}
    for low in (65, 75, 85, 95):
        temp = gray_im.copy()
        ret, thresh1 = cv.threshold(gray_im, low, 125, cv.THRESH_BINARY)
        thresh1 = binary_closing(thresh1, selem=np.ones((3, 3)))
        # ax.flatten()[0].imshow(thresh1)
        thresh1 = binary_erosion(thresh1, selem=np.ones((5, 5)))
        thresh1 = binary_closing(thresh1, selem=np.ones((4, 4)))
        # ax.flatten()[1].imshow(thresh1)
        dwg = image.copy()
        dwg = dwg * (thresh1[:, :, None].astype(dwg.dtype))
        t = cv.inRange(dwg, 0, 125)
        # cv.imshow("hui", cv.inRange(np.array(thresh1 * 255),0, 125))
        # cv.waitKey()
        t_gr = cv.morphologyEx(t, cv.MORPH_GRADIENT, np.ones((2, 2)))
        t = 255 - t
        level_imgs[low] = t
        contours, hierarchy = cv.findContours(t_gr, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        f_contours = filter_contours_hot(contours)
        if low != 95:
            f_contours = threshold_intensity(f_contours, gray_im, low)
        approx_contours = []
        for c in f_contours:
            # epsilon = 0.03 * cv.arcLength(c, True)
            # approx = cv.approxPolyDP(c, epsilon, True)
            # approx_contours.append(approx)
            cv.drawContours(temp, [c[0]], -1, color=(0, 0, 255), thickness=2)
            # rect = cv.minAreaRect(c)
            # box = cv.boxPoints(rect)
            # box = np.int0(box)
            # cv.drawContours(gray_im, [box], 0, (0, 0, 255), 2)
        f_contours = [(cnt, low) for cnt in f_contours]
        all_contours.append(f_contours)
        ax.flatten()[k].imshow(temp, cmap="gray")
        ax.flatten()[k].set_title('Пороговый уровень: ' + str(low), size=30)
        k = k + 1
    all_contours = [item for sublist in all_contours for item in sublist]
    res_contours = get_unique_parts(all_contours, 'h')
    temp = gray_im.copy()
    for c in res_contours:
        cv.drawContours(temp, [c[0]], -1, color=(0, 0, 255), thickness=2)
    res_contours = [(cnt[0], cnt[3]) for cnt in res_contours]
    ax.flatten()[5].imshow(temp, cmap="gray")
    ax.flatten()[5].set_title('Результат детектирования', size=30)
    plt.savefig(name, dpi=100, quality=70)
    return level_imgs, res_contours


def threshold_intensity(contours, gray, low):
    lst_intensities = []
    # cv.imshow("12", gray)
    contours_i = [cnt[0] for cnt in contours]
    # For each list of contour points...
    for i in range(len(contours)):
        # Create a mask image that contains the contour filled in
        cimg = np.zeros_like(gray)
        cv.drawContours(cimg, contours_i, i, color=255, thickness=cv.FILLED)
        # cv.imshow(str(i), cimg)
        # Access the image pixels and create a 1D numpy array then add to list
        pts = np.where(cimg == 255)
        lst_intensities.append(gray[pts[0], pts[1]])
    # cv.waitKey()
    mean_intensities = list(map(np.mean, lst_intensities))
    l = len(mean_intensities)
    if l < 2:
        return contours
    thresh = sorted(mean_intensities)[int(l * 0.7)]
    tr_c = []
    for i in range(len(contours)):
        c = contours[i]
        intensity = mean_intensities[i]
        if intensity < thresh or intensity < low + 7:
            continue
        tr_c.append(c)
    return tr_c


def filter_contours_hot(contours):
    from utility import dist
    res = []
    ind = []
    cnt_params = get_cnt_params(contours, "hot")

    for i in range(len(cnt_params['contour'])):
        add = True
        c_area = cnt_params['area'][i]
        c_xy = cnt_params['center'][i]
        # check for width and height => threshold 50-100 and thresh relatively w/l
        if cnt_params['borders']['w'][i] > 200 or cnt_params['borders']['h'][i] > 200:
            continue
        r = min(cnt_params['borders']['w'][i] / cnt_params['borders']['h'][i],
                cnt_params['borders']['h'][i] / cnt_params['borders']['w'][i])
        if c_area < 150 and r < 0.75:
            continue
        elif r < 0.3:
            continue
        for j in range(len(cnt_params['contour'])):
            if i == j:
                continue
            if cnt_params['contour'][j] is None:
                continue
            if dist(c_xy, cnt_params['center'][j]) < 40:
                if c_area <= cv.contourArea(cnt_params['contour'][j]):
                    add = False
                    break
                else:
                    cnt_params['contour'][j] = None
        if add:
            ind.append(i)
    for i in ind.copy():
        if cnt_params['contour'][i] is None:
            ind.remove(i)
    for i in ind:
        res.append(cnt_params['contour'][i])
    res_params = [(cnt_params['contour'][i],
                   cnt_params['center'][i],
                   cnt_params['area'][i],
                   cnt_params['borders']['w'][i],
                   cnt_params['borders']['h'][i],
                   cnt_params['length'][i])
                  for i in ind]
    return res_params


def solve_collision_hot(contour, lvl, unique_contours):
    from utility import dist
    cnt, center, area, _, _, _ = contour
    idx = -1
    for i, u_c in enumerate(unique_contours):
        u_cnt, u_center, u_area, _ = u_c
        if dist(center, u_center) < 40:
            if 400 > area > u_area:
                idx = i
                break
            if 800 > area > 400 and area > u_area:
                idx = i
                break
            if area / u_area > 5:
                return False
            if u_area / area > 5:
                idx = i
                break
            return True
    if idx == -1:
        return False
    unique_contours[idx] = (cnt, center, area, lvl)
    return True


def pre_filter_hot(c_area, c_len):
    if c_area > 6500 or c_area < 100:
        return False
    if c_len > 1000 or c_len < 40:
        return False
    if c_area < 300 and c_len > 100:
        return False
    if c_area > 2000 and (c_area / c_len < 6):
        return False
    return True


def pre_filter_other(c_area, c_len):
    if c_area > 5200 or c_area < 450:
        return False
    if c_len > 1000 or c_len < 80:
        return False
    if c_area < 800 and c_len > 250:
        return False
    if c_area > 600 and (c_area / c_len < 7):
        return False
    return True


def get_cnt_params(contours, t):
    cnt_params = {'center': [], 'area': [], 'length': [],
                  'borders': {'w': [], 'h': []},
                  'contour': []}
    for i in range(len(contours)):
        c = contours[i]
        c_area = get_area(c)
        c_len = get_length(c)
        w, h = get_width_height(c)
        if t == "other" and not pre_filter_other(c_area, c_len):
            continue
        if t == "hot" and not pre_filter_hot(c_area, c_len):
            continue
        c_xy = get_center(c)
        cnt_params['borders']['w'].append(w)
        cnt_params['borders']['h'].append(h)
        cnt_params['center'].append(c_xy)
        cnt_params['area'].append(c_area)
        cnt_params['length'].append(c_len)
        cnt_params['contour'].append(c)
    return cnt_params


def filter_contours_other(contours):
    from utility import dist
    res = []
    ind = []
    cnt_params = get_cnt_params(contours, "other")

    for i in range(len(cnt_params['contour'])):
        add = True
        c_area = cnt_params['area'][i]
        c_xy = cnt_params['center'][i]
        # check for width and height => threshold 50-100 and thresh relatively w/l
        if cnt_params['borders']['w'][i] > 200 or cnt_params['borders']['h'][i] > 200:
            continue
        r = min(cnt_params['borders']['w'][i] / cnt_params['borders']['h'][i],
                cnt_params['borders']['h'][i] / cnt_params['borders']['w'][i])
        if r < 0.4:
            continue
        # large area and not circle
        if c_area > 1000 and r < 0.75:
            continue
        for j in range(len(cnt_params['contour'])):
            if i == j:
                continue
            if cnt_params['contour'][j] is None:
                continue
            if dist(c_xy, cnt_params['center'][j]) < 50:
                if c_area >= cv.contourArea(cnt_params['contour'][j]):
                    add = False
                    break
                else:
                    cnt_params['contour'][j] = None
        if add:
            ind.append(i)
    for i in ind.copy():
        if cnt_params['contour'][i] is None:
            ind.remove(i)
    for i in ind:
        res.append(cnt_params['contour'][i])
    res_params = [(cnt_params['contour'][i],
                   cnt_params['center'][i],
                   cnt_params['area'][i],
                   cnt_params['borders']['w'][i],
                   cnt_params['borders']['h'][i],
                   cnt_params['length'][i])
                  for i in ind]
    return res_params


def solve_collision_other(contour, lvl, unique_contours):
    from utility import dist
    cnt, center, area, _, _, _ = contour
    idx = -1
    for i, u_c in enumerate(unique_contours):
        u_cnt, u_center, u_area, _ = u_c
        if dist(center, u_center) < 50:
            if 200 < area < u_area and u_area / area < 1.3:
                idx = i
                break
            if area / u_area > 2:
                idx = i
                break
            return True
    if idx == -1:
        return False
    unique_contours[idx] = (cnt, center, area, lvl)
    return True


def get_unique_parts(all_contours, type):
    unique_contours = []
    if type == 'o':
        for i, contour_lvl in enumerate(all_contours):
            contour, lvl = contour_lvl
            cnt, center, area, _, _, _ = contour
            if not solve_collision_other(contour, lvl, unique_contours):
                unique_contours.append((cnt, center, area, lvl))
    else:
        for i, contour_lvl in enumerate(all_contours):
            contour, lvl = contour_lvl
            cnt, center, area, _, _, _ = contour
            if not solve_collision_hot(contour, lvl, unique_contours):
                unique_contours.append((cnt, center, area, lvl))
    return unique_contours


def detect_other_parts(image, name):
    from scipy.ndimage import binary_fill_holes
    # detects low temperature parts
    fig, ax = plt.subplots(3, 2, figsize=(30, 20))
    gray_im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_im = 255 - gray_im
    k = 0
    level_imgs = {'lvl': int, 'image': []}
    all_contours = []
    for low in (215, 220, 225, 230, 235):
        temp = gray_im.copy()
        ret, thresh1 = cv.threshold(gray_im, low, 255, cv.THRESH_BINARY)
        thresh1 = binary_closing(thresh1, selem=np.ones((2, 2)))
        t = np.array(thresh1 * 255)
        t = cv.inRange(t, 0, 125)
        t = 255 - t
        level_imgs[low] = t
        contours, hierarchy = cv.findContours(t, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        f_contours = filter_contours_other(contours)
        for c in f_contours:
            cv.drawContours(temp, [c[0]], -1, color=(0, 0, 255), thickness=2)
        ax.flatten()[k].imshow(temp, cmap="gray")
        ax.flatten()[k].set_title('Пороговый уровень: ' + str(low), size=30)
        f_contours = [(cnt, low) for cnt in f_contours]
        all_contours.append(f_contours)
        k = k + 1
    # cv.waitKey()
    # plt.savefig(name, dpi=100, quality=50)
    all_contours = [item for sublist in all_contours for item in sublist]
    res_contours = get_unique_parts(all_contours, 'o')
    temp = gray_im.copy()
    for c in res_contours:
        cv.drawContours(temp, [c[0]], -1, color=(0, 0, 255), thickness=2)
    res_contours = [(cnt[0], cnt[3]) for cnt in res_contours]
    ax.flatten()[5].imshow(temp, cmap="gray")
    ax.flatten()[5].set_title('Результат детектирования', size=30)
    plt.savefig(name, dpi=100, quality=50)
    # plt.show()
    # save changed_contours
    return level_imgs, res_contours