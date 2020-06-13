from PIL import Image, ImageDraw
import numpy as np

angle = 2
rotates = (angle, 180 - angle, 4*angle)
sq_rotates = (angle, 90 - angle, 2*angle)

aff_matrix_1 = [[[6, 6], [26, 6], [26, 26]], [[3, 9], [23, 9], [28, 26]]]
aff_matrix_2 = [[[6, 6], [26, 6], [26, 26]], [[5, 3], [25, 3], [28, 27]]]
aff_matrix_3 = [[[6, 6], [26, 6], [26, 26]], [[9, 6], [28, 6], [23, 26]]]
aff_matrix_4 = [[[6, 6], [26, 6], [26, 26]], [[10, 10], [28, 10], [22, 25]]]
aff_matrices = [aff_matrix_1, aff_matrix_2, aff_matrix_3, aff_matrix_4]


def generate_affine_matrices(n):
    first = [[6, 6], [26, 6], [26, 26]]
    second = [[3, 9], [23, 9], [28, 26]]
    for i in range(n):
        yield [first, second]
        second[0][0] = second[0][0] + 1
        second[0][1] = second[0][1] - 1
        second[1][0] = second[1][0] + 1
        second[1][1] = second[1][1] - 1


# image can be interpreted as 32x32 binary array
def generate_affines(image, affine_matrix):
    from cv2 import getAffineTransform, warpAffine, BORDER_TRANSPARENT, BORDER_DEFAULT
    img = np.array(image)
    rows, cols, ch = img.shape
    pts1, pts2 = affine_matrix
    M = getAffineTransform(np.array(pts1).astype(np.float32), np.array(pts2).astype(np.float32))
    dst = warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255), borderMode=BORDER_DEFAULT)
    return Image.fromarray(dst)


def generate_rotates(image, rotations):
    k = 2
    rot = image
    low_angle, high_angle, delta_angle = rotations
    for angle in range(low_angle, high_angle, delta_angle):
        rot = rot.rotate(angle, resample=Image.BICUBIC, fillcolor='white')
        k = k + 1
        yield rot
    pass


def draw_size(draw_fun, size, fig_size, rect_diff):
    draw_fun((size, size + rect_diff, fig_size - size, fig_size - size - rect_diff),
             fill='black', outline='black')


def generate_sizes(low, high, step, type):
    fig_size = 32
    if type == "c":
        for size in range(low, high, step):
            image = Image.new('RGB', (fig_size, fig_size), 'white')
            draw = ImageDraw.Draw(image)
            draw_size(draw.ellipse, size, fig_size, 0)
            yield image
    if type == "r":
        for size in range(low, high, step):
            image = Image.new('RGB', (fig_size, fig_size), 'white')
            draw = ImageDraw.Draw(image)
            draw_size(draw.rectangle, size, fig_size, 2)
            yield image
    if type == "s":
        for size in range(low, high, step):
            image = Image.new('RGB', (fig_size, fig_size), 'white')
            draw = ImageDraw.Draw(image)
            draw_size(draw.rectangle, size, fig_size, 0)
            yield image


def generate_squares(path, num):
    # image = Image.new('RGB', (32, 32), 'white')
    # draw = ImageDraw.Draw(image)
    # draw.rectangle((6, 6, 26, 26), fill='black',
    #                outline='black')
    images_sizes = list(generate_sizes(4, 10, 2, 's'))
    images = images_sizes
    for image in images_sizes.copy():
        images.extend(generate_rotates(image.copy(), sq_rotates))
        for aff in aff_matrices:
            images.extend(generate_rotates(
                generate_affines(images[0], aff),
                rotates))
    res = [(x, path + str(i) + '_square.png') for (i, x) in enumerate(images)]
    return res[0:num:1]


def generate_circles(path, num):
    # image = Image.new('RGB', (32, 32), 'white')
    # draw = ImageDraw.Draw(image)
    # draw.ellipse((6, 6, 26, 26), fill='black',
    #              outline='black')
    images_sizes = list(generate_sizes(4, 10, 2, 'c'))
    images = images_sizes
    for image in images_sizes.copy():
        images.extend(generate_rotates(image.copy(), sq_rotates))
        for aff in aff_matrices:
            images.extend(generate_rotates(
                generate_affines(images[0], aff),
                rotates))
    res = [(x, path + str(i) + '_circle.png') for (i, x) in enumerate(images)]
    return res[0:num:1]


def generate_rects(path, num):
    # image = Image.new('RGB', (32, 32), 'white')
    # draw = ImageDraw.Draw(image)
    # draw.rectangle((6, 8, 26, 22), fill='black',
    #                outline='black')
    images_sizes = list(generate_sizes(4, 10, 2, 'r'))
    images = images_sizes
    for image in images_sizes.copy():
        images.extend(generate_rotates(image.copy(), rotates))
        for aff in aff_matrices:
            images.extend(generate_rotates(
                generate_affines(images[0], aff),
                rotates))
    res = [(x, path + str(i) + '_rectangle.png') for (i, x) in enumerate(images)]
    return res[0:num:1]


def generate_images(path, num):
    save_images = generate_squares('data/templates/square/', num)
    save_images.extend(generate_circles('data/templates/circle/', num))
    save_images.extend(generate_rects('data/templates/rectangle/', num))
    print(save_images)
    for obj in save_images:
        i, name = obj
        i.save(name)
    pass
