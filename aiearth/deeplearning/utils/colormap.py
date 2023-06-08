import random
import numpy as np

def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


def create_colormap_by_class_color(palette:list):
    colormap = np.zeros((256, 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        colormap[idx] = color
    return colormap


def color_to_bgr(color_code):
    if color_code[0] != "#":
        raise Exception("color code error")
    red = int(color_code[1:3], 16)
    green = int(color_code[3:5], 16)
    blue = int(color_code[5:7], 16)
    return blue, green, red


def generate_color_code_list(class_num):
    color_code_list = []
    colormap = create_pascal_label_colormap()
    for i in range(class_num):
        bgr_color = tuple(colormap[i+1])
        color_code = bgr2color_code(bgr_color)
        color_code_list.append(color_code)
    return color_code_list


def bgr2color_code(bgr_tuple):
    red_code = int_to_color_string(bgr_tuple[2])
    green_code = int_to_color_string(bgr_tuple[1])
    blue_code = int_to_color_string(bgr_tuple[0])
    return "#" + red_code + green_code + blue_code


def int_to_color_string(i: int):
    assert i < 256
    hex_string = hex(i)
    ret = hex_string[2:].upper()
    if len(ret) == 1:
        ret = '0' + ret
    return ret


if __name__ == '__main__':
    classes = ["a", "b", "c"]
    color_list = generate_color_code_list(len(classes))
    for cls, color in zip(classes, color_list):
        print(cls, color)