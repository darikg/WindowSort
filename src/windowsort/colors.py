from __future__ import annotations

import itertools


def unit_color_generator():
    colors = ['yellow', 'cyan', 'lime', 'magenta', 'gold', 'deepskyblue', 'hotpink']
    return itertools.cycle(colors)

def window_color_generator():
    colors = [
        'greenyellow', 'aquamarine', 'orchid', 'lightseagreen', 'darkturquoise',
        'palevioletred', 'plum', 'lightsalmon', 'khaki', 'mediumorchid',
        'lightskyblue', 'lightcoral', 'palegreen', 'peachpuff', 'lightslategray', 'rosybrown'
    ]
    return itertools.cycle(colors)
