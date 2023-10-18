from __future__ import annotations

import itertools


def unit_color_generator():
    colors = ['yellow', 'cyan', 'lime', 'magenta', 'gold', 'deepskyblue', 'hotpink']
    return itertools.cycle(colors)

def window_color_generator():
    colors = [
        'darkolivegreen', 'teal', 'saddlebrown', 'darkred', 'darkgreen',
        'darkgoldenrod', 'firebrick', 'darkcyan', 'maroon', 'darkslategray',
        'darkorange', 'darkkhaki', 'dimgray', 'darkmagenta', 'darkseagreen', 'sienna'
    ]
    return itertools.cycle(colors)
