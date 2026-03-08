# custom_colormap.py

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np

class CustomColormap:
    def __init__(self, name, cdict):
        self.name = name
        self.cdict = cdict
        self.colormap = LinearSegmentedColormap(name, cdict)
    
    def get_colormap(self):
        return self.colormap
    
    def plot_colormap(self):
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        fig, ax = plt.subplots(figsize=(5, 2))
        ax.imshow(gradient, aspect='auto', cmap=self.colormap)
        ax.set_axis_off()
        plt.title(self.name)
        plt.show()

    def reversed_colormap(self):
        reversed_cdict = {color: [(1.0 - pos, val1, val2) for pos, val1, val2 in reversed(self.cdict[color])] for color in self.cdict}
        return CustomColormap(self.name + '_r', reversed_cdict)

    def apply_gamma(self, gamma_value):
        gamma_cmap = self.colormap(np.linspace(0, 1, self.colormap.N) ** gamma_value)
        return ListedColormap(gamma_cmap)

class ColorDefinitions:
    @staticmethod
    def rainbow():
        return {'red':   ((0.0, 0.0, 0.0),
                          (0.075, 0.25, 0.25),
                          (0.25, 0.0, 0.0),
                          (0.4, 0.0, 0.0),
                          (0.75, 1.0, 1.0),
                          (0.9, 1.0, 1.0),
                          (1.0, 1.0, 1.0)),
                'green': ((0.0, 0.0, 0.0),
                          (0.1, 0.0, 0.0),
                          (0.25, 0.0, 0.0),
                          (0.4, 0.75, 0.75),
                          (0.75, 1.0, 1.0),
                          (0.9, 0.0, 0.0),
                          (1.0, 1.0, 1.0)),
                'blue':  ((0.0, 0.0, 0.0),
                          (0.075, 0.25, 0.25),
                          (0.25, 0.8, 0.8),
                          (0.4, 1.0, 1.0),
                          (0.75, 0.0, 0.0),
                          (0.9, 0.0, 0.0),
                          (1.0, 1.0, 1.0))}

    @staticmethod
    def cool():
        return {'red':   ((0.0, 0.0, 0.0),
                          (0.75, 0.0, 0.0),
                          (1.0, 1.0, 1.0)),
                'green': ((0.0, 0.0, 0.0),
                          (0.4, 0.0, 0.0),
                          (0.7, 0.5, 0.5),
                          (0.95, 1.0, 1.0),
                          (1.0, 1.0, 1.0)),
                'blue':  ((0.0, 0.0, 0.0),
                          (0.5, 0.9, 0.9),
                          (1.0, 1.0, 1.0))}
