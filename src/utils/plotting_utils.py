import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_in_row(data: list, titles: list[str], figheight: int = 6) -> mpl.figure.Figure:
    """Plot subplots in a row.
    """
    num_subplots = len(data)
    fig, ax = plt.subplots(nrows=1, ncols=num_subplots, figsize=(num_subplots * figheight, figheight))

    for idx, dat in enumerate(data):
        ax[idx].plot(dat)
        ax[idx].set_title(titles[idx])
    
    return fig
        

