"""
VISUALIZATION
=============

This module contains visualization utilities for plotting and displaying data
within the interested opticalib framework. This includes wavefront maps, plotting
DM commands, and other relevant visualizations for optical data analysis.
"""

import numpy as np
from matplotlib import pyplot as plt
from . import typings as _ot
from .ground import osutils as _osu


def matshow(matrix: _ot.MatrixLike, **kwargs: dict[str, _ot.Any]):
    """
    Display a matrix using matplotlib's matshow.

    Parameters
    ----------
    matrix : array-like
        2D array representing the matrix to be displayed.
    **kwargs
        Additional keyword arguments to be passed to `matshow`, as well as some
        additions:

        - title: str, optional
            Title of the plot. Default is "Matrix".
        - xlabel: str, optional
            Label for the x-axis. Default is "X [px]".
        - ylabel: str, optional
            Label for the y-axis. Default is "Y [px]".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    im : matplotlib.image.AxesImage
        The image object returned by matshow.
    """
    title = kwargs.pop("title", "Matrix")
    xlabel = kwargs.pop("xlabel", "X [px]")
    ylabel = kwargs.pop("ylabel", "Y [px]")

    fig, ax = plt.subplots()
    im = ax.matshow(matrix, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    return fig, ax, im


def surfshow(
    image: _ot.ImageData,
    d3: bool = False,
    cut: bool = True,
    **kwargs: dict[str, _ot.Any],
):
    """
    Display a 3D surface plot of an image using matplotlib's 3D plotting capabilities.

    Parameters
    ----------
    image : array-like
        2D array representing the image to be displayed as a surface.
    d3 : bool, optional
        Whether to display the image as a 3D surface plot. Default is False.
    cut : bool, optional
        Whether to cut the image to the region of interest. Default is True.
    **kwargs
        Additional keyword arguments to be passed to `plot_surface`, as well as some
        additions:

        - title: str, optional
            Title of the plot. Default is "Surface Plot".
        - xlabel: str, optional
            Label for the x-axis. Default is "X [px]".
        - ylabel: str, optional
            Label for the y-axis. Default is "Y [px]".
        - zlabel: str, optional
            Label for the z-axis. Default is "Z [ADU]" (if ``d3`` is True).
        - clabel: str, optional
            Label for the colorbar. Default is "meters".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes object.
    surf : matplotlib.surface.Poly3DCollection
        The surface object returned by plot_surface.
    """
    if not d3:
        fig, ax, im = myimshow(image, cut=cut, **kwargs)
        cbar = im.colorbar
        cbar.set_label(kwargs.get("clabel", "meters"))
        return fig, ax, im
    else:
        fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": "3d"})
        x = np.arange(image.shape[1])
        y = np.arange(image.shape[0])
        x, y = np.meshgrid(x, y)
        surf = ax.plot_surface(x, y, image, cmap="viridis")
        ax.set_xlabel(kwargs.get("xlabel", "X [px]"))
        ax.set_ylabel(kwargs.get("ylabel", "Y [px]"))
        ax.set_zlabel(kwargs.get("zlabel", "Z [ADU]"))
        ax.set_title(kwargs.get("title", "Surface Plot"))
        fig.colorbar(
            surf,
            ax=ax,
            shrink=0.5,
            aspect=7.5,
            pad=0.15,
            label=kwargs.get("clabel", "meters"),
        )
        return fig, ax, surf


def cmdplot(cmd: _ot.ArrayLike, **kwargs: dict[str, _ot.Any]):
    """
    Plot a DM command as a 2D image.

    Parameters
    ----------
    cmd : array-like
        1D array representing the DM command to be displayed.
    **kwargs
        Additional keyword arguments to be passed to `imshow`, as well as some
        additions:

        - title: str, optional
            Title of the plot. Default is "DM Command".
        - xlabel: str, optional
            Label for the x-axis. Default is "X [px]".
        - ylabel: str, optional
            Label for the y-axis. Default is "Y [px]".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    l : list[matplotlib.lines.Line2D]
        The list of line objects returned by plot.
    """
    title = kwargs.pop("title", "DM Command")
    xlabel = kwargs.pop("xlabel", "X [px]")
    ylabel = kwargs.pop("ylabel", "Y [px]")
    c = _osu.get_kwargs(("color", "c"), default="black", kwargs=kwargs)

    fig, ax = plt.subplots()
    l = ax.plot(cmd, "-o", linewidth=2, markersize=5, color=c, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig, ax, l


def myimshow(
    image: np.ndarray,
    cut: bool = False,
    **kwargs: dict[str, _ot.Any],
):
    """
    Display a single image using matplotlib's imshow.

    Generic purpose wrapper with in-function options for common settings (title,
    labels, aspect ratio) and a cut option for cutting the image to the region
    of interest.

    Parameters
    ----------
    image : array-like
        2D array representing the image to be displayed.
    **kwargs
        Additional keyword arguments to be passed to `imshow`, as well as some
        additions:

        - title: str, optional
            Title of the plot. Default is "Image".
        - xlabel: str, optional
            Label for the x-axis. Default is "X [px]".
        - ylabel: str, optional
            Label for the y-axis. Default is "Y [px]".
        - axis: str, optional
            Aspect ratio for the plot. Default is "equal". Can be set to "auto"
            for automatic aspect ratio.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    im : matplotlib.image.AxesImage
        The image object returned by imshow.
    """
    if cut:
        from opticalib.ground.roi import imgCut

        image = imgCut(image)

    title = kwargs.pop("title", "Image")
    xlabel = kwargs.pop("xlabel", "X [px]")
    ylabel = kwargs.pop("ylabel", "Y [px]")
    axis = kwargs.pop("axis", "equal")

    fig, ax = plt.subplots()
    im = ax.imshow(image, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect(axis)
    fig.colorbar(im, ax=ax, aspect=10)

    return fig, ax, im


def superimshow(
    data: list[_ot.ImageData] | _ot.CubeData,
    nrows: int = None,
    ncols: int = None,
    titles: list[str] | str = None,
    xlabels: list[str] | str = None,
    ylabels: list[str] | str = None,
    *mplargs: _ot.Any,
    **mplkwargs: dict[str, _ot.Any],
):
    """
    Display multiple images in a single figure.

    Parameters
    ----------
    data : list or array-like
        List of 2D arrays or a cube of images (3D array of shape `(npx, npx, nimg)`).
    nrows : int, optional
        Number of rows in the figure.
    ncols : int, optional
        Number of columns in the figure.
    titles : list or str, optional
        List of titles for each image. If not provided, the titles are the index of the image.
    xlabels : list or str, optional
        List of x-axis labels for each image. If not provided, the labels are 'X [px]'.
    ylabels : list or str, optional
        List of y-axis labels for each image. If not provided, the labels are 'Y [px]'.
    *mplargs
        Additional arguments to be passed to `imshow`.
    **mplkwargs
        Additional keyword arguments to be passed to `imshow`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the subplots.
    axs : numpy.ndarray
        Array of axes objects corresponding to each subplot.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if not isinstance(data, np.ndarray):
        data = np.ma.dstack(data)
    Nplots = data.shape[-1]
    titles = (
        [f"{i}" for i in range(Nplots)]
        if titles is None
        else titles * Nplots if isinstance(titles, str) else titles
    )
    xlabel = (
        ["X [px]"] * Nplots
        if xlabels is None
        else xlabels * Nplots if isinstance(xlabels, str) else xlabels
    )
    ylabel = (
        ["Y [px]"] * Nplots
        if ylabels is None
        else ylabels * Nplots if isinstance(ylabels, str) else ylabels
    )
    ncol = int(np.ceil(np.sqrt(Nplots))) if ncols is None else ncols
    nrow = int(np.ceil(Nplots / ncol)) if nrows is None else nrows
    if (nrow * ncol) < Nplots:
        raise ValueError(
            f"Number of rows and columns is not enough to display {Nplots} images."
        )
    fig, axs = plt.subplots(nrow, ncol, figsize=(16, 9))
    # rendiamo ax un array per consistenza
    if isinstance(axs, plt.Axes):  # type: ignore
        axs = np.array([axs])
    # "Appiattiamo" l'array di assi per iterare in modo lineare
    axs = axs.ravel()
    # Cicliamo su ogni immagine fino a esaurirle
    for i in range(Nplots):
        ax = axs[i]
        # Creiamo la colormap e la colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # Plot dell'immagine
        im = ax.imshow(data[..., i], *mplargs, **mplkwargs)
        ax.set_title(titles[i])
        ax.set_xlabel(xlabel[i])
        ax.set_ylabel(ylabel[i])
        ax.set_aspect("equal")
        # Aggiunta del colorbar
        fig.colorbar(im, cax=cax)
    # Se rimangono assi vuoti, li spegniamo
    for j in range(Nplots, len(axs)):
        axs[j].axis("off")
    fig.tight_layout()
    plt.show()
    return fig, axs
