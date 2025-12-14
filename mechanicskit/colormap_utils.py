"""
MATLAB-style colormap utilities for automatic colorbar creation.

Provides convenient functions for adding colorbars to patch visualizations
without manual ScalarMappable creation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Module-level storage for last patch state
_last_patch_state = {
    'collection': None,
    'cdata': None,
    'ax': None,
}


def _store_patch_state(collection, cdata, ax):
    """
    Store patch state for later colorbar creation.

    Called internally by patch() function.

    Parameters
    ----------
    collection : LineCollection, PolyCollection, or Poly3DCollection
        The matplotlib collection object created by patch()
    cdata : np.ndarray
        Color data array (per-face or per-vertex)
    ax : matplotlib.axes.Axes
        Axes where the patch was drawn
    """
    global _last_patch_state
    _last_patch_state = {
        'collection': collection,
        'cdata': np.asarray(cdata),
        'ax': ax,
    }


def colorbar(cmap_name='viridis', label=None, ax=None, vmin=None, vmax=None, **kwargs):
    """
    Apply colormap and add colorbar to the last created patch.

    MATLAB-style convenience function that applies a colormap to the most
    recent patch() call and adds a colorbar. Eliminates the need for manual
    ScalarMappable creation and colormap specification in patch().

    Parameters
    ----------
    cmap_name : str, default 'viridis'
        Colormap name. Common values: 'jet', 'viridis', 'plasma', 'coolwarm',
        'RdBu', 'seismic', 'rainbow'.
    label : str, optional
        Colorbar label text.
    ax : matplotlib.axes.Axes, optional
        Axes for colorbar placement. If None, uses axes from last patch().
    vmin, vmax : float, optional
        Color normalization limits. If None, uses data min/max.
    **kwargs : dict
        Additional keyword arguments passed to plt.colorbar():
        - shrink : float, default 1.0 (fraction of original size)
        - orientation : 'vertical' or 'horizontal'
        - pad : float, spacing from axes
        - aspect : float, ratio of long to short dimensions

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        The created colorbar object.

    Raises
    ------
    RuntimeError
        If patch() has not been called before this function.

    Examples
    --------
    Basic usage with element forces (default viridis):

    >>> import mechanicskit as mk
    >>> mk.patch('Faces', elements, 'Vertices', nodes, 'FaceVertexCData', forces)
    >>> mk.colorbar(label='Force [N]')

    Specify colormap:

    >>> mk.patch('Faces', elements, 'Vertices', nodes, 'FaceVertexCData', stress)
    >>> mk.colorbar('jet', label='Stress [MPa]')

    Custom range and horizontal colorbar:

    >>> mk.patch('Faces', elements, 'Vertices', nodes, 'FaceVertexCData', temps)
    >>> mk.colorbar('jet', vmin=0, vmax=100, orientation='horizontal', shrink=0.8)

    Interpolated nodal temperatures:

    >>> mk.patch('Faces', elements, 'Vertices', nodes,
    ...          'FaceVertexCData', nodal_temps, 'FaceColor', 'interp')
    >>> mk.colorbar('jet', label='Temperature [°C]')

    See Also
    --------
    patch : Create colored patches
    cmap : Alias for colorbar (MATLAB-compatible name)
    matplotlib.pyplot.colorbar : Underlying colorbar function
    """
    global _last_patch_state

    # Validate patch was called
    if _last_patch_state['cdata'] is None:
        raise RuntimeError(
            "No patch data found. Call patch() before colorbar()."
        )

    # Get stored state
    collection = _last_patch_state['collection']
    cdata = _last_patch_state['cdata']
    ax_to_use = ax if ax is not None else _last_patch_state['ax']

    # Compute limits if not provided
    if vmin is None:
        vmin = cdata.min()
    if vmax is None:
        vmax = cdata.max()

    # Create normalization
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Apply colormap to stored data
    cmap_obj = cm.get_cmap(cmap_name)
    colors = cmap_obj(norm(cdata))

    # Update collection colors
    collection.set_colors(colors)

    # Create ScalarMappable for colorbar
    mappable = cm.ScalarMappable(cmap=cmap_name, norm=norm)
    mappable.set_array([])

    # Create colorbar
    cbar = plt.colorbar(mappable, ax=ax_to_use, **kwargs)

    # Set label if provided
    if label is not None:
        cbar.set_label(label)

    return cbar


# MATLAB-compatible alias
cmap = colorbar
