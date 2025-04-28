import typing
from typing import List
import numbers 

import numpy as np

import pandas as pd


def smart_plotly_export(fig, save_path):
    img_format = save_path.split('.')[-1]
    if img_format == 'html':
        fig.write_html(save_path)
    elif img_format == 'bytes':
        return fig.to_image(format='png')
    #TODO: come back and make this prettier
    elif img_format == 'numpy':
        import io 
        from PIL import Image

        def plotly_fig2array(fig):
            #convert Plotly fig to  an array
            fig_bytes = fig.to_image(format="png", width=1200, height=700)
            buf = io.BytesIO(fig_bytes)
            img = Image.open(buf)
            return np.asarray(img)
        
        return plotly_fig2array(fig)
    elif img_format == 'jpeg' or 'png' or 'webp':
        fig.write_image(save_path)
    else:
        raise ValueError("invalid image format")


def dim_reduce(
        emb: np.ndarray, 
        labels: List[str], 
        metadata=List[dict], 
        save_path: str = "plot.html", 
        n_components: int =3,
        method: str ='tsne', 
        title: str=''
    ):
    """
    dimensionality reduction for visualization!
    
    parameters:
        emb (np.ndarray): the samples to be reduces with shape (n_example, n_dim)
        labels (list): list of labels // captions for the embedding
        save_path (str): path to save the figure
        method (str): umap, tsne, or pca
        title (str): title for ur figure
    returns:    
        saves an html plotly figure to save_path
    """
    import pandas as pd
    import plotly.express as px
    if method == 'umap':
        from umap import UMAP
        reducer = umap.UMAP(n_components=n_components)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f"incorrect method: {method}")
 
    print(f"reducing to {n_components} dimensions")
    proj = reducer.fit_transform(emb)
    print(f"dim reduction done! ")

    if n_components == 2:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            instrument=labels
        ))
        # add metadata
        for key in metadata[0].keys():
            df[key] = [m[key] for m in metadata]

        fig = px.scatter(df, x='x', y='y', color='instrument',
                        title=title+f"_{method}", hover_data=[key for key in metadata[0].keys()])

    elif n_components == 3:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            z=proj[:, 2],
            instrument=labels
        ))
        # add metadata
        for key in metadata[0].keys():
            df[key] = [m[key] for m in metadata]

        fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='instrument',
                        title=title, hover_data=[key for key in metadata[0].keys()])
    else:
        raise ValueError("cant plot more than 3 components")

    fig.update_traces(marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    return smart_plotly_export(fig, save_path)


def random_state(seed: typing.Union[int, np.random.RandomState]):
    """
    Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : typing.Union[int, np.random.RandomState] or None
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    np.random.RandomState
        Random state object.

    Raises
    ------
    ValueError
        If seed is not valid, an error is thrown.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, (numbers.Integral, np.integer, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError(
            "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
        )


def seed(random_seed, set_cudnn=False):
    """
    Seeds all random states with the same random seed
    for reproducibility. Seeds ``numpy``, ``random`` and ``torch``
    random generators.
    For full reproducibility, two further options must be set
    according to the torch documentation:
    https://pytorch.org/docs/stable/notes/randomness.html
    To do this, ``set_cudnn`` must be True. It defaults to
    False, since setting it to True results in a performance
    hit.

    Args:
        random_seed (int): integer corresponding to random seed to
        use.
        set_cudnn (bool): Whether or not to set cudnn into determinstic
        mode and off of benchmark mode. Defaults to False.
    """

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if set_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sample_from_dist(dist_tuple: tuple, state: np.random.RandomState = None):
    """Samples from a distribution defined by a tuple. The first
    item in the tuple is the distribution type, and the rest of the
    items are arguments to that distribution. The distribution function
    is gotten from the ``np.random.RandomState`` object.

    Parameters
    ----------
    dist_tuple : tuple
        Distribution tuple
    state : np.random.RandomState, optional
        Random state, or seed to use, by default None

    Returns
    -------
    typing.Union[float, int, str]
        Draw from the distribution.

    Examples
    --------
    Sample from a uniform distribution:

    >>> dist_tuple = ("uniform", 0, 1)
    >>> sample_from_dist(dist_tuple)

    Sample from a constant distribution:

    >>> dist_tuple = ("const", 0)
    >>> sample_from_dist(dist_tuple)

    Sample from a normal distribution:

    >>> dist_tuple = ("normal", 0, 0.5)
    >>> sample_from_dist(dist_tuple)

    """
    if dist_tuple[0] == "const":
        return dist_tuple[1]
    state = random_state(state)
    dist_fn = getattr(state, dist_tuple[0])
    return dist_fn(*dist_tuple[1:])
