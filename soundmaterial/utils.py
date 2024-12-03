from typing import List

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

