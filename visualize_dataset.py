import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from embedding_atlas.widget import EmbeddingAtlasWidget
from embedding_atlas.projection import compute_text_projection

def visualize_tsne(features, labels, num_samples=1000, perplexity=50, title="t-SNE Visualization", random_state=42):
    if len(features) > num_samples:
        idx = np.random.choice(len(features), num_samples, replace=False)
        features_sub = features[idx]
        labels_sub = labels[idx]
    else:
        features_sub = features
        labels_sub = labels

    features_sub = StandardScaler().fit_transform(features_sub)
    tsne_results = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(features_sub)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_sub, cmap='tab10', alpha=0.7)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Label")
    plt.show()

    df = pd.DataFrame({
        "x": tsne_results[:, 0],
        "y": tsne_results[:, 1],
        "label": labels_sub
    })
    
    #compute_text_projection(df,text="label",x="x", y="y", neighbors="neighbors")
    
    return EmbeddingAtlasWidget(df,x="x", y="y", neighbors="neighbors")
