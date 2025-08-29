from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from scipy.spatial.distance import *
from sklearn.metrics import *
from sklearn.preprocessing import *


# The class is a wrapper of evaluation approaches.
class STFMEvaluation:
    # ===================
    # prediction accuracy
    # ===================
    # Adjusted Rand Index
    def compute_ARI(self, labels_true: pd.Series, labels_pred: pd.Series):
        return adjusted_rand_score(labels_true, labels_pred)

    # Normalized Mutual Information
    def compute_NMI(self, labels_true: pd.Series, labels_pred: pd.Series):
        return normalized_mutual_info_score(labels_true, labels_pred)

    # Homogeneity
    def compute_HOM(self, labels_true: pd.Series, labels_pred: pd.Series):
        return homogeneity_score(labels_true, labels_pred)

    # Completeness
    def compute_COM(self, labels_true: pd.Series, labels_pred: pd.Series):
        return completeness_score(labels_true, labels_pred)

    # V measure score: harmonic mean of HOM and COM
    def compute_V_measure(self, labels_true: pd.Series, labels_pred: pd.Series):
        return v_measure_score(labels_true, labels_pred)

    def compute_prediction_accuracy(self, labels_true: pd.Series, labels_pred: pd.Series):
        metrics = pd.Series()
        metrics['ARI'] = self.compute_ARI(labels_true, labels_pred)
        metrics['NMI'] = self.compute_NMI(labels_true, labels_pred)
        metrics['HOM'] = self.compute_HOM(labels_true, labels_pred)
        metrics['COM'] = self.compute_COM(labels_true, labels_pred)
        metrics['V_Measure'] = self.compute_V_measure(labels_true, labels_pred)
        return metrics

    # ===================
    # spatial domain continuity
    # ===================
    # Average Silhouette Width
    def compute_ASW(self, spatial_coordinates: pd.DataFrame, labels_pred: pd.Series):
        d = squareform(pdist(spatial_coordinates))
        return silhouette_score(X=d, labels=labels_pred, metric='precomputed')

    def _fx_1NN(self, i, location_in):
        location_in = np.array(location_in)
        dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
        dist_array[i] = np.inf
        return np.min(dist_array)

    # Spatial CHAOS score
    def compute_CHAOS(self, spatial_coordinates: pd.DataFrame, labels_pred: pd.Series):
        clusterlabel = np.array(labels_pred)
        location = np.array(spatial_coordinates)
        matched_location = StandardScaler().fit_transform(spatial_coordinates)

        clusterlabel_unique = np.unique(clusterlabel)
        dist_val = np.zeros(len(clusterlabel_unique))
        count = 0
        for k in clusterlabel_unique:
            location_cluster = matched_location[clusterlabel == k, :]
            if len(location_cluster) <= 2:
                continue
            n_location_cluster = len(location_cluster)
            results = [self._fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
            dist_val[count] = np.sum(results)
            count = count + 1

        return np.sum(dist_val) / len(clusterlabel)

    def _fx_kNN(self, i, location_in, k, cluster_in):

        location_in = np.array(location_in)
        cluster_in = np.array(cluster_in)

        dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
        dist_array[i] = np.inf
        ind = np.argsort(dist_array)[:k]
        cluster_use = np.array(cluster_in)
        if np.sum(cluster_use[ind] != cluster_in[i]) > (k / 2):
            return 1
        else:
            return 0

    # Percentage of Abnormal Spots
    def compute_PAS(self, spatial_coordinates: pd.DataFrame, labels_pred: pd.Series):
        clusterlabel = np.array(labels_pred)
        location = np.array(spatial_coordinates)
        matched_location = location
        results = [self._fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel) for i in
                   range(matched_location.shape[0])]
        return np.sum(results) / len(clusterlabel)

    def compute_domain_continuity(self, spatial_coordinates: pd.DataFrame, labels_pred: pd.Series):
        metrics = pd.Series()
        metrics['ASW'] = self.compute_ASW(spatial_coordinates, labels_pred)
        metrics['CHAOS'] = self.compute_CHAOS(spatial_coordinates, labels_pred)
        metrics['PAS'] = self.compute_PAS(spatial_coordinates, labels_pred)
        return metrics


    # ===================
    # visualization
    # ===================

    def plot_spatial_domains(
            self,
            adata,
            domain_col: str = "domain",
            use_histology: bool = True,
            slice_id: Optional[str] = None,
            spot_size: Optional[float] = None,
            save: Optional[str] = None
    ) -> None:
        """
        Visualize domains on spatial coordinates. If 10x Visium histology is available,
        optionally overlay domains on the tissue image; otherwise draw a plain scatter.

        Args:
            adata (anndata.AnnData):
                Must contain `adata.obsm['spatial']` with N x 2 coordinates.
                If Visium images are present, `adata.uns['spatial']` should exist.
            domain_col (str, default: "domain"):
                Column in `adata.obs` containing domain labels to color by.
            use_histology (bool, default: True):
                If True and Visium images are present, use `sc.pl.spatial` to overlay on histology.
                If False or images are absent, fall back to a plain scatter plot.
            slice_id (Optional[str], default: None):
                When multiple Visium libraries/slices exist, choose which one to plot.
                If None, Scanpy will pick a default.
            spot_size (Optional[float], default: None):
                Spot/marker size; `None` lets the plotting function choose defaults.
            save (Optional[str], default: None):
                File path to save the figure (e.g., "spatial_domains.png"). If None, do not save.

        Returns:
            None

        Notes:
            - The annotate option applies only to the plain scatter fallback. For histology plots,
              annotation requires access to the created figure/axes; for broad compatibility we skip it.
        """
        if "spatial" not in adata.obsm:
            raise ValueError("Missing `adata.obsm['spatial']`. Please ensure spatial coordinates are present.")

        has_visium_img = isinstance(adata.uns.get("spatial", None), dict) and use_histology

        if has_visium_img:
            # Prefer Scanpy's spatial plot with histology overlay.
            # Handle API differences between Scanpy versions (spot_size vs size).
            try:
                fig = sc.pl.spatial(
                    adata,
                    color=domain_col,
                    library_id=slice_id,
                    spot_size=spot_size,
                    legend_loc="right margin",
                    frameon=False,
                    show=True,
                    return_fig=True
                )
                if save:
                    fig[0].figure.savefig(save, bbox_inches="tight")
            except TypeError:
                # Older Scanpy uses `size` instead of `spot_size`
                fig = sc.pl.spatial(
                    adata,
                    color=domain_col,
                    library_id=slice_id,
                    size=spot_size,
                    legend_loc="on data",
                    frameon=False,
                    show=True,
                    return_fig=True
                )
                if save:
                    fig[0].figure.savefig(save, bbox_inches="tight")
        else:
            # Plain scatter using spatial coordinates
            xy = adata.obsm["spatial"]
            labels = adata.obs[domain_col].astype("category")

            fig, ax = plt.subplots(figsize=(6, 6))
            # Map categories to integer codes for colorization
            codes = labels.cat.codes.to_numpy()
            sca = ax.scatter(
                xy[:, 0], xy[:, 1],
                c=codes,
                s=(spot_size if spot_size is not None else 8),
                alpha=0.9
            )

            # Spatial coordinate systems typically have origin at top-left; invert Y for a natural view
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Spatial domains: {domain_col}")

            # Build a category legend (labels only; colors follow the scatter's colormap)
            handles = []
            for lab in labels.cat.categories:
                handles.append(plt.Line2D([0], [0], marker='o', linestyle='', markersize=6, label=str(lab)))
            ax.legend(handles=handles, title="Domain", loc="best", frameon=False, fontsize=8)

            if save:
                plt.savefig(save, bbox_inches="tight", dpi=300)
            plt.show()

    def plot_embedding_domains(
            self,
            adata,
            domain_col: str = "domain",
            basis: str = "umap",
            compute_if_missing: bool = True,
            s: int = 8,
            save: Optional[str] = None
    ) -> None:
        """
        Visualize domains on a low-dimensional embedding (UMAP/t-SNE/PCA/etc.).

        Args:
            adata (anndata.AnnData):
                AnnData object with embeddings in `adata.obsm` (e.g., 'X_umap', 'X_tsne', 'X_pca').
            domain_col (str, default: "domain"):
                Column in `adata.obs` with domain labels to color by.
            basis (str, default: "umap"):
                Embedding name to use. Typical values: "umap", "tsne", "pca".
                The function will look for `adata.obsm[f"X_{basis}"]`.
            compute_if_missing (bool, default: True):
                If True and the chosen embedding is missing, compute a minimal pipeline:
                  - PCA (always), neighbors (for UMAP/t-SNE), and the requested embedding.
                If False and the embedding is missing, raise an error.
            s (int, default: 8):
                Point size (marker size) for the embedding scatter.
            save (Optional[str], default: None):
                File path to save the figure (e.g., "umap_domains.png"). If None, do not save.

        Returns:
            None
        """
        key = f"X_{basis}"

        # Compute a minimal embedding pipeline if missing and allowed
        if key not in adata.obsm:
            if not compute_if_missing:
                raise ValueError(f"Missing `adata.obsm['{key}']`. Precompute {basis} or set compute_if_missing=True.")

            # Always compute PCA first
            sc.pp.pca(adata, n_comps=50, svd_solver="arpack")

            # For UMAP/t-SNE we need neighbors; for PCA-only we don't
            if basis.lower() in ("umap", "tsne", "tsne"):
                sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

            # Compute the requested embedding
            if basis.lower() == "umap":
                sc.tl.umap(adata)
            elif basis.lower() in ("tsne", "tsne"):
                sc.tl.tsne(adata)
            elif basis.lower() == "pca":
                # PCA coords live in 'X_pca' after sc.pp.pca
                pass
            else:
                raise NotImplementedError(f"Unknown basis '{basis}'. Use 'umap', 'tsne', or 'pca'.")

            key = f"X_{basis}"

        # Prefer Scanpy's high-level embedding plot (handles legends/palettes well)
        try:
            fig = sc.pl.embedding(
                adata,
                basis=basis,
                color=domain_col,
                s=s,
                legend_loc="on data",
                frameon=False,
                return_fig=True,
                show=False
            )
            ax = fig.axes[0]

            if save:
                fig.savefig(save, bbox_inches="tight", dpi=300)
            plt.show()

        except TypeError:
            # Fallback for older Scanpy without return_fig: manual Matplotlib scatter
            coords = adata.obsm[key]
            labels = adata.obs[domain_col].astype("category")
            fig, ax = plt.subplots(figsize=(6, 6))
            codes = labels.cat.codes.to_numpy()
            ax.scatter(coords[:, 0], coords[:, 1], c=codes, s=s, alpha=0.9)
            ax.set_xlabel(f"{basis.upper()} 1")
            ax.set_ylabel(f"{basis.upper()} 2")
            ax.set_title(f"{basis.upper()} domains: {domain_col}")

            handles = []
            for lab in labels.cat.categories:
                handles.append(plt.Line2D([0], [0], marker='o', linestyle='', markersize=6, label=str(lab)))
            ax.legend(handles=handles, title="Domain", loc="best", frameon=False, fontsize=8)

            if save:
                plt.savefig(save, bbox_inches="tight", dpi=300)
            plt.show()
