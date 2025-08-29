import pandas as pd
from typing import Optional

def merge_df_into_adata_obs(
    adata,
    df: pd.DataFrame,
    cell_col: Optional[str] = None,
    how: str = "left",
    suffix: str = "_df",
    copy: bool = False,
    allow_duplicates: bool = False,
):
    """
    Merge a pandas DataFrame into `adata.obs` by cell name.

    Parameters
    ----------
    adata : anndata.AnnData
        Target AnnData. We'll merge onto `adata.obs` whose index == `adata.obs_names`.
    df : pd.DataFrame
        The DataFrame to merge. It must contain cell identifiers either:
          (A) in a dedicated column (pass its name via `cell_col`), or
          (B) as the DataFrame index (leave `cell_col=None`).
    cell_col : Optional[str], default None
        Column in `df` that holds cell IDs. If None, we assume `df.index` are cell IDs.
    how : {"left","inner","right","outer"}, default "left"
        Join strategy, same as pandas join. Typically "left" is safest to preserve all cells in `adata`.
    suffix : str, default "_df"
        Suffix to append to overlapping column names from `df` to avoid collisions with `adata.obs` columns.
    copy : bool, default False
        If True, do not modify `adata` in place; instead return a new AnnData with merged `obs`.
    allow_duplicates : bool, default False
        If False, duplicated cell IDs in `df` will raise an error (recommended).
        If True, we keep the first occurrence and drop subsequent duplicates.

    Returns
    -------
    anndata.AnnData
        The modified AnnData (same object if copy=False; a new object if copy=True).

    Notes
    -----
    - We always merge on index: `adata.obs.index` (cell names) vs `df`'s index.
      If `cell_col` is given, we set `df.index = df[cell_col]` before merging.
    - Overlapping columns are auto-renamed with `suffix` to prevent overwriting.
    """
    obs = adata.obs

    # 1) Prepare df with cell IDs on index
    df_indexed = df.copy()
    if cell_col is not None:
        if cell_col not in df_indexed.columns:
            raise ValueError(f"`cell_col='{cell_col}'` not found in df columns.")
        df_indexed = df_indexed.set_index(cell_col, drop=True)

    # 2) Handle duplicate cell IDs
    dup_mask = df_indexed.index.duplicated(keep="first")
    if dup_mask.any():
        dups = df_indexed.index[dup_mask].unique().tolist()
        if not allow_duplicates:
            raise ValueError(f"Found duplicated cell IDs in df index: {dups[:10]}{' ...' if len(dups)>10 else ''}")
        df_indexed = df_indexed[~dup_mask]

    # 3) Avoid column name collisions
    df_indexed = df_indexed.rename(columns={c: f"{c}{suffix}" for c in df_indexed})

    # 4) Join (on index)
    joined = obs.join(df_indexed, how=how)

    # 6) Assign back to adata (in-place or copy)
    if copy:
        ad = adata.copy()
        ad.obs = joined
        target = ad
    else:
        adata.obs = joined
        target = adata

    return target
