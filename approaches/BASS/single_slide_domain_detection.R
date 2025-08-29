# Required input fields:
#    a) gene expression count data; [N_slides = 1, N_genes, N_cells]
#.   b) spatial coordinates; [N_slides = 1, N_cells, 2]
#.   c) # layers
# Expected output:
#    a csv file of predicted domain layers [N_slides = 1, N_cells]

if (!requireNamespace("SummarizedExperiment", quietly = TRUE)) {
  install.packages("BiocManager")
  BiocManager::install("SummarizedExperiment")
}
if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
  BiocManager::install("SingleCellExperiment")
}

devtools::install_github("zhengli09/BASS")

library(BASS)
library(SummarizedExperiment)   # 提供 assayNames(), assay(), assays()
library(SingleCellExperiment)   # 提供 reducedDim(), reducedDimNames()
library(zellkonverter)





# step 1: read from h5ad.
sce <- readH5AD("~/Documents/misc/biology/Ji Lab/ST_FM_benchmark/ST_FM_benchmark/approaches/data/1_visium.h5ad", use_hdf5 = TRUE)

# gene expression count data; [N_slides, N_genes, N_cells]
pick_assay <- function(sce) {
  an <- assayNames(sce)
  if ("counts" %in% an) return("counts")
  if ("X" %in% an)      return("X")
  stop("cannot find `X` or `counts` in h5ad file.")
}
assay_name <- pick_assay(sce)
expr <- assay(sce, assay_name)

# spatial coordinates; [N_slides, N_cells, 2]
get_spatial_xy <- function(sce) {
  # try to find common 2D spatial coordination names in rowData.
  rdn <- reducedDimNames(sce)
  cand <- c("spatial", "X_spatial", rdn[grepl("spatial", rdn, ignore.case = TRUE)])
  for (nm in unique(cand)) {
    if (nm %in% rdn) {
      xy <- reducedDim(sce, nm)
      if (!is.null(xy) && ncol(xy) >= 2) {
        rownames(xy) <- colnames(sce)  # align cell ID
        return(as.matrix(xy[, 1:2, drop = FALSE]))
      }
    }
  }
  # find common coordination names in colData.
  cd <- as.data.frame(colData(sce))
  candidates <- list(
    c("pxl_col_in_fullres","pxl_row_in_fullres"),
    c("array_col","array_row"),
    c("x","y"),
    c("imagecol","imagerow")
  )
  for (kk in candidates) {
    if (all(kk %in% colnames(cd))) {
      xy <- as.matrix(cd[, kk, drop = FALSE])
      rownames(xy) <- colnames(sce)
      return(xy)
    }
  }
  stop("cannot find spatial coordinates (reducedDims/colData)")
}
xy_all <- get_spatial_xy(sce)


## ---- Slice splitting (single slice is treated as 1 slice)）----
slice_col  <- NULL
if (is.null(slice_col)) {
  slice_index <- factor(rep("slice1", ncol(sce)))
} else {
  if (!slice_col %in% colnames(colData(sce))) {
    stop(sprintf("colData does not exist slice name '%s'", slice_col))
  }
  slice_index <- factor(colData(sce)[[slice_col]])
}

## ---- Assembled as BASS input: cntm (gene × cell), xym (cell × 2)----
levels_slice <- levels(slice_index)
cntm <- vector("list", length(levels_slice))
xym  <- vector("list", length(levels_slice))
names(cntm) <- names(xym) <- levels_slice

for (i in seq_along(levels_slice)) {
  lev <- levels_slice[i]
  cells_i <- colnames(sce)[slice_index == lev]
  if (length(cells_i) == 0) stop(sprintf("slice %s does not have any cell", lev))
  
  mat_i <- expr[, cells_i, drop = FALSE]      # genes x cells
  xy_i  <- xy_all[cells_i, , drop = FALSE]    # cells x 2
  
  # consistency check
  stopifnot(identical(colnames(mat_i), rownames(xy_i)))
  
  cntm[[i]] <- as.matrix(mat_i)
  xym[[i]]  <- as.matrix(xy_i)
}






# step 2: run BASS for domain detection

# when the purpose of the analysis is solely to detect spatial domains,
# C can be specified to be a relatively large number (e.g. C = 20) while exploring R.
R_domains <- 6
Bobj <- createBASSObject(cntm, xym, C = 20, R = R_domains, beta_method = "SW")
Bobj <- BASS.preprocess(
  Bobj,
  doLogNormalize = TRUE,
  geneSelect     = "sparkx",
  nSE            = 3000,
  doPCA          = TRUE,
  scaleFeature   = FALSE,
  nPC            = 20
)
Bobj <- BASS.run(Bobj)
Bobj <- BASS.postprocess(Bobj)




# step 3: write results to csv
zlabels <- Bobj@results$z   # list，Length = number of slices; each element is the domain label of each cell/spot in the slice
clabels <- Bobj@results$c   # Optional: Cell type labeling

out_list <- lapply(seq_along(levels_slice), function(i){
  data.frame(
    slice  = levels_slice[i],
    cell   = colnames(cntm[[i]]),
    x      = xym[[i]][,1],
    y      = xym[[i]][,2],
    domain = zlabels[[i]],
    row.names = NULL
  )
})
bass_domains_df <- do.call(rbind, out_list)
head(bass_domains_df)
write.csv(bass_domains_df, "~/Documents/misc/biology/Ji Lab/ST_FM_benchmark/ST_FM_benchmark/approaches/BASS/data/1_visium_bass_domains.csv",
          row.names = FALSE)

