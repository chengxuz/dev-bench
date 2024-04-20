library(tidyverse)

rsa <- function(mat1, mat2, method = "spearman") {
  mat1_lower <- mat1[lower.tri(mat1)]
  mat2_lower <- mat2[lower.tri(mat2)]
  cor(mat1_lower, mat2_lower, use = "pairwise.complete.obs", method = method)
}

run_permutation_test <- function(mat1, mat2, method = "spearman", nsim = 1000, seed = 42) {
  set.seed(seed)
  sims <- sapply(1:nsim, \(sim) {
    idx <- sample(nrow(mat1))
    mat1_perm <- mat1[idx, idx]
    rsa(mat1_perm, mat2, method = method)
  })
  obs_cor <- rsa(mat1, mat2)
  sum(abs(obs_cor) < abs(sims)) / nsim
}