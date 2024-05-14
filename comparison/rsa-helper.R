library(tidyverse)

rsa <- function(mat1, mat2, method = "spearman") {
  mat1_lower <- mat1[lower.tri(mat1)]
  mat2_lower <- mat2[lower.tri(mat2)]
  cor(mat1_lower, mat2_lower, use = "pairwise.complete.obs", method = method)
}

get_permutations <- function(mat1, mat2, method = "spearman", nsim = 1000, seed = 42) {
  set.seed(seed)
  sims <- sapply(1:nsim, \(sim) {
    idx <- sample(nrow(mat1))
    mat1_perm <- mat1[idx, idx]
    rsa(mat1_perm, mat2, method = method)
  })
}

calc_permuted_p <- function(sim_cors, obs_cor) {
  sum(abs(obs_cor) < abs(sim_cors)) / length(sim_cors)
}

softmax_images <- function(data) {
  data |> 
    mutate(across(starts_with("image"), exp),
           rowsum = rowSums(across(starts_with("image"))),
           across(starts_with("image"), \(x) x / rowsum)) |> 
    select(-rowsum)
}
