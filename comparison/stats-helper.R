library(tidyverse)
library(philentropy)
library(nloptr)

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

softmax_images <- function(data, beta = 1) {
  data |> 
    mutate(across(starts_with("image"), \(i) exp(beta * i)),
           rowsum = rowSums(across(starts_with("image"))),
           across(starts_with("image"), \(i) i/rowsum)) |> 
    select(-rowsum)
}

get_mean_kl <- function(human_probs_wide, model_probs_wide) {
  combined_distribs <- bind_rows(human_probs_wide, model_probs_wide) |> 
    nest(distribs = -trial) |> 
    filter(lapply(distribs, nrow) == 2) |> 
    mutate(kl = lapply(distribs, \(d) {
      d |> select(starts_with("image")) |> 
        as.matrix() |> 
        {\(m) quietly(KL)(m, unit = "log")["result"][[1]]}()
    }) |> list_c())
  combined_distribs |> 
    pull(kl) |> 
    mean(na.rm = TRUE)
}

get_opt_kl <- function(human_probs_wide, model_logits_wide) {
  mean_kl <- \(beta) {get_mean_kl(human_probs_wide, 
                                  softmax_images(model_logits_wide, beta))}
  res <- nloptr(x0 = 1, 
                eval_f = mean_kl, 
                lb = 0.1,
                ub = 10,
                opts = list(algorithm = "NLOPT_GN_DIRECT_L",
                            ftol_abs = 1e-4,
                            maxeval = 200))
}
