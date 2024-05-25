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
    mutate(across(starts_with("image"), \(i) replace_na(i, 0))) |> 
    nest(distribs = -trial) |> 
    filter(lapply(distribs, nrow) == 2) |> 
    mutate(kl = sapply(distribs, \(d) {
      d |> select(starts_with("image")) |> 
        as.matrix() |> 
        {\(m) quietly(KL)(m, unit = "log")["result"][[1]]}()
    }))
  combined_distribs |> 
    pull(kl) |> 
    mean(na.rm = TRUE)
}

get_opt_kl <- function(human_probs_wide, model_logits_wide) {
  mean_kl <- \(beta) {get_mean_kl(human_probs_wide, 
                                  softmax_images(model_logits_wide, beta))}
  res <- nloptr(x0 = 1, 
                eval_f = mean_kl, 
                lb = 0.025,
                ub = 40,
                opts = list(algorithm = "NLOPT_GN_DIRECT_L",
                            ftol_abs = 1e-4,
                            maxeval = 200))
}

get_reg_kl <- function(human_probs_wide, model_logits_wide, beta = 1) {
  res <- get_mean_kl(human_probs_wide, softmax_images(model_logits_wide, beta))
  list(objective = res,
       solution = beta,
       iterations = 0)
}

get_opt_kl_allage <- function(human_data, model_logits_wide, beta) {
  human_data |> 
    select(age_bin, trial, starts_with("image")) |> 
    nest(data = -age_bin) |> 
    mutate(kl = sapply(data, \(d) get_mean_kl(d, softmax_images(model_logits_wide, beta)))) |> 
    pull(kl) |> 
    mean(na.rm = TRUE)
}

get_opt_kl_allepoch <- function(human_probs_wide, model_data, beta) {
  model_data_nested <- model_data |> 
    select(epoch, trial, starts_with("image")) |> 
    nest(data = -epoch) |> 
    mutate(kl = sapply(data, \(d) get_mean_kl(human_probs_wide, softmax_images(d, beta)))) |> 
    pull(kl) |> 
    mean(na.rm = TRUE)
}
