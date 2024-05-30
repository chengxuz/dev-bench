library(tidyverse)
library(here)
library(R.matlab)
library(reticulate)
library(psych)
source("comparison/stats-helper.R")

np <- import("numpy")
oc_dir <- here("evals/sem-viz_obj_cat/openclip/")
oc_files <- list.files(oc_dir)
# clip <- np$load(here("evals/sem-viz_obj_cat/clip.npy"))

## make human_data
collapse_matrix <- function(mat) {
  mat[is.nan(mat)] <- NA
  apply(mat, c(1,2), mean, na.rm = TRUE)
}

human_data_voc <- tibble(age = c(0.333, 0.833, 1.583),
                         mat = list(readMat(here("assets/sem-viz_obj_cat/original/Matrices/4months.mat"))$Matwindow[1][[1]],
                                    readMat(here("assets/sem-viz_obj_cat/original/Matrices/10months.mat"))$Matwindow[1][[1]],
                                    readMat(here("assets/sem-viz_obj_cat/original/Matrices/19months.mat"))$Matwindow[1][[1]]) |> 
                           lapply(collapse_matrix))

# clip_4mo_cor <- rsa(mat_4mo_mean, clip)
# clip_10mo_cor <- rsa(mat_10mo_mean, clip)
# clip_19mo_cor <- rsa(mat_19mo_mean, clip)
# 
# # permutation test
# clip_4mo_perms <- get_permutations(mat_4mo_mean, clip)
# clip_10mo_perms <- get_permutations(mat_10mo_mean, clip)
# clip_19mo_perms <- get_permutations(mat_19mo_mean, clip)
# 
# clip_4mo_p <- calc_permuted_p(clip_4mo_perms, clip_4mo_cor)
# clip_10mo_p <- calc_permuted_p(clip_10mo_perms, clip_10mo_cor)
# clip_19mo_p <- calc_permuted_p(clip_19mo_perms, clip_19mo_cor)
# 
# # noise ceiling---notably, all very low
# get_noise_ceiling <- function(mats) {
#   sapply(1:dim(mats)[3], \(i) {
#     mat1 <- mats[,,i]
#     mato_mean <- collapse_matrix(mats[,,-1])
#     rsa(mat1, mato_mean)
#   })
# }
# 
# noise_4mo <- get_noise_ceiling(mat_4mo)
# noise_10mo <- get_noise_ceiling(mat_10mo)
# noise_19mo <- get_noise_ceiling(mat_19mo)

reduce_mat <- function(mat) {
  sapply(1:8, \(i) {
    sapply(1:8, \(j) {
      mean(mat[(9*i-8):(9*i), (9*j-8):(9*j)])
    })
  })
}

## comparisons for openclip
openclip_cors <- lapply(oc_files, \(ocf) {
  epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
  res <- np$load(here(oc_dir, ocf))
  res_mat <- philentropy::distance(res, method = "cosine",
                                   mute.message = TRUE) |> 
    reduce_mat()
  cors <- human_data_voc |> 
    mutate(similarity = sapply(mat, \(m) {
      rsa(m, res_mat)
    })) |> 
    select(-mat) |> 
    mutate(epoch = epoch)
}) |> bind_rows()

voc_oc <- ggplot(openclip_cors, aes(x = log(epoch), y = similarity, col = as.factor(age))) +
  geom_point() +
  geom_smooth(span = 1) +#, method = "lm") +
  # scale_colour_continuous() +
  theme_classic() +
  labs(x = "log(Epoch)",
       y = "RSA similarity",
       col = "Age")

ggsave("comparison/sem-voc-oc.png", 
       voc_oc, 
       width = 6.2, height = 4.2, units = "in")

## comparisons for other models
voc_files <- c(list.files("evals/sem-viz_obj_cat", pattern = "*.npy"), "openclip/openclip_epoch_256.npy")

other_res <- lapply(voc_files, \(vocf) {
  res <- np$load(here("evals/sem-viz_obj_cat", vocf))
  res_mat <- philentropy::distance(res, method = "cosine",
                                   mute.message = TRUE) |> 
    reduce_mat()
  human_data_voc |> 
    mutate(similarity = sapply(mat, \(m) {
      rsa(m, res_mat)
    })) |> 
    select(-mat) |> 
    mutate(model = vocf |> str_remove_all("voc_") |> str_remove_all(".npy"))
}) |> bind_rows() |> 
  mutate(model = str_replace_all(model, model_rename))

voc_all <- ggplot(other_res |> filter(!is.na(similarity)),
                  aes(x = fct_reorder(model, similarity), y = similarity, fill = as.factor(age))) + 
  geom_col(position = "dodge") + 
  # scale_shape_manual(values = c(16, 1, 17, 15, 18, 0, 2, 3)) +
  theme_classic() +
  labs(x = "Model",
       y = "RSA similarity",
       fill = "Age") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("comparison/sem-voc-all.png", 
       voc_all, 
       width = 6.2, height = 4.2, units = "in")

