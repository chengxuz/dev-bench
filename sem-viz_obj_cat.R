library(tidyverse)
library(here)
library(R.matlab)
library(reticulate)
library(psych)
source("rsa-helper.R")

# read data
mat_4mo <- readMat(here("assets/sem-viz_obj_cat/original/Matrices/4months.mat"))$Matwindow[1][[1]]
mat_10mo <- readMat(here("assets/sem-viz_obj_cat/original/Matrices/10months.mat"))$Matwindow[1][[1]]
mat_19mo <- readMat(here("assets/sem-viz_obj_cat/original/Matrices/19months.mat"))$Matwindow[1][[1]]

np <- import("numpy")
clip <- np$load(here("evals/sem-viz_obj_cat/clip.npy"))

# get mean matrices and correlations
collapse_matrix <- function(mat) {
  mat[is.nan(mat)] <- NA
  apply(mat, c(1,2), mean, na.rm = TRUE)
}

mat_4mo_mean <- collapse_matrix(mat_4mo)
mat_10mo_mean <- collapse_matrix(mat_10mo)
mat_19mo_mean <- collapse_matrix(mat_19mo)

clip_4mo_cor <- rsa(mat_4mo_mean, clip)
clip_10mo_cor <- rsa(mat_10mo_mean, clip)
clip_19mo_cor <- rsa(mat_19mo_mean, clip)

# permutation test
clip_4mo_perms <- get_permutations(mat_4mo_mean, clip)
clip_10mo_perms <- get_permutations(mat_10mo_mean, clip)
clip_19mo_perms <- get_permutations(mat_19mo_mean, clip)

clip_4mo_p <- calc_permuted_p(clip_4mo_perms, clip_4mo_cor)
clip_10mo_p <- calc_permuted_p(clip_10mo_perms, clip_10mo_cor)
clip_19mo_p <- calc_permuted_p(clip_19mo_perms, clip_19mo_cor)

# noise ceiling---notably, all very low
get_noise_ceiling <- function(mats) {
  sapply(1:dim(mats)[3], \(i) {
    mat1 <- mats[,,i]
    mato_mean <- collapse_matrix(mats[,,-1])
    rsa(mat1, mato_mean)
  })
}

noise_4mo <- get_noise_ceiling(mat_4mo)
noise_10mo <- get_noise_ceiling(mat_10mo)
noise_19mo <- get_noise_ceiling(mat_19mo)

