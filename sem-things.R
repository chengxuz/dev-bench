library(tidyverse)
library(here)
library(R.matlab)
library(reticulate)
library(psych)
source("rsa-helper.R")

# read data
mat_things <- readMat(here("assets/sem-things/spose_similarity.mat"))$spose.sim

np <- import("numpy")
clip <- np$load(here("evals/sem-things/clip.npy"))

clip_things_cor <- rsa(mat_things, clip)
clip_things_perms <- get_permutations(mat_things, clip)
clip_things_p <- calc_permuted_p(clip_things_perms, clip_things_cor)
