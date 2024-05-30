library(tidyverse)
library(here)
library(R.matlab)
library(reticulate)
library(psych)
library(philentropy)
source("comparison/stats-helper.R")

np <- import("numpy")
oc_dir <- here("evals/sem-things/openclip/")
oc_files <- list.files(oc_dir)

# clip <- np$load(here("evals/sem-things/things_clip_small.npy"))

## make human_data
human_data_things <- readMat(here("assets/sem-things/spose_similarity.mat"))$spose.sim

# clip_things_cor <- rsa(mat_things, clip)
# clip_things_perms <- get_permutations(mat_things, clip)
# clip_things_p <- calc_permuted_p(clip_things_perms, clip_things_cor)

## comparisons for openclip
openclip_cors <- lapply(oc_files, \(ocf) {
  epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
  res <- np$load(here(oc_dir, ocf))
  res_mat <- philentropy::distance(res, method = "cosine",
                                   mute.message = TRUE)
  tibble(similarity = rsa(human_data_things, res_mat),
         epoch = epoch)
}) |> bind_rows()

things_oc <- ggplot(openclip_cors, aes(x = log(epoch), y = similarity)) +
  geom_point() +
  geom_smooth(span = 1) +#, method = "lm") +
  scale_colour_continuous() +
  theme_classic() +
  labs(x = "log(Epoch)",
       y = "RSA similarity")

ggsave("comparison/sem-things-oc.png", 
       things_oc, 
       width = 6.2, height = 4.2, units = "in")

## comparisons for other models
things_files <- c(list.files("evals/sem-things", pattern = "*.npy"), "openclip/openclip_epoch_256.npy")

other_res <- lapply(things_files, \(thingsf) {
  res <- np$load(here("evals/sem-things", thingsf))
  res_mat <- philentropy::distance(res, method = "cosine",
                                   mute.message = TRUE)
  tibble(similarity = rsa(human_data_things, res_mat),
         model = thingsf |> str_remove_all("things_") |> str_remove_all(".npy"))
}) |> bind_rows() |> 
  mutate(model = str_replace_all(model, model_rename))

things_all <- ggplot(other_res |> filter(!is.na(similarity)),
                     aes(x = fct_reorder(model, similarity), y = similarity)) + 
  geom_col() + 
  # scale_shape_manual(values = c(16, 1, 17, 15, 18, 0, 2, 3)) +
  theme_classic() +
  labs(x = "Model",
       y = "RSA similarity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("comparison/sem-things-all.png", 
       things_all, 
       width = 6.2, height = 4.2, units = "in")

