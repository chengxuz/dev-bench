library(tidyverse)
library(here)
library(broom)
library(reticulate)
source("comparison/stats-helper.R")

np <- import("numpy")
oc_dir <- here("evals/lex-viz_vocab/openclip/")
oc_files <- list.files(oc_dir)

## make human data
get_human_data_vv <- function(manifest_file = "assets/lex-viz_vocab/manifest.csv", 
                              data_file = "evals/lex-viz_vocab/trial_level_for_modeling.csv") {
  manifest_long <- read_csv(manifest_file) |> 
    mutate(trial = seq_along(text1)) |> 
    pivot_longer(cols = -c(text1, trial), names_to = "image", values_to = "item") |> 
    mutate(item = str_extract(item, "(?<=images/)[a-z]+(?=.jpg)"))
  human_data <- read_csv(data_file) |> 
    mutate(age_bin = case_when(
      age_group <= 5 ~ 4,
      age_group <= 8 ~ 7,
      age_group <= 11 ~ 10,
      .default = 25
    )) |> 
    group_by(targetWord, age_bin) |> 
    count(answerWord) |> 
    left_join(manifest_long, by = join_by(targetWord == text1, answerWord == item)) |> 
    mutate(prob = n / sum(n)) |> 
    select(-n, -answerWord) |> 
    arrange(trial, age_bin, image) |> 
    pivot_wider(names_from = image, values_from = prob) |> 
    mutate(across(starts_with("image"), \(x) replace_na(x, 0))) |> 
    rename(text1 = targetWord) |> 
    ungroup()
    # pivot_longer(cols = starts_with("image"), names_to = "image", values_to = "prob") # |> 
    # arrange(trial, age_group, image)
  human_data
}

human_data_vv <- get_human_data_vv()

## comparison fxn
compare_vv <- function(model_data, human_data) {
  human_data_nested <- human_data |> 
    select(age_bin, trial, starts_with("image")) |> 
    nest(data = -age_bin) |> 
    mutate(opt_kl = lapply(data, \(d) {get_opt_kl(d, model_data)}),
           kl = sapply(opt_kl, \(r) {r$objective}),
           beta = sapply(opt_kl, \(r) {r$solution}),
           iters = sapply(opt_kl, \(r) {r$iterations})) |> 
    select(-data, -opt_kl)
}

## comparisons for openclip
openclip_div <- lapply(oc_files, \(ocf) {
  epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
  res <- np$load(here(oc_dir, ocf)) |> 
    as_tibble() |> 
    `colnames<-`(value = c("image1", "image2", "image3", "image4")) |> 
    mutate(trial = seq_along(image1))
  kls <- compare_vv(res, human_data_vv) |> 
    mutate(epoch = epoch)
}) |> bind_rows()

ggplot(openclip_div, aes(x = log(epoch), y = kl, col = age_bin)) +
  geom_point() +
  geom_smooth(aes(group = age_bin)) +#, method = "lm") +
  scale_colour_continuous() +
  theme_classic() +
  labs(x = "log(Epoch)",
       y = "Response KL divergence",
       col = "log(Age)")

mod_vv <- lm(cor ~ log(epoch) * age, data = openclip_cors) |> 
  tidy()
