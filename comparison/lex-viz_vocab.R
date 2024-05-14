library(tidyverse)
library(here)
library(broom)
library(reticulate)

oc_dir <- here("evals/lex-viz_vocab/openclip/")

np <- import("numpy")

oc_files <- list.files(oc_dir)

## make human data
get_human_data_vv <- function(manifest_file = "assets/lex-viz_vocab/manifest.csv", 
                              data_file = "evals/lex-viz_vocab/trial_level_for_modeling.csv") {
  manifest_long <- read_csv(manifest_file) |> 
    mutate(trial = seq_along(text1)) |> 
    pivot_longer(cols = -c(text1, trial), names_to = "image", values_to = "item") |> 
    mutate(item = str_extract(item, "(?<=images/)[a-z]+(?=.jpg)"))
  human_data <- read_csv(data_file) |> 
    group_by(targetWord, age_group) |> 
    count(answerWord) |> 
    left_join(manifest_long, by = join_by(targetWord == text1, answerWord == item)) |> 
    mutate(prob = n / sum(n)) |> 
    select(-n, -answerWord) |> 
    pivot_wider(names_from = image, values_from = prob) |> 
    mutate(across(starts_with("image"), \(x) replace_na(x, 0))) |> 
    rename(text1 = targetWord) |> 
    pivot_longer(cols = starts_with("image"), names_to = "image", values_to = "prob")
  human_data
}

human_data <- get_human_data_vv()

## comparison fxn
compare_vv <- function(model_data, human_data) {
  model_probs <- model_data |> 
    mutate(across(starts_with("image"), exp),
           rowsum = image1 + image2 + image3 + image4,
           across(starts_with("image"), \(x) x / rowsum)) |> 
    select(trial, starts_with("image")) |> 
    pivot_longer(cols = -trial, names_to = "image", values_to = "prob")
  all_cors <- human_data |> 
    pivot_wider(names_from = age_group, names_prefix = "age_", values_from = "prob") |> 
    left_join(model_probs, by = join_by(trial, image)) |> 
    ungroup() |> 
    filter(image != "image1") |> 
    summarise(across(starts_with("age_"), \(x) cor(x, prob, use = "pairwise.complete.obs"))) |> 
    pivot_longer(everything(), names_to = "age", names_prefix = "age_", values_to = "cor") |> 
    mutate(age = as.numeric(age))
  all_cors
}

openclip_cors <- lapply(oc_files, \(ocf) {
  epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
  res <- np$load(here(oc_dir, ocf)) |> 
    as_tibble() |> 
    `colnames<-`(value = c("image1", "image2", "image3", "image4")) |> 
    mutate(trial = seq_along(image1))
  cors <- compare_vv(res, human_data) |> 
    mutate(epoch = epoch)
}) |> bind_rows()

ggplot(openclip_cors, aes(x = log(epoch), y = cor, col = log(age))) +
  geom_point() +
  geom_smooth(aes(group = age), method = "lm") +
  scale_colour_continuous() +
  theme_classic() +
  labs(x = "log(Epoch)",
       y = "Error correlation",
       col = "log(Age)")

mod_lm <- lm(cor ~ log(epoch) * age, data = openclip_cors) |> 
  tidy()
