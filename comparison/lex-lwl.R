library(tidyverse)
library(here)
library(glue)
library(latex2exp)
library(reticulate)
source("comparison/stats-helper.R")

np <- import("numpy")
oc_dir <- here("evals/lex-lwl/openclip/")
oc_files <- list.files(oc_dir)

## make human data
get_human_data_lwl <- function(manifest_file = "assets/lex-lwl/manifest.csv",
                               data_file = "evals/lex-lwl/experiment_info/eye.tracking.csv") {
  novel_words <- c("wug", "dax", "dofa", "fep", "pifo", "kreeb", "modi", "toma")
  manifest <- read_csv(manifest_file) |> 
    mutate(trial = seq_along(text1)) |> 
    filter(!text1 %in% novel_words,
           !text2 %in% novel_words) |> 
    select(trial, text1, text2) |> 
    pivot_longer(cols = c(text1, text2), names_to = "options", values_to = "word") |> 
    mutate(trial = glue("{trial}_{options}")) |> 
    select(trial, word, options)
  human_data <- read_csv(data_file) |> 
    filter(word.type == "Familiar-Familiar") |> 
    group_by(age.grp, word) |> 
    summarise(mean_prop = mean(prop),
              n = n()) |> 
    left_join(manifest, by = join_by(word)) |> 
    mutate(image1 = ifelse(options == "text1", mean_prop, 1-mean_prop),
           image2 = ifelse(options == "text2", mean_prop, 1-mean_prop))
}

human_data_lwl <- get_human_data_lwl()

## comparison fxn
compare_lwl <- function(model_data, human_data) {
  model_data_by_text <- model_data |> 
    pivot_longer(cols = starts_with("image"),
                 names_to = c("image", "text"),
                 names_pattern = "(image[12])(text[12])",
                 values_to = "score") |> 
    pivot_wider(names_from = image,
                values_from = score) |> 
    mutate(trial = glue("{trial}_{text}"),
           correct = ifelse(text == "text1", image1 > image2, image2 > image1)) |> 
    filter(trial %in% human_data$trial)
  
  human_data_nested <- human_data |> 
    select(age_bin = age.grp, trial, starts_with("image")) |> 
    nest(data = -age_bin) |> 
    mutate(opt_kl = lapply(data, \(d) {get_opt_kl(d, model_data_by_text)}),
           kl = sapply(opt_kl, \(r) {r$objective}),
           beta = sapply(opt_kl, \(r) {r$solution}),
           iters = sapply(opt_kl, \(r) {r$iterations}),
           accuracy = mean(model_data_by_text$correct, na.rm = TRUE)) |> 
    select(-data, -opt_kl)
}

## comparisons for openclip
openclip_div <- lapply(oc_files, \(ocf) {
  epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
  res <- np$load(here(oc_dir, ocf)) |> 
    as_tibble() |> 
    `colnames<-`(value = c("image1text1", "image2text1", "image1text2", "image2text2")) |> 
    mutate(trial = seq_along(image1text1))
  kls <- compare_lwl(res, human_data_lwl) |> 
    mutate(epoch = epoch)
}) |> bind_rows()

lwl_oc <- ggplot(openclip_div, aes(x = log(epoch), y = kl, col = as.factor(age_bin))) +
  geom_point() +
  geom_smooth(span = 1) +#, method = "lm") +
  # scale_colour_continuous() +
  theme_classic() +
  labs(x = "log(Epoch)",
       y = TeX("$D^*_{KL}$"),
       col = "Age")

ggsave("comparison/lex-lwl-oc.png", 
       lwl_oc, 
       width = 6.2, height = 4.2, units = "in")

# mod_lwl <- lm(cor ~ log(epoch) * age, data = openclip_cors) |> 
#   tidy()

## comparisons for other models
lwl_files <- c(list.files("evals/lex-lwl", pattern = "*.npy"), "openclip/openclip_epoch_256.npy")

other_res <- lapply(lwl_files, \(lwlf) {
  res <- np$load(here("evals/lex-lwl", lwlf)) |> 
    as_tibble()
  res <- res |> 
    `colnames<-`(value = c("image1text1", "image2text1", "image1text2", "image2text2")) |> 
    mutate(trial = seq_along(image1text1))
  acc <- res |>
    mutate(correct = case_when(
      trial %in% c(4, 5, 8, 11) ~ image1text1 > image2text1,
      trial %in% c(2, 3, 15, 16) ~ image2text2 > image1text2)) |>
    summarise(accuracy = mean(correct, na.rm = TRUE)) |>
    pull(accuracy)
  kls <- compare_lwl(res, human_data_lwl) |> 
    mutate(model = lwlf |> str_remove_all("lwl_") |> str_remove_all(".npy"),
           accuracy = acc)
}) |> bind_rows() |> 
  mutate(model = str_replace_all(model, model_rename))

lwl_all <- ggplot(other_res, 
       aes(x = accuracy, y = kl, col = as.factor(age_bin), shape = model)) + 
  geom_point() + 
  scale_shape_manual(values = c(16, 1, 17, 15, 18, 0, 2, 3)) +
  theme_classic() +
  labs(x = "Accuracy",
       y = TeX("$D^*_{KL}$"),
       shape = "Model",
       col = "Age")

ggsave("comparison/lex-lwl-all.png", 
       lwl_all, 
       width = 6.2, height = 4.2, units = "in")

