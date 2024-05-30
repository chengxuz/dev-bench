library(tidyverse)
library(here)
library(glue)
library(broom)
library(latex2exp)
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

# compare_vv_fixed <- function(model_data, human_data) {
#   human_data_nested <- human_data |> 
#     select(age_bin, trial, starts_with("image")) |> 
#     nest(data = -age_bin) |> 
#     mutate(opt_kl = lapply(data, \(d) {get_reg_kl(d, model_data, opt$solution)}),
#            kl = sapply(opt_kl, \(r) {r$objective}),
#            beta = sapply(opt_kl, \(r) {r$solution}),
#            iters = sapply(opt_kl, \(r) {r$iterations})) |> 
#     select(-data, -opt_kl)
# }

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

# openclip_div_allepoch <- human_data_vv |>
#   select(age_bin, trial, starts_with("image")) |>
#   nest(data = -age_bin) |>
#   mutate(
#     opt_kl = lapply(data, \(d) {
#       nloptr(x0 = 1,
#              eval_f = partial(get_opt_kl_allepoch, 
#                               human_probs_wide = d,
#                               model_data = openclip_res),
#              lb = .1, ub = 15,
#              opts = list(algorithm = "NLOPT_GN_DIRECT_L",
#                          ftol_abs = 1e-4,
#                          maxeval = 200))
#     }),
#     kl = sapply(opt_kl, \(r) r$objective),
#     beta = sapply(opt_kl, \(r) r$solution),
#     iters = sapply(opt_kl, \(r) r$iterations)) |> 
#   mutate(all_kl = map2(data, beta, \(d, b) {
#     lapply(oc_files, \(ocf) {
#       epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
#       res <- np$load(here(oc_dir, ocf)) |>
#         as_tibble() |>
#         `colnames<-`(value = c("image1", "image2", "image3", "image4")) |>
#         mutate(trial = seq_along(image1))
#       val <- get_reg_kl(d, res, b)
#       val$epoch = epoch
#       val
#     }) |> bind_rows()
#   })) |> select(-data) |> unnest(all_kl)

# openclip_div_allages <- lapply(oc_files, \(ocf) {
#   epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
#   res <- np$load(here(oc_dir, ocf)) |> 
#     as_tibble() |> 
#     `colnames<-`(value = c("image1", "image2", "image3", "image4")) |> 
#     mutate(trial = seq_along(image1))
#   kls <- compare_vv_fixed(res, human_data_vv) |> 
#     mutate(epoch = epoch)
# }) |> bind_rows()

vv_oc <- ggplot(openclip_div, aes(x = log(epoch), y = kl, col = as.factor(age_bin))) +
  geom_point() +
  geom_smooth(span = 1) +#, method = "lm") +
  # scale_colour_continuous() +
  theme_classic() +
  labs(x = "log(Epoch)",
       y = TeX("$D^*_{KL}$"),
       col = "Age")

ggsave("comparison/lex-vv-oc.png", 
       vv_oc, 
       width = 6.2, height = 4.2, units = "in")

# mod_vv <- lm(cor ~ log(epoch) * age, data = openclip_cors) |> 
#   tidy()

## comparisons for other models
vv_files <- c(list.files("evals/lex-viz_vocab", pattern = "*.npy"), "openclip/openclip_epoch_256.npy")

other_res <- lapply(vv_files, \(vvf) {
  res <- np$load(here("evals/lex-viz_vocab", vvf)) |> 
    as_tibble()
  res <- res |> 
    `colnames<-`(value = c("image1", "image2", "image3", "image4")) |> 
    mutate(trial = seq_along(image1))
  acc <- res |> 
    mutate(correct = image1 > image2 & image1 > image3 & image1 > image4) |> 
    summarise(accuracy = mean(correct)) |> 
    pull(accuracy)
  kls <- compare_vv(res, human_data_vv) |> 
    mutate(model = vvf |> str_remove_all("vizvocab_") |> str_remove_all(".npy"),
           accuracy = acc)
}) |> bind_rows() |> 
  mutate(model = str_replace_all(model, model_rename))

vv_all <- ggplot(other_res, 
       aes(x = accuracy, y = kl, col = as.factor(age_bin), shape = model)) + 
  geom_point() + 
  scale_shape_manual(values = c(16, 1, 17, 15, 18, 0, 2, 3)) +
  theme_classic() +
  labs(x = "Accuracy",
       y = TeX("$D^*_{KL}$"),
       shape = "Model",
       col = "Age")

ggsave("comparison/lex-vv-all.png", 
       vv_all, 
       width = 6.2, height = 4.2, units = "in")

