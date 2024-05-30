library(tidyverse)
library(here)
library(glue)
library(latex2exp)
library(reticulate)
source("comparison/stats-helper.R")

np <- import("numpy")
oc_dir <- here("evals/gram-winoground/openclip/")
oc_files <- list.files(oc_dir)

## make human data
get_human_data_wg <- function(manifest_file = "assets/gram-winoground/manifest.csv",
                              data_file = "evals/gram-winoground/human.jsonl") {
  included_trials <- read_csv(manifest_file) |> 
    mutate(trial = as.numeric(str_extract(image1, "(?<=ex_)[0-9]+(?=_img)")) + 1) |> 
    pull(trial)
  human_data <- read_lines(data_file) |> 
    str_remove_all('\\{\\"label\\"\\: \\"') |> 
    str_replace_all('\\", \\"score\\"\\: ', ',') |> 
    str_remove_all('\\}') |> 
    I() |> 
    read_csv(col_names = c("label", "score")) |> 
    separate_wider_delim(label, delim = "_", names = c("trial", "text", "image")) |> 
    mutate(trial = as.numeric(trial) + 1,
           image = (str_sub(image, 2) |> as.numeric()) + 1,
           text = (str_sub(text, 2) |> as.numeric()) + 1,
           pair = glue("image{image}text{text}")) |> 
    select(-image, -text) |> 
    filter(trial %in% included_trials) |> 
    mutate(trial = sapply(trial, \(t) which(included_trials == t)))
}

human_data_wg <- get_human_data_wg()

## comparison fxn
compare_wg <- function(model_data, human_data) {
  human_data_by_text <- human_data |> 
    separate_wider_regex(cols = pair,
                         patterns = c(image = "(?:image[12])", 
                                      text = "(?:text[12])")) |> 
    pivot_wider(names_from = image,
                values_from = score) |> 
    mutate(rowsum = rowSums(across(starts_with("image"))),
           across(starts_with("image"), \(x) x / rowsum),
           trial = glue("{trial}_{text}")) |> 
    select(-rowsum)
  
  model_data_by_text <- model_data |> 
    pivot_longer(cols = starts_with("image"),
                 names_to = c("image", "text"),
                 names_pattern = "(image[12])(text[12])",
                 values_to = "score") |> 
    pivot_wider(names_from = image,
                values_from = score) |> 
    mutate(trial = glue("{trial}_{text}"))
  
  opt_kl = get_opt_kl(human_data_by_text, model_data_by_text)
  tibble(kl = opt_kl$objective,
         beta = opt_kl$solution,
         iters = opt_kl$iterations)
}

## comparisons for openclip
openclip_cors <- lapply(oc_files, \(ocf) {
  epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
  res <- np$load(here(oc_dir, ocf)) |> 
    as_tibble() |> 
    `colnames<-`(value = c("image1text1", "image2text1", "image1text2", "image2text2")) |> 
    mutate(trial = seq_along(image1text1))
  kls <- compare_wg(res, human_data_wg) |> 
    mutate(epoch = epoch)
}) |> bind_rows()

wg_oc <- ggplot(openclip_cors, aes(x = log(epoch), y = kl)) +
  geom_point() +
  geom_smooth() +
  scale_colour_continuous() +
  theme_classic() +
  labs(x = "log(Epoch)",
       y = TeX("$D^*_{KL}$"))

ggsave("comparison/gram-wg-oc.png", 
       wg_oc, 
       width = 6.2, height = 4.2, units = "in")

# mod_wg <- lm(cor ~ log(epoch), data = openclip_cors) |> 
#   tidy()

## comparisons for other models
wg_files <- c(list.files("evals/gram-winoground", pattern = "*.npy"), "openclip/openclip_epoch_256.npy")

other_res <- lapply(wg_files, \(wgf) {
  res <- np$load(here("evals/gram-winoground", wgf)) |> 
    as_tibble()
  res <- res |> 
    `colnames<-`(value = c("image1text1", "image2text1", "image1text2", "image2text2")) |> 
    mutate(trial = seq_along(image1text1))
  acc <- res |> 
    mutate(correct = ((image1text1 > image1text2) + (image2text2 > image2text1))/2) |> 
    summarise(accuracy = mean(correct)) |> 
    pull(accuracy)
  kls <- compare_wg(res, human_data_wg) |> 
    mutate(model = wgf |> str_remove_all("winoground_") |> str_remove_all(".npy"),
           accuracy = acc)
}) |> bind_rows() |> 
  mutate(model = str_replace_all(model, model_rename))

wg_all <- ggplot(other_res, 
       aes(x = accuracy, y = kl, shape = model)) + 
  geom_point() + 
  scale_shape_manual(values = c(16, 1, 17, 15, 18, 0, 2, 3)) +
  theme_classic() +
  labs(x = "Accuracy",
       y = TeX("$D^*_{KL}$"),
       shape = "Model")

ggsave("comparison/gram-wg-all.png", 
       wg_all, 
       width = 6.2, height = 4.2, units = "in")
