library(tidyverse)
library(here)
library(glue)
library(reticulate)
source("comparison/stats-helper.R")

np <- import("numpy")
clip <- np$load(here("evals/lex-lwl/lwl_clip.npy"))

## make human data
get_human_data_lwl <- function(manifest_file = "assets/lex-lwl/manifest.csv",
                               data_file = "evals/lex-lwl/experiment_info/eye.tracking.csv") {
  novel_words <- c("wug", "dax", "dofa", "fep", "pifo", "kreeb", "modi", "toma")
  manifest <- read_csv(manifest_file) |> 
    filter(!text1 %in% novel_words,
           !text2 %in% novel_words)
  human_data <- read_csv(data_file) |> 
    filter(word.type == "Familiar-Familiar") |> 
    group_by(age.grp, word) |> 
    summarise(mean_prop = mean(prop),
              n = n())
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
           across(starts_with("image"), \(x) x / rowsum)) |> 
    select(-rowsum)
  
  full_data <- model_data |> 
    pivot_longer(cols = starts_with("image"),
                 names_to = c("image", "text"),
                 names_pattern = "(image[12])(text[12])",
                 values_to = "score") |> 
    pivot_wider(names_from = image,
                values_from = score) |> 
    softmax_images() |> 
    left_join(human_data_by_text, by = join_by(trial, text)) |> 
    mutate(error_model = ifelse(text == "text2", image1.x, image2.x),
           error_human = ifelse(text == "text2", image1.y, image2.y))
  
  all_cors <- cor(full_data$error_model,
                  full_data$error_human,
                  use = "pairwise.complete.obs")
}

## comparisons for openclip
openclip_cors <- lapply(oc_files, \(ocf) {
  epoch <- str_match(ocf, "[0-9]+")[1] |> as.numeric()
  res <- np$load(here(oc_dir, ocf)) |> 
    as_tibble() |> 
    `colnames<-`(value = c("image1text1", "image2text1", "image1text2", "image2text2")) |> 
    mutate(trial = seq_along(image1text1))
  cors <- compare_wg(res, human_data_wg)
  tibble(cor = cors,
         epoch = epoch)
}) |> bind_rows()

ggplot(openclip_cors, aes(x = log(epoch), y = cor)) +
  geom_point() +
  geom_smooth(method = "lm") +
  scale_colour_continuous() +
  theme_classic() +
  labs(x = "log(Epoch)",
       y = "Error correlation")

mod_wg <- lm(cor ~ log(epoch), data = openclip_cors) |> 
  tidy()
