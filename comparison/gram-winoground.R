library(tidyverse)
library(here)
library(glue)
library(reticulate)

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
