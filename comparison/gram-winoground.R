library(tidyverse)
library(here)
library(reticulate)

oc_dir <- here("evals/gram-winoground/openclip/")

np <- import("numpy")

oc_files <- list.files(oc_dir)

results <- lapply(oc_files, \(ocf) {
  res <- np$load(here(oc_dir, ocf)) |> 
    as_tibble() |> 
    `colnames<-`(value = c("image1text1", "image2text1", "image1text2", "image2text2")) |> 
    mutate(trial = seq_along(image1text1),
           correct = image1text1 > image1text2 & image2text2 > image2text1,
           # target_prob = exp(image1) / rowSums(cbind(exp(image1), exp(image2), exp(image3), exp(image4))),
           epoch = str_match(ocf, "[0-9]+")[1] |> as.numeric())
}) |> bind_rows()

ggplot(results |> group_by(epoch) |> summarise(prop_cor = sum(correct) / 171), aes(x = epoch, y = prop_cor)) +
  geom_point(col = "#3c78d8") +
  geom_smooth(method = "loess", col = "#3c78d8") +
  geom_hline(yintercept = .25, lty = "dashed") +
  theme_classic() +
  labs(x = "Epoch", y = "Accuracy")

ggplot(results |> mutate(correct = correct |> as.numeric()), aes(x = epoch, y = correct)) +
  geom_point(alpha = .1, col = "#3c78d8",
             position = position_jitter(width = 2, height = .05)) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"),
              col = "#3c78d8") +
  geom_hline(yintercept = .25, lty = "dashed") +
  theme_classic() +
  labs(x = "Epoch", y = "Accuracy")

