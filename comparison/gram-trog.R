library(tidyverse)
library(here)
library(glue)
library(lubridate)
library(reticulate)
source("comparison/stats-helper.R")

np <- import("numpy")
oc_dir <- here("evals/gram-trog/openclip/")
oc_files <- list.files(oc_dir)

## make human data
get_human_data_trog <- function(manifest_file = "assets/gram-trog/manifest.csv",
                                data_file = "evals/gram-trog/syntax-trials-2024-05-15-full-clean.csv") {
  manifest <- read_csv(manifest_file) |> 
    mutate(trial = seq_along(text1)) |> 
    pivot_longer(cols = starts_with("image"),
                 names_to = "option",
                 values_to = "image")
  human_data <- read_csv(data_file) |> 
    mutate(age_yr = interval(parse_date_time(paste(user.birthYear, user.birthMonth, "01"), orders = "ymd"), 
                             user.CreateTime) |> as.numeric('years') |> floor()) |> 
    filter(age_yr <= 12) |> 
    select(text1 = item, answer, response, age_yr) |> 
    group_by(text1, answer) |> 
    count(response) |> 
    mutate(prop = n / sum(n, na.rm = TRUE),
           text1 = tolower(text1) |> str_replace("\\.", ""),
           image = glue("images/{response}.png")) |> 
    left_join(manifest, by = join_by(text1, image)) |> 
    arrange(trial, option) |> 
    filter(!is.na(option))
}

human_data_trog <- get_human_data_trog()

## comparison fxn
compare_trog <- function(model_data, human_data) {

  
}
