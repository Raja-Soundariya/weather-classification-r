# ============================================================
# SEATTLE WEATHER PREDICTION USING MACHINE LEARNING
# Decision Tree vs Random Forest
# ============================================================


# ============================================================
# 1. LOAD LIBRARIES
# ============================================================

library(ggplot2)
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(rstudioapi)


# ============================================================
# 2. SET WORKING DIRECTORY TO R SCRIPT LOCATION
# ============================================================

script_path <- rstudioapi::getSourceEditorContext()$path

if (script_path == "") {
  stop("Please save the R program before running it.")
}

project_folder <- dirname(script_path)

setwd(project_folder)

cat("Project folder:", project_folder, "\n")


# ============================================================
# 3. CREATE OUTPUT FOLDER
# ============================================================

output_folder <- file.path(project_folder, "outputs")

if (!dir.exists(output_folder)) {
  dir.create(output_folder)
}

cat("Output folder:", output_folder, "\n\n")


# ============================================================
# 4. LOAD DATASET
# ============================================================

weather <- read.csv(file.choose())


# ============================================================
# 5. VIEW DATASET INFORMATION
# ============================================================

cat("Dataset Structure:\n")
str(weather)

cat("\nMissing Values:\n")
print(colSums(is.na(weather)))


# ============================================================
# 6. DATA PREPROCESSING
# ============================================================

weather$date <- as.Date(weather$date)

weather_model <- weather %>%
  select(
    precipitation,
    temp_max,
    temp_min,
    wind,
    weather
  )

weather_model$weather <- as.factor(
  weather_model$weather
)


# ============================================================
# 7. WEATHER DISTRIBUTION
# ============================================================

weather_distribution <- ggplot(
  weather_model,
  aes(x = weather)
) +
  geom_bar() +
  labs(
    title = "Distribution of Weather Conditions",
    x = "Weather",
    y = "Number of Days"
  ) +
  theme_minimal()

print(weather_distribution)

ggsave(
  filename = file.path(
    output_folder,
    "01_weather_distribution.png"
  ),
  plot = weather_distribution,
  width = 8,
  height = 6,
  dpi = 300
)


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

set.seed(123)

train_index <- createDataPartition(
  weather_model$weather,
  p = 0.80,
  list = FALSE
)

train_data <- weather_model[
  train_index,
]

test_data <- weather_model[
  -train_index,
]


cat("\nTraining records:", nrow(train_data), "\n")
cat("Testing records:", nrow(test_data), "\n")


# ============================================================
# 9. DECISION TREE MODEL
# ============================================================

decision_tree <- rpart(
  weather ~ precipitation +
    temp_max +
    temp_min +
    wind,
  data = train_data,
  method = "class"
)


# ============================================================
# 10. SAVE DECISION TREE IMAGE
# ============================================================

png(
  filename = file.path(
    output_folder,
    "02_decision_tree.png"
  ),
  width = 1200,
  height = 900,
  res = 150
)

rpart.plot(
  decision_tree,
  main = "Decision Tree for Weather Prediction"
)

dev.off()


# ============================================================
# 11. DECISION TREE PREDICTION
# ============================================================

tree_prediction <- predict(
  decision_tree,
  test_data,
  type = "class"
)


# ============================================================
# 12. DECISION TREE CONFUSION MATRIX
# ============================================================

tree_cm <- confusionMatrix(
  tree_prediction,
  test_data$weather
)

cat("\n==============================\n")
cat("DECISION TREE RESULTS\n")
cat("==============================\n")

print(tree_cm)


# ============================================================
# 13. RANDOM FOREST MODEL
# ============================================================

set.seed(123)

random_forest <- randomForest(
  weather ~ precipitation +
    temp_max +
    temp_min +
    wind,
  data = train_data,
  ntree = 100,
  importance = TRUE
)


# ============================================================
# 14. RANDOM FOREST PREDICTION
# ============================================================

rf_prediction <- predict(
  random_forest,
  test_data
)


# ============================================================
# 15. RANDOM FOREST CONFUSION MATRIX
# ============================================================

rf_cm <- confusionMatrix(
  rf_prediction,
  test_data$weather
)

cat("\n==============================\n")
cat("RANDOM FOREST RESULTS\n")
cat("==============================\n")

print(rf_cm)


# ============================================================
# 16. EXTRACT PERFORMANCE
# ============================================================

tree_accuracy <- as.numeric(
  tree_cm$overall["Accuracy"]
)

rf_accuracy <- as.numeric(
  rf_cm$overall["Accuracy"]
)


tree_f1 <- mean(
  tree_cm$byClass[, "F1"],
  na.rm = TRUE
)

rf_f1 <- mean(
  rf_cm$byClass[, "F1"],
  na.rm = TRUE
)


performance <- data.frame(
  Algorithm = c(
    "Decision Tree",
    "Random Forest"
  ),
  Accuracy = c(
    tree_accuracy,
    rf_accuracy
  ),
  F1_Score = c(
    tree_f1,
    rf_f1
  )
)


cat("\n==============================\n")
cat("MODEL PERFORMANCE COMPARISON\n")
cat("==============================\n")

print(performance)


# ============================================================
# 17. SAVE PERFORMANCE RESULTS
# ============================================================

write.csv(
  performance,
  file = file.path(
    output_folder,
    "model_performance.csv"
  ),
  row.names = FALSE
)


# ============================================================
# 18. ACCURACY COMPARISON GRAPH
# ============================================================

accuracy_plot <- ggplot(
  performance,
  aes(
    x = Algorithm,
    y = Accuracy
  )
) +
  geom_col() +
  geom_text(
    aes(
      label = paste0(
        round(
          Accuracy * 100,
          2
        ),
        "%"
      )
    ),
    vjust = -0.5
  ) +
  labs(
    title = "Accuracy Comparison",
    x = "Machine Learning Algorithm",
    y = "Accuracy"
  ) +
  ylim(0, 1) +
  theme_minimal()


print(accuracy_plot)


ggsave(
  filename = file.path(
    output_folder,
    "03_accuracy_comparison.png"
  ),
  plot = accuracy_plot,
  width = 8,
  height = 6,
  dpi = 300
)


# ============================================================
# 19. F1 SCORE COMPARISON GRAPH
# ============================================================

f1_plot <- ggplot(
  performance,
  aes(
    x = Algorithm,
    y = F1_Score
  )
) +
  geom_col() +
  geom_text(
    aes(
      label = round(
        F1_Score,
        2
      )
    ),
    vjust = -0.5
  ) +
  labs(
    title = "F1 Score Comparison",
    x = "Machine Learning Algorithm",
    y = "F1 Score"
  ) +
  ylim(0, 1) +
  theme_minimal()


print(f1_plot)


ggsave(
  filename = file.path(
    output_folder,
    "04_f1_score_comparison.png"
  ),
  plot = f1_plot,
  width = 8,
  height = 6,
  dpi = 300
)


# ============================================================
# 20. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance_values <- importance(
  random_forest
)


importance_df <- data.frame(
  Feature = rownames(
    importance_values
  ),
  Importance = importance_values[
    ,
    "MeanDecreaseGini"
  ]
)


importance_plot <- ggplot(
  importance_df,
  aes(
    x = reorder(
      Feature,
      Importance
    ),
    y = Importance
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Random Forest Feature Importance",
    x = "Feature",
    y = "Importance"
  ) +
  theme_minimal()


print(importance_plot)


ggsave(
  filename = file.path(
    output_folder,
    "05_feature_importance.png"
  ),
  plot = importance_plot,
  width = 8,
  height = 6,
  dpi = 300
)


# ============================================================
# 21. FINAL RESULTS
# ============================================================

cat("\n\n========================================\n")
cat("FINAL MODEL PERFORMANCE\n")
cat("========================================\n")

cat(
  "Decision Tree Accuracy: ",
  round(
    tree_accuracy * 100,
    2
  ),
  "%\n"
)

cat(
  "Decision Tree F1 Score: ",
  round(
    tree_f1,
    2
  ),
  "\n"
)

cat(
  "Random Forest Accuracy: ",
  round(
    rf_accuracy * 100,
    2
  ),
  "%\n"
)

cat(
  "Random Forest F1 Score: ",
  round(
    rf_f1,
    2
  ),
  "\n"
)


# ============================================================
# 22. OUTPUT LOCATION
# ============================================================

cat("\n========================================\n")
cat("OUTPUTS SAVED SUCCESSFULLY\n")
cat("========================================\n")

cat(
  "All output files are saved in:\n",
  output_folder,
  "\n"
)

