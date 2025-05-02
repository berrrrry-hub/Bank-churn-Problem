# Loading training data
#setwd("/Users/abaabatian/Downloads/ds dataset")

# Load the train.csv file into R
train_data <- read.csv("train.csv", header = TRUE, stringsAsFactors = TRUE)

# Display the first few rows of the dataset to verify it's loaded correctly
head(train_data)

str(train_data)
# Make the below into factors from int as only 2 levels
# Convert variables into factors
train_data$HasCrCard <- as.factor(train_data$HasCrCard)
train_data$IsActiveMember <- as.factor(train_data$IsActiveMember)
train_data$Exited <- as.factor(train_data$Exited)

# Check the structure to confirm changes
str(train_data)

# Load the necessary packages.
install.packages("class")
install.packages("caret")
install.packages("randomForest")
install.packages("glmnet")
library(glmnet)
library(class)
library(caret)
library(randomForest)
library(dplyr)

##############################Data Visualization#################################
# Remove Surname, id and CustomerId  because they do not matter
# Create a copy of train_data without the 'id', 'CustomerId', and 'Surname' columns
train_data2 <- train_data[, !(names(train_data) %in% c("id", "CustomerId", "Surname"))]

# Check the structure of the modified dataset
str(train_data2)

# Select only numeric variables for correlation matrix
numeric_vars <- train_data2[, sapply(train_data2, is.numeric)]

# Correlation matrix
correlation_matrix <- cor(numeric_vars, use = "complete.obs")

# Visualize the correlation matrix
library(corrplot)
corrplot(correlation_matrix, method = "circle", type = "upper", tl.cex = 0.8)

# Bar plot for 'Exited' variable
library(ggplot2)
ggplot(train_data2, aes(x = factor(Exited))) +
  geom_bar(fill = "steelblue") +
  labs(x = "Exited (Churn)", y = "Count", title = "Distribution of 'Exited' Variable")

# Boxplot for Age vs Exited
ggplot(train_data2, aes(x = factor(Exited), y = Age)) +
  geom_boxplot(fill = "orange") +
  labs(x = "Exited", y = "Age", title = "Boxplot of Age by Exited")

# Histogram for Balance by Exited
ggplot(train_data2, aes(x = Balance, fill = factor(Exited))) +
  geom_histogram(position = "dodge", bins = 30) +
  labs(x = "Balance", y = "Count", title = "Histogram of Balance by Exited")

# Bar plot for Gender
ggplot(train_data2, aes(x = Gender, fill = factor(Exited))) +
  geom_bar(position = "dodge") +
  labs(x = "Gender", y = "Count", fill = "Exited", title = "Gender Distribution by Exited")

# Bar plot for Geography
ggplot(train_data2, aes(x = Geography, fill = factor(Exited))) +
  geom_bar(position = "dodge") +
  labs(x = "Geography", y = "Count", fill = "Exited", title = "Geography Distribution by Exited")

# Load the GGally package for pair plots
library(GGally)

# Select a few numeric variables to plot
ggpairs(train_data2[, c("CreditScore", "Age", "Balance", "EstimatedSalary", "Exited")], aes(color = factor(Exited)))


# Load necessary library
library(GGally)
library(ggplot2)

# Custom pair plot function to only plot against 'Exited'
ggpairs(train_data2[, c("CreditScore", "Age", "Balance", "EstimatedSalary", "Exited")],
        columnLabels = c("CreditScore", "Age", "Balance", "EstimatedSalary", "Exited"),
        mapping = ggplot2::aes(color = factor(Exited)),
        upper = list(continuous = "blank"),   # Remove upper triangle
        lower = list(combo = "blank"))        # Remove lower triangle

############################ Logistic Regression ######################

# Create a logistic regression model with 'Exited' as the dependent variable
logistic_model <- glm(Exited ~ CreditScore + Geography + Gender + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary,
                      data = train_data2, family = binomial)

# View the summary of the logistic model
summary(logistic_model)

############################ Random Forest ######################

# Set seed for reproducibility
set.seed(123) 
# Build the Random Forest model using 'Exited' as the target variable
rf_model <- randomForest(Exited ~ . , data = train_data2, ntree = 100)

# Plot the OOB error rate vs number of trees
plot(rf_model, main = "OOB Error Rate vs. Number of Trees")
#Using the plot, we identify that n = 100 is ideal.

# Print model summary
print(rf_model)

# Define training control with k-fold cross-validation
train_control <- trainControl(method = "cv", number = 5)

# Use caret's train function to perform k-fold cross-validation

# For reproducibility
set.seed(123)  
rf_cv_model <- train(Exited ~ CreditScore + Geography + Gender + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary,
                     data = train_data2, 
                     method = "rf",  # Random Forest method
                     trControl = train_control,
                     ntree = 100)

# Print the cross-validation results
print(rf_cv_model)

# Set seed for reproducibility
set.seed(123)

# Re-run the Random Forest with mtry = 2
final_rf_model <- randomForest(Exited ~ ., data = train_data2, ntree = 100, mtry = 2)

# Print the final model summary
print(final_rf_model)

# Set seed for reproducibility
set.seed(123)

# Re-run Random Forest with class weight adjustment----final random forest moel
rf_model_weighted <- randomForest(Exited ~ ., 
                                  data = train_data2, 
                                  ntree = 100, 
                                  mtry = 2, 
                                  classwt = c(0.45, 0.55))  # adjust class weight 

# Print model summary
print(rf_model_weighted)

# Extract variable importance using the correct function
var_importance <- importance(rf_model_weighted)

# Print variable importance
print(var_importance)

# Plot variable importance using the built-in randomForest function
varImpPlot(rf_model_weighted, main = "Feature Importance Plot")


############################ K-Nearest Neighbor ######################

# Prepare the data
# Convert categorical variables to dummy variables
train_data2_numeric <- model.matrix(Exited ~ . - 1, data = train_data2)  # The -1 removes the intercept column

# Normalize the data (scaling between 0 and 1)
train_data2_scaled <- as.data.frame(scale(train_data2_numeric))

# Add back the 'Exited' target variable
train_data2_scaled$Exited <- train_data2$Exited

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(train_data2_scaled$Exited, p = 0.8, list = FALSE)
train_set <- train_data2_scaled[train_index, ]
test_set <- train_data2_scaled[-train_index, ]

# Separate features and target variable for training and testing sets
train_features <- train_set[, -ncol(train_set)]
train_target <- train_set$Exited
test_features <- test_set[, -ncol(test_set)]
test_target <- test_set$Exited

# Perform KNN
k_value <- 5  # You can adjust the value of k
knn_predictions <- knn(train = train_features, 
                       test = test_features, 
                       cl = train_target, 
                       k = k_value)

# Evaluate the model using a confusion matrix
confusion_matrix <- confusionMatrix(knn_predictions, test_target)

# Print confusion matrix and knn
print(confusion_matrix)
print(knn_predictions)

########################### Lasso ##############################
# glmnet for Lasso
library(glmnet)

# Prepare data for glmnet (Lasso regression)
x <- model.matrix(Exited ~ CreditScore + Geography + Gender + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary, 
                  data = train_data2)[, -1]  # Remove intercept
y <- as.numeric(as.character(train_data2$Exited))

# Perform cross-validated Lasso regression
set.seed(123)
cv_lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")

# Get the best lambda value (penalty)
best_lambda <- cv_lasso$lambda.min
print(paste("Best Lambda:", best_lambda))

# Train the final Lasso model with the best lambda
lasso_model <- glmnet(x, y, alpha = 1, lambda = best_lambda, family = "binomial")

summary(lasso_model)

# Extract and print the coefficients for the final Lasso model
lasso_coefficients <- coef(lasso_model)

# Convert the sparse matrix to a readable format and print
lasso_coefficients <- as.matrix(lasso_coefficients)
print(lasso_coefficients)


# Define the formula without manually specifying dummy variables
formula <- Exited ~ CreditScore + Geography + Gender + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary

# Fit a logistic regression model using the original Geography variable
logistic_model_lasso <- glm(formula, data = train_data2, family = "binomial")

# Display the summary of the model
summary(logistic_model_lasso)


##################### ALL PREDICTIONS ###############

# Predict churn probability for each customer in the training dataset
train_data2$churn_prob <- predict(logistic_model, newdata = train_data2, type = "response")

# Predict churn probability with random forest
train_data2$rf_churn_prob <- predict(rf_cv_model, newdata = train_data2, type = "prob")[, 2]

# Predict churn probability with Lasso
train_data2$lasso_churn_prob <- predict(logistic_model_lasso, newx = x, type = "response")

############################ AUC #######################
library(pROC)

# Logistic regression AUC
logistic_roc <- roc(train_data2$Exited, train_data2$churn_prob)
print(paste("Logistic AUC:", auc(logistic_roc)))

# Random forest AUC
rf_roc <- roc(train_data2$Exited, train_data2$rf_churn_prob)
print(paste("Random Forest AUC:", auc(rf_roc)))

# Lasso AUC
lasso_roc <- roc(train_data2$Exited, train_data2$lasso_churn_prob)
print(paste("Lasso AUC:", auc(lasso_roc)))

########plots 
# Logistic regression ROC curve
plot(logistic_roc, main = "ROC Curve for Logistic Regression")

# Random forest ROC curve
plot(rf_roc, main = "ROC Curve for Random Forest", col = "blue", add = TRUE)

# Lasso ROC curve
plot(lasso_roc, main = "ROC Curve for Lasso", col = "red", add = TRUE)

# Add a legend to the plot
legend("bottomright", legend = c("Logistic Regression", "Random Forest", "Lasso"),
       col = c("black", "blue", "red"), lwd = 2)

# We chose Random Forest because it has the highest AUC


############################## Test Dataset #######################

# Load the test dataset
test_data <- read.csv("test.csv")

# Ensure the test_data is preprocessed the same way as train_data2
# Remove unnecessary columns like 'id', 'CustomerId', 'Surname' from test_data
test_data2 <- test_data[, !(names(test_data) %in% c("id", "CustomerId", "Surname"))]

str(test_data2)
#Make the below into factors from int as only 2 levels
# Convert variables into factors
test_data2$HasCrCard <- as.factor(test_data2$HasCrCard)
test_data2$IsActiveMember <- as.factor(test_data2$IsActiveMember)
test_data2$Geography <- as.factor(test_data2$Geography)
test_data2$Gender <- as.factor(test_data2$Gender)

#train_data$Exited <- as.factor(train_data$Exited)

# Check the structure to confirm changes
str(test_data2)
str(train_data2)

# Predict the Exited variable using the random forest model. We use type=prob to get decimal values 
rf_probabilities <- predict(rf_model, newdata = test_data2, type = "prob")

# View the first few probabilities for class 1 (Exited = 1)
head(rf_probabilities[, 2])  # This gives the probability of the customer exiting (churn)

# View the first few predictions
head(rf_probabilities)

# Assign the predicted probabilities for Exited = 1 to a new column in test_data
test_data2$Exited_Prediction <- rf_probabilities[, 2]

# View the test data with predictions
head(test_data2)

# Predicted churn probabilities are already in test_data2$Exited
# Define thresholds to evaluate
thresholds <- seq(0, 1, by = 0.01)

# Define cost-benefit parameters
clv <- 500   # Customer Lifetime Value
cost_offer <- 100  # Cost of promotional offer

# Initialize a vector to store the profit for each threshold
profits <- numeric(length(thresholds))

# Loop over thresholds to calculate expected profit
for (i in seq_along(thresholds)) {
  # Classify customers as churners based on the threshold
  predicted_class <- ifelse(test_data2$Exited >= thresholds[i], 1, 0)
  
  # Calculate expected profit for customers classified as churners
  # Expected profit = p * (CLV - Cost of Offer) + (1 - p) * (-CLV)
  expected_profit <- sum(predicted_class * ((test_data2$Exited * (clv - cost_offer)) - ((1 - test_data2$Exited) * clv)))
  
  # Store the expected profit for this threshold
  profits[i] <- expected_profit
}

# Find the optimal threshold that maximizes expected profit
optimal_threshold <- thresholds[which.max(profits)]
print(paste("Optimal Threshold:", optimal_threshold))
print(paste("Maximum Profit:", max(profits)))


# Assuming thresholds and profits are already calculated
# thresholds = seq(0, 1, by = 0.01)
# profits = calculated profit values

# Filter out the thresholds where profits are greater than 0
positive_profit_indices <- which(profits > 0)
positive_thresholds <- thresholds[positive_profit_indices]
positive_profits <- profits[positive_profit_indices]

# Create the profit curve, showing only the part where profit > 0
plot(positive_thresholds, positive_profits, type = "l", col = "blue", lwd = 2,
     xlab = "Threshold", ylab = "Profit", main = "Profit vs Threshold (Profit > 0)")
# Assuming test_data2 contains the churn probabilities and other features
# Replace 'churn_probability' with the actual column name in your dataset

################ ROI of offer ##################

eligible_customers <- sum(ifelse(test_data2$Exited >= optimal_threshold, 1, 0))

# Calculate total cost (eligible customers * cost per offer)
total_cost <- eligible_customers * cost_offer

# Calculate ROI using the formula (Profit / Cost) * 100
roi <- (max(profits) / total_cost) * 100

# Display the results
cat("Number of Eligible Customers:", eligible_customers, "\n")
cat("Total Cost: $", total_cost, "\n")
cat("Profit: $", max(profits), "\n")
cat("ROI: ", roi, "%", "\n")



