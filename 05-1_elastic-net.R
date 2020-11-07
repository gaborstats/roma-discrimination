####################
# szovegelemzes    #
####################

library(quanteda)
library(tidytext)
library(dplyr)
library(tm)
library(glmnet)
library(caret)
library(randomForest)
library(ROCR)
library(quanteda.textmodels)
library(xgboost)
library (e1071)

# getwd()
setwd("C:/Users/Gabor/Documents/01_ELTE/00_szakdoga/03_Adatok/05_Adatfeldolgozas/05_NLP-proba")

df_teljes = read.csv("2020-09-02_tokens_short.csv", sep = ";", stringsAsFactors = F,
                     header= T, dec = ",")

df <- subset(df_teljes, select = -c(sentence, id))
rm(df_teljes)

# 0.1. adatstrukturak letrehozasa ----------------------------------------

# tf, tf_idf sulyozas, majd adjuk hozza a feature-öket
dfm <- quanteda::dfm(x = df$word, verbose = F, group = df$telepules)

require(tidytext)
td <- tidy(dfm)

tf_idf <- td %>%
  bind_tf_idf(term, document, count) %>%
  arrange(desc(tf)) # NB: ebben a tbl-ben benne van a count is

dfm_count = cast_dfm(tf_idf, document, term, count)
dfm_tf = cast_dfm(tf_idf, document, term, tf)
dfm_tf_idf = cast_dfm(tf_idf, document, term, tf_idf)

df_count <- convert(dfm_count, to = "data.frame")
df_tf <- convert(dfm_tf, to = "data.frame")
df_tf_idf <- convert(dfm_tf_idf, to = "data.frame")

feat <- subset(df, select = -c(word))
feat_agg <- feat[!duplicated(feat[ ,"telepules"]),]
feat_agg$roma_felado = as.factor(feat_agg$roma_felado)

# caret ROC-hoz kell
levels(feat_agg$roma_felado) <- c("nem_roma", "roma")


# feature-ok es szovektorok osszekapcsolasa

require(dplyr)
dat_count_j = left_join(x = df_count, 
                 y = feat_agg, 
                 by = c("doc_id" = "telepules"), keep = F)
 
dat_tf_j = left_join(x = df_tf, 
                      y = feat_agg, 
                      by = c("doc_id" = "telepules"), keep = F)

dat_tf_idf_j = left_join(x = df_tf_idf, 
                      y = feat_agg, 
                      by = c("doc_id" = "telepules"), keep = F)

dat_count_ns <- subset(dat_count_j, select = -c(doc_id))
dat_tf_ns <- subset(dat_tf_j, select = -c(doc_id))
dat_tf_idf_ns <- subset(dat_tf_idf_j, select = -c(doc_id))


# lets scale variables
dat_count_X <- subset(dat_count_ns, select = -c(roma_felado))
dat_count_X_s = as.data.frame(scale(dat_count_X, scale = T, center = T))
dat_count = as.data.frame(cbind(dat_count_X_s, roma_felado = dat_count_ns$roma_felado))

dat_tf_X <- subset(dat_tf_ns, select = -c(roma_felado))
dat_tf_X_s = as.data.frame(scale(dat_tf_X, scale = T, center = T))
dat_tf = as.data.frame(cbind(dat_tf_X_s, roma_felado = dat_tf_ns$roma_felado))

dat_tf_idf_X <- subset(dat_tf_idf_ns, select = -c(roma_felado))
dat_tf_idf_X_s = as.data.frame(scale(dat_tf_idf_X, scale = T, center = T))
dat_tf_idf = as.data.frame(cbind(dat_tf_idf_X_s, roma_felado = dat_tf_idf_ns$roma_felado))


###########
# elemzes #
###########

# methods in train: http://topepo.github.io/caret/train-models-by-tag.html

# 1.1. count data -------------------------------------------------------

get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}

require(caret)
require(glmnet)
library(Matrix)


train_control <- trainControl(
  method="cv", 
  number = 3, 
  savePredictions = T,
  classProbs = TRUE,
  summaryFunction = twoClassSummary)

alfa = seq(0, 1, 0.25) # L1/L2 mixelo parameter 
lamb = seq(0, 1, 0.25) # regularizacios buntetotag

# set up tuning grid
search_grid <- expand.grid(
  alpha = alfa, 
  lambda = lamb
)


set.seed(3)
model1 <- train(
  roma_felado ~ ., 
  data = dat_count, 
  trControl = train_control, 
  tuneGrid = search_grid,
  method = "glmnet",
  metric = "ROC")

print(model1)
get_best_result(model1)

caret::confusionMatrix(model1)
tab <- round(caret::confusionMatrix(model1)$table,0)
"A legjobb modellhez tartozo tevesztesi matrix (százalékban):" # sum(tab)
tab 

# AUC

roc = round(get_best_result(model1)[[3]],2)
cat("A legjobb modellhez tartozo AUC ertek:", roc)


# feature importance

enetImp = varImp(model1, lambda = 0,scale = F) # ez vonatozna a roma feladora? 
plot(enetImp, top = 10)

# minden parameter t statisztikajanak abszoluterteke alapjan vannak rangsorolva. 
# Ergo, sajnos csak a valtozo hatasanak nagysagat mondja meg, az iranyt nem. 

# 1.2. tf data -------------------------------------------------------


train_control <- trainControl(
  method="cv", 
  number = 3, 
  savePredictions = T,
  classProbs = TRUE,
  summaryFunction = twoClassSummary)

# set up tuning grid
search_grid <- expand.grid(
  alpha = alfa, 
  lambda = lamb
)


set.seed(3)
model2 <- train(
  roma_felado ~ ., 
  data = dat_tf, 
  trControl = train_control, 
  tuneGrid = search_grid,
  method = "glmnet",
  metric = "ROC")

print(model2)
get_best_result(model2)

caret::confusionMatrix(model2)
tab <- round(caret::confusionMatrix(model1)$table,0)
"A legjobb modellhez tartozo tevesztesi matrix:"
tab # szazalekban van sum(tab)

roc = round(get_best_result(model2)[[3]],2)
cat("A legjobb modellhez tartozo AUC ertek:", roc)

# 1.2. tf-idf data -------------------------------------------------------


train_control <- trainControl(
  method="cv", 
  number = 3, 
  savePredictions = T,
  classProbs = TRUE,
  summaryFunction = twoClassSummary)

# set up tuning grid
search_grid <- expand.grid(
  alpha = alfa, 
  lambda = lamb 
)


set.seed(3)
model3 <- train(
  roma_felado ~ ., 
  data = dat_tf_idf, 
  trControl = train_control, 
  tuneGrid = search_grid,
  method = "glmnet",
  metric = "ROC")

print(model3)
get_best_result(model3)

caret::confusionMatrix(model3)
tab <- round(caret::confusionMatrix(model1)$table,0)
"A legjobb modellhez tartozo tevesztesi matrix:"
tab # szazalekban van sum(tab)

roc = round(get_best_result(model3)[[3]],2)
cat("A legjobb modellhez tartozo AUC ertek:", roc)

