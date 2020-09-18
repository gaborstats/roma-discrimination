####################
# szovegelemzes    #
####################

library(quanteda)
library(tidytext)
library(tidyverse)
library(tm)
library(glmnet)
library(caret)
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


require(dplyr)
dat_count = left_join(x = df_count, 
                 y = feat_agg, 
                 by = c("doc_id" = "telepules"), keep = F)
 
dat_tf = left_join(x = df_tf, 
                      y = feat_agg, 
                      by = c("doc_id" = "telepules"), keep = F)

dat_tf_idf = left_join(x = df_tf_idf, 
                      y = feat_agg, 
                      by = c("doc_id" = "telepules"), keep = F)

dat_count <- subset(dat_count, select = -c(doc_id))
dat_tf <- subset(dat_tf, select = -c(doc_id))
dat_tf_idf <- subset(dat_tf_idf, select = -c(doc_id))


# nearzero var kezelese
#### ez meg lefut pca-val: 
j = nearZeroVar(dat_count, saveMetrics = F, freqCut = 25, uniqueCut = 2, names = F) # 220 db prediktor

# explanation: https://rstatisticsblog.com/data-science-in-action/data-preprocessing/how-to-identify-variables-with-zero-variance/

dat_count_nzv = dat_count[,-j] # 0.49 az AUC akkor is, 
dat_tf_nzv = dat_tf[,-j]
dat_tf_idf_nzv = dat_tf_idf[,-j]

# split train test
set.seed(3)
train_index = sample(1:nrow(dat_count_nzv), 4*nrow(dat_count_nzv)/5)

train_count = dat_count_nzv[train_index,]
train_tf = dat_count_nzv[train_index,]
train_tf_idf = dat_count_nzv[train_index,]

test_count = dat_count_nzv[-train_index,]
test_tf = dat_count_nzv[-train_index,]
test_tf_idf = dat_count_nzv[-train_index,]



xgb_ab_train <- xgb.DMatrix(
  data    = select(train_count, -c(roma_felado)) %>% as.matrix(),
  label   = train_count$roma_felado
)


xgb_ab_train_test <- xgb.DMatrix(
  data    = select(test_count, -c(roma_felado)) %>% as.matrix(),
  label   = test_count$roma_felado
)

# 0.2. EDA -----------------------------------------------------------

# find most common words
p = td %>%
  count(term, sort = T)
print(p) 

# wordcloud (requires a dfm)
textplot_wordcloud(dfm, max_words = 50, rotation = 0.25, 
                   color = rev(RColorBrewer::brewer.pal(10, "RdBu")))



###########
# elemzes #
###########

# forrasok:
# alapos: http://uc-r.github.io/gbm_regression
# tuninghoz hasznos: https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/#:~:text=nrounds%5Bdefault%3D100%5D&text=For%20classification%2C%20it%20is%20similar,number%20of%20trees%20to%20grow.
# https://xgboost.ai/rstats/2016/03/10/xgboost.html
# http://www.sthda.com/english/articles/35-statistical-machine-learning-essentials/139-gradient-boosting-essentials-in-r-using-xgboost/

# 1.1. extreme gradient boosting  w count data -------------------------------------------------------

# create parameter list
params <- list(
  eta = .3,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 1,
  colsample_bytree = 1
)

set.seed(3)
start.time <- Sys.time()
model <- xgb.cv(
  data = xgb_ab_train,
  params = params,
  nrounds = 1000,
  nfold = 3,
  objective = "binary:logistic",  
  verbose = 0,               # silent,
  early_stopping_rounds = 20 # stop if no improvement for 10 consecutive
)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken 

model

# get number of trees that minimize error
model$evaluation_log %>%
  dplyr::summarise(
    ntrees.train = which(train_error_mean == min(train_error_mean))[1],
    error.train   = min(train_error_mean),
    ntrees.test  = which(test_error_mean == min(test_error_mean))[1],
    error.test   = min(test_error_mean),
  )


# plot error vs number trees
ggplot(model$evaluation_log) +
  geom_line(aes(iter, train_error_mean), color = "red") +
  geom_line(aes(iter, test_error_mean), color = "blue")


# train optimal model
model_final <- xgboost(
  params = params,
  data = xgb_ab_train,
  nrounds = 1000,
  objective = "binary:logistic",
  verbose = 0
)


# create importance matrix
importance_matrix <- xgb.importance(model = model_final)

# variable importance plot
xgb.plot.importance(importance_matrix, top_n = 10, measure = "Gain")


# cm
predictions_enet_count <- model_final %>% predict(xgb_ab_train_test, type = "prob") %>% as.vector()

pred_class <- ifelse(predictions_enet_count> 0.5, 1, 0) 
# itt vmiert csak egy pred oszlop van, erdekes.
pred_f = factor(pred_class, levels= c("0", "1"))
obs = as.factor(test_count$roma_felado)

confusionMatrix(data = pred_f, reference = obs)$table


# AUC
rocplot=function(pred_class, obs, ...){
  predob = prediction(pred_class, obs)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

rocplot(predictions_enet_count, obs, main="ROC plot")


pred_ROCR <- prediction(pred_class, obs)
auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- round(auc_ROCR@y.values[[1]],2)
cat("Az XGB count AUC erteke:", auc_ROCR) 
# default beallitassal:
# 0.45 AUC 220 prediktorral
# 0.49 AUC 413 prediktorral 


# tuning 1

# create hyperparameter grid
hyper_grid <- expand.grid(
  eta = c(.1, .3, .5),
  max_depth = c(5, 6, 7),
  min_child_weight = c(1, 3),
  subsample = c(.8, 1), 
  colsample_bytree = c(.9, 1),
  gamma = c(0,5)
)


# grid search 
# most csak 100 iteracioval
start.time <- Sys.time()
for(i in 1:nrow(hyper_grid)) {
  
  # create parameter list
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  # reproducibility
  set.seed(3)
  
  # train model
  xgb.tune <- xgb.cv(
    data = xgb_ab_train,
    params = params,
    nfold = 3,
    nrounds = 100,
    objective = "binary:logistic",  
    verbose = 0,               # silent,
    early_stopping_rounds = 20 # stop if no improvement for 10 consecutive
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_error_mean)
  hyper_grid$min_error[i] <- min(xgb.tune$evaluation_log$test_error_mean)
}
  
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken # 6.2 perc

top_params = hyper_grid %>%
  dplyr::arrange(min_error) %>%
  head(1)

top_params
top_params_l = lapply(top_params[1:6], as.data.frame)


saveRDS(top_params, file = "xgb_params.rds") # Save an object to a file
readRDS(file = "xgb_params.rds") # Restore the object


# train optimal model
model_final_tuned <- xgboost(
  params = top_params_l,
  data = xgb_ab_train,
  nrounds = 100,
  objective = "binary:logistic",
  verbose = 0
)


# cm
predictions_enet_count <- model_final_tuned %>% predict(xgb_ab_train_test, type = "prob") %>% as.vector()

pred_class <- ifelse(predictions_enet_count> 0.5, 1, 0) 
# itt vmiert csak egy pred oszlop van, erdekes.
pred_f = factor(pred_class, levels= c("0", "1"))
obs = as.factor(test_count$roma_felado)

confusionMatrix(data = pred_f, reference = obs)$table


# AUC
rocplot=function(pred_class, obs, ...){
  predob = prediction(pred_class, obs)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

rocplot(predictions_enet_count, obs, main="ROC plot")


pred_ROCR <- prediction(pred_class, obs)
auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- round(auc_ROCR@y.values[[1]],2)
cat("Az XGB count AUC erteke:", auc_ROCR) 
# 0.78 AUC 220 prediktorral
# 0.48 AUC 413 prediktorral (ez fura)


### tuning 2 ###

# create hyperparameter grid
hyper_grid <- expand.grid(
  eta = c(.09, .1),
  max_depth = c(4),
  min_child_weight = c(3),
  subsample = c(.7, .8), 
  colsample_bytree = c(.9),
  gamma = c(0)
)


# grid search 
# most csak 100 iteracioval
start.time <- Sys.time()
for(i in 1:nrow(hyper_grid)) {
  
  # create parameter list
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  # reproducibility
  set.seed(3)
  
  # train model
  xgb.tune <- xgb.cv(
    data = xgb_ab_train,
    params = params,
    nfold = 3,
    nrounds = 1000,
    objective = "binary:logistic",  
    verbose = 0,               # silent,
    early_stopping_rounds = 20 # stop if no improvement for 10 consecutive.
    # Q: az early stopping at optimal trees-re vonatk? 
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_error_mean)
  hyper_grid$min_error[i] <- min(xgb.tune$evaluation_log$test_error_mean)
}

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken # 1.2 perc

top_params = hyper_grid %>%
  dplyr::arrange(min_error) %>%
  head(1)

top_params_l = lapply(top_params[1:6], as.data.frame)


#saveRDS(top_params, file = "xgb_params.rds") # Save an object to a file
#readRDS(file = "xgb_params.rds") # Restore the object


# train optimal model
model_final_tuned <- xgboost(
  params = top_params_l,
  data = xgb_ab_train,
  nrounds = 1000,
  objective = "binary:logistic",
  verbose = 0
)


# cm
predictions_enet_count <- model_final_tuned %>% predict(xgb_ab_train_test, type = "prob") %>% as.vector()

pred_class <- ifelse(predictions_enet_count> 0.5, 1, 0) 
# itt vmiert csak egy pred oszlop van, erdekes.
pred_f = factor(pred_class, levels= c("0", "1"))
obs = as.factor(test_count$roma_felado)

confusionMatrix(data = pred_f, reference = obs)$table


# AUC
rocplot=function(pred_class, obs, ...){
  predob = prediction(pred_class, obs)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

rocplot(predictions_enet_count, obs, main="ROC plot")


pred_ROCR <- prediction(pred_class, obs)
auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- round(auc_ROCR@y.values[[1]],2)
cat("Az XGB count AUC erteke:", auc_ROCR) 
# 0.83 AUC 220 prediktorral
# 0.41 AUC 413 prediktorral (ez fura)

