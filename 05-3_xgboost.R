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

# lets scale variables, w centering

dat_count_X <- subset(dat_count_ns, select = -c(roma_felado))
dat_count_X_s = as.data.frame(scale(dat_count_X, scale = T, center = T))
dat_count = as.data.frame(cbind(dat_count_X_s, roma_felado = dat_count_ns$roma_felado))

dat_tf_X <- subset(dat_tf_ns, select = -c(roma_felado))
dat_tf_X_s = as.data.frame(scale(dat_tf_X, scale = T, center = T))
dat_tf = as.data.frame(cbind(dat_tf_X_s, roma_felado = dat_tf_ns$roma_felado))

dat_tf_idf_X <- subset(dat_tf_idf_ns, select = -c(roma_felado))
dat_tf_idf_X_s = as.data.frame(scale(dat_tf_idf_X, scale = T, center = T))
dat_tf_idf = as.data.frame(cbind(dat_tf_idf_X_s, roma_felado = dat_tf_idf_ns$roma_felado))


xgb_ab_train <- xgb.DMatrix(
  data    = select(dat_count, -c(roma_felado)) %>% as.matrix(),
  label   = dat_count$roma_felado
)




###########
# elemzes #
###########

# 1.1. extreme gradient boosting  w count data -------------------------------------------------------

# create parameter list
params <- list(
  eta = .3,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 1,
  colsample_bytree = 1,
  gamma = 0
)

set.seed(3)
# nfold a cv, nrounds pedig, h hanyszor ismeteljuk az egesz cv-t: 
start.time <- Sys.time()
model <- xgb.cv(
  data = xgb_ab_train,
  params = params,
  nrounds = 100,
  nfold = 3,
  objective = "binary:logistic",  
  verbose = 0,               # silent,
  early_stopping_rounds = 20, # stop if no improvement for 20 consecutive
  eval_metric = 'auc',
  prediction = T
)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken 


# CM

# model$pred # itt vannak a predikciok
pred_class <- ifelse(model$pred> 0.5, 1, 0) 
pred_f = factor(pred_class, levels= c("0", "1"))
obs = as.factor(dat_count$roma_felado)

confusionMatrix(data = pred_f, reference = obs)$table

# AUC
auc = round(max(model$evaluation_log$test_auc_mean),2)
cat("Az XGB count AUC erteke:", auc) 



# 1.2. tuning -------------------------------------------------------

# create hyperparameter grid
# explanation: https://xgboost.readthedocs.io/en/latest/parameter.html
hyper_grid <- expand.grid(
  eta = c(.3,.5), 
  max_depth = c(4, 6), 
  min_child_weight = c(1, 2), 
  subsample = c(.6,.8, 1), 
  colsample_bytree = c(.6,.8, 1),  
  gamma = c(0,5) 
)


# grid search 
start.time <- Sys.time()
for(i in 1:nrow(hyper_grid)) {
  
  # create parameter list
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i],
    gamma = hyper_grid$gamma[i]
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
    verbose = 0,              
    early_stopping_rounds = 20, # stop if no improvement for 10 consecutive
    eval_metric = 'auc',
    prediction = T
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.max(xgb.tune$evaluation_log$test_auc_mean)
  hyper_grid$max_auc[i] <- max(xgb.tune$evaluation_log$test_auc_mean)
}
  
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken # 6 perc

top_params = hyper_grid %>%
  dplyr::arrange(max_auc) %>%
  tail(1)

top_params
top_params_l = lapply(top_params[1:6], as.data.frame)

setwd("C:/Users/Gabor/Documents/01_ELTE/00_szakdoga/03_Adatok/05_Adatfeldolgozas/07_NLP-proba")

saveRDS(top_params, file = "xgb_params_ethnicity_count_10-30.rds") # Save an object to a file (6 p)
#top_params = readRDS(file = "xgb_params_ethnicity_count_10-30.rds") # Restore the object



# 1.3. train optimal model -------------------------------------------------------

set.seed(3)
model_final_tuned <- xgb.cv(
  data = xgb_ab_train,
  params = top_params_l,
  nrounds = 100,
  nfold = 3,
  objective = "binary:logistic",  
  verbose = 0,              
  early_stopping_rounds = 20, # stop if no improvement for 10 consecutive
  eval_metric = 'auc',
  prediction = T
)

pred_class <- ifelse(model_final_tuned$pred> 0.5, 1, 0) 
pred_f = factor(pred_class, levels= c("0", "1"))
obs = as.factor(dat_count$roma_felado)

confusionMatrix(data = pred_f, reference = obs)$table

# AUC
auc = round(max(model_final_tuned$evaluation_log$test_auc_mean),2)
cat("Az XGB count AUC erteke:", auc) 




# 1.4. extra -------------------------------------------------------

# eleg lett volna 8 iteracio

# get number of trees that maximize auc
model_final_tuned$evaluation_log %>%
  dplyr::summarise(
    ntrees.train = which(train_auc_mean == max(train_auc_mean))[1],
    auc.train   = max(train_auc_mean),
    ntrees.test  = which(test_auc_mean == max(test_auc_mean))[1],
    auc.test   = max(test_auc_mean)
  )


# plot error vs number trees
ggplot(model_final_tuned$evaluation_log) +
  geom_line(aes(iter, train_auc_mean), color = "red") +
  geom_line(aes(iter, test_auc_mean), color = "blue")



# feature importance

# train final model
xgb.fit.final <- xgboost(
  params = top_params_l,
  data = xgb_ab_train,
  nrounds = 100,
  objective = "binary:logistic",  
  verbose = 0,              
  early_stopping_rounds = 20, # stop if no improvement for 10 consecutive
  eval_metric = 'auc'
)


# create importance matrix
importance_matrix <- xgb.importance(model = xgb.fit.final)

# variable importance plot
xgb.plot.importance(importance_matrix, top_n = 10, measure = "Gain",
                    rel_to_first = T, xlab = "Prediktorok relatív fontossága az etnicitás modellben")







