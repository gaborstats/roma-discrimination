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
#df = select(df_teljes, roma_felado, magas_statusz, word, bulletP, sent_len, telepules, id)


# 0.1. adatstrukturak letrehozasa ----------------------------------------

# tf, tf_idf sulyozas, majd adjuk hozza a feature-öket
dfm <- quanteda::dfm(x = df$word, verbose = F, group = df$telepules)
require(tidytext)
td <- tidy(dfm)
#arrange(td, desc(count))
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
#class(feat_agg$roma_felado)


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

# haromszor ismeteltuk, mehetne fv-be
# names(dat_count)[3800:3817]

# nearzero var kezelese
#### ez meg lefut pca-val: 
j = nearZeroVar(dat_count, saveMetrics = F, freqCut = 25, uniqueCut = 3, names = F) # 273 db, nem fut le

# explanation: https://rstatisticsblog.com/data-science-in-action/data-preprocessing/how-to-identify-variables-with-zero-variance/
# a naive bayeshez meg kell nezni a valtozok felteteles eloszalasat (as suggested by neaZero help)

#dat_count_nzv = dat_count
dat_count_nzv = dat_count[,-j] # megnyolcadoltuk a valtozok szamat
dat_tf_nzv = dat_tf[,-j]
dat_tf_idf_nzv = dat_tf_idf[,-j]

a = ncol(dat_tf_nzv)-18
b = dat_tf_nzv[,1:a]
legkisebb_tf = min( b[b!=min(b)] )

a = ncol(dat_tf_idf_nzv)-18
b2 = dat_tf_idf_nzv[,1:a]
legkisebb_tf_idf = min( b2[b2!=min(b2)] )

# stringek laplace korrekciohoz:
e =  0:5
legkisebb_tf_s = legkisebb_tf * e
legkisebb_tf_idf_s = legkisebb_tf_idf * e


# split train test
set.seed(3)
# rule of thumb to divide a data set: https://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio#:~:text=Split%20your%20data%20into%20training,20%20is%20a%20fair%20split).
train_index = sample(1:nrow(dat_count_nzv), 4*nrow(dat_count_nzv)/5)
train_count = dat_count_nzv[train_index,]
train_tf = dat_count_nzv[train_index,]
train_tf_idf = dat_count_nzv[train_index,]

test_count = dat_count_nzv[-train_index,]
test_tf = dat_count_nzv[-train_index,]
test_tf_idf = dat_count_nzv[-train_index,]

# 0.2. EDA -----------------------------------------------------------

# find most common words
p = td %>%
  count(term, sort = T)
p 

# wordcloud (requires a dfm)
textplot_wordcloud(dfm, max_words = 50, rotation = 0.25, 
                   color = rev(RColorBrewer::brewer.pal(10, "RdBu")))



###########
# elemzes #
###########

# nezzuk meg how to do cv on lasso es redge es azokat futassuk kulon-kulon

# naive bayes: https://uc-r.github.io/naive_bayes
# nb 2 : https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/#:~:text=Naive%20Bayes%20Model-,What%20is%20Naive%20Bayes%20algorithm%3F,presence%20of%20any%20other%20feature.
# methods in train: http://topepo.github.io/caret/train-models-by-tag.html

# 1.1. naive bayes w count data -------------------------------------------------------

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
  savePredictions = TRUE)

# set up tuning grid
search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = 0:5, # laplace correction
  adjust = seq(0, 5, by = 1) # allows us to adjust the bandwidth of the 
  # kernel density (larger numbers mean more flexible density estimate)
)


set.seed(3)
model <- train(
  roma_felado ~ ., 
  data = train_count, 
  trControl = train_control, 
  tuneGrid = search_grid,
  preProc = c("BoxCox", "center", "scale", "pca"), # scale: divides by the SD
  method = "nb")

# van vmi modszer a zero variance predictorokra: https://topepo.github.io/caret/pre-processing.html


get_best_result(model)

# cm
predictions_nb_count <- model %>% predict(test_count, type = "prob") %>% as.vector()

pred_class <- ifelse(predictions_nb_count[,2] > 0.5, 1, 0)
pred_f = factor(pred_class, levels= c("0", "1"))
obs = as.factor(test_count$roma_felado)

confusionMatrix(data = pred_f, reference = obs)$table


# AUC
rocplot=function(pred_class, obs, ...){
  predob = prediction(pred_class, obs)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

rocplot(pred_class, obs, main="ROC plot")

pred_ROCR <- prediction(pred_class, obs)
auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- round(auc_ROCR@y.values[[1]],2)
cat("Az nb count AUC erteke:", auc_ROCR) # 0.49

#head(predictions_nb_count)
#write.table(predictions_nb_count, file = "pred_nb_count.txt", sep = "\t",
#            row.names = TRUE)

#warnings()
# 48 megfigyelesnel panaszkodik, h 0 a valoszinuseg minden osztalyra
# nem segitett a laplace korrekcio sem (nem az a legjobb modell
# ahol korrigalunk)



# 1.2. naive bayes w tf data -------------------------------------------------------

require(caret)
require(glmnet)
library(Matrix)


train_control <- trainControl(
  method="cv", 
  number = 3, 
  savePredictions = TRUE)

# set up tuning grid
search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = legkisebb_tf_s, # laplace correction
  adjust = seq(0, 5, by = 1) # allows us to adjust the bandwidth of the 
  # kernel density (larger numbers mean more flexible density estimate)
)


set.seed(3)
model <- train(
  roma_felado ~ ., 
  data = train_tf, 
  trControl = train_control, 
  tuneGrid = search_grid,
  preProc = c("BoxCox", "center", "scale", "pca"), # scale: divides by the SD
  method = "nb")

# van vmi modszer a zero variance predictorokra: https://topepo.github.io/caret/pre-processing.html


get_best_result(model)

# cm
predictions_nb_tf <- model %>% predict(test_tf, type = "prob") %>% as.vector()

pred_class <- ifelse(predictions_nb_tf[,2] > 0.5, 1, 0)
pred_f = factor(pred_class, levels= c("0", "1"))
obs = as.factor(test_tf$roma_felado)

confusionMatrix(data = pred_f, reference = obs)$table


# AUC
rocplot=function(pred_class, obs, ...){
  predob = prediction(pred_class, obs)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

rocplot(pred_class, obs, main="ROC plot")

pred_ROCR <- prediction(pred_class, obs)
auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- round(auc_ROCR@y.values[[1]],2)
cat("Az nb count AUC erteke:", auc_ROCR) # 49 


# 1.3. naive bayes w tf_idf data -------------------------------------------------------

require(caret)
require(glmnet)
library(Matrix)


train_control <- trainControl(
  method="cv", 
  number = 3, 
  savePredictions = TRUE)

# set up tuning grid
search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = legkisebb_tf_idf_s, # laplace correction
  adjust = seq(0, 5, by = 1) # allows us to adjust the bandwidth of the 
  # kernel density (larger numbers mean more flexible density estimate)
)


set.seed(3)
model <- train(
  roma_felado ~ ., 
  data = train_tf_idf, 
  trControl = train_control, 
  tuneGrid = search_grid,
  preProc = c("BoxCox", "center", "scale", "pca"), # scale: divides by the SD
  method = "nb")

# van vmi modszer a zero variance predictorokra: https://topepo.github.io/caret/pre-processing.html


get_best_result(model)

# cm
predictions_nb_tf_idf <- model %>% predict(test_tf_idf, type = "prob") %>% as.vector()

head(text)
head(predictions_nb_tf_idf)


pred_class <- ifelse(predictions_nb_tf_idf[,2] > 0.5, 1, 0)
pred_f = factor(pred_class, levels= c("0", "1"))
obs = as.factor(test_tf$roma_felado)

confusionMatrix(data = pred_f, reference = obs)$table


# AUC
rocplot=function(pred_class, obs, ...){
  predob = prediction(pred_class, obs)
  perf = performance(predob, "tpr", "fpr")
  plot(perf,...)}

rocplot(pred_class, obs, main="ROC plot")

pred_ROCR <- prediction(pred_class, obs)
auc_ROCR <- performance(pred_ROCR, measure = "auc")
auc_ROCR <- round(auc_ROCR@y.values[[1]],2)
cat("Az nb count AUC erteke:", auc_ROCR) # 49 
