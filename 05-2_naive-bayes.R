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
library(Matrix)


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

# lets scale variables, wt centering

dat_count_X <- subset(dat_count_ns, select = -c(roma_felado))
dat_count_X_s = as.data.frame(scale(dat_count_X, scale = T, center = T))
dat_count = as.data.frame(cbind(dat_count_X_s, roma_felado = dat_count_ns$roma_felado))

dat_tf_X <- subset(dat_tf_ns, select = -c(roma_felado))
dat_tf_X_s = as.data.frame(scale(dat_tf_X, scale = T, center = T))
dat_tf = as.data.frame(cbind(dat_tf_X_s, roma_felado = dat_tf_ns$roma_felado))

dat_tf_idf_X <- subset(dat_tf_idf_ns, select = -c(roma_felado))
dat_tf_idf_X_s = as.data.frame(scale(dat_tf_idf_X, scale = T, center = T))
dat_tf_idf = as.data.frame(cbind(dat_tf_idf_X_s, roma_felado = dat_tf_idf_ns$roma_felado))


# nearzero var kezelese
# ez meg lefut pca-val: 
j = nearZeroVar(dat_count, saveMetrics = F, freqCut = 20, uniqueCut = 2, names = F) # 167 db prediktor


# explanation: https://rstatisticsblog.com/data-science-in-action/data-preprocessing/how-to-identify-variables-with-zero-variance/
# a naive bayeshez meg kell nezni a valtozok felteteles eloszalasat (as suggested by neaZero help)

dat_count_nzv = dat_count[,-j] # megnyolcadoltuk a valtozok szamat
dat_tf_nzv = dat_tf[,-j]
dat_tf_idf_nzv = dat_tf_idf[,-j]

# check conditionals
# checkConditionalX(x = dat_count_nzv, y = dat_count_nzv$roma_felado)
# there are 153 columns of x that are sparse in y

a = ncol(dat_tf_nzv)-18
b = dat_tf_nzv[,1:a]
legkisebb_tf = min( abs(b)[abs(b)!=abs(min(b))] )

a = ncol(dat_tf_idf_nzv)-18
b2 = dat_tf_idf_nzv[,1:a]
legkisebb_tf_idf = min( abs(b2)[abs(b2)!=min(abs(b2))] )

# stringek laplace korrekciohoz:
e =  0:5
legkisebb_tf_s = legkisebb_tf * e
legkisebb_tf_idf_s = legkisebb_tf_idf * e



###########
# elemzes #
###########

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


train_control <- trainControl(
  method="cv", 
  number = 3, 
  savePredictions = TRUE,
  classProbs = TRUE,
  summaryFunction = twoClassSummary)

# set up tuning grid
search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = c(0,1), # laplace correction, azert van, ha egyik csoport erteke 0 vmelyik kategoriaban, akkor ne legyen az egesz egyenloseg 0.
  adjust = seq(0, 5, by = 1) # allows us to adjust the bandwidth of the 
  # kernel density (larger numbers mean more flexible density estimate)
  # ez a folytonos, nem normallis eloszlasu valtozokhoz kell
)

set.seed(3)
start.time <- Sys.time()

model <- train(
  roma_felado ~ ., 
  data = dat_count_nzv, 
  trControl = train_control, 
  tuneGrid = search_grid, 
  preProcess = "pca",
  method = "nb",
  metric = "ROC")

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken 

# pca nelkul csak par tucat prediktorral fut le az nb modell.

print(model)
get_best_result(model)

caret::confusionMatrix(model)
tab <- round(caret::confusionMatrix(model)$table,0)
"A legjobb modellhez tartozo tevesztesi matrix:"
tab # szazalekban van 

roc = round(get_best_result(model)[[4]],2)
cat("A legjobb modellhez tartozo AUC ertek:", roc)


# feature importance

enetImp = varImp(model, scale = F) 
plot(enetImp, top = 10)




# 1.2. naive bayes w tf data -------------------------------------------------------

# set up tuning grid
search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = c(0,1), # laplace correction
  adjust = seq(0, 5, by = 1) # allows us to adjust the bandwidth of the 
  # kernel density (larger numbers mean more flexible density estimate)
)


set.seed(3)
start.time <- Sys.time()

model_tf <- train(
  roma_felado ~ ., 
  data = dat_tf_nzv, 
  trControl = train_control, 
  tuneGrid = search_grid,
  preProcess = "pca",
  method = "nb",
  metric = "ROC")

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken 

get_best_result(model_tf)

caret::confusionMatrix(model_tf)
tab <- round(caret::confusionMatrix(model_tf)$table,0)
"A legjobb modellhez tartozo tevesztesi matrix:"
tab # szazalekban van 

roc = round(get_best_result(model_tf)[[4]],2)
cat("A legjobb modellhez tartozo AUC ertek:", roc)

# feature importance

enetImp = varImp(model_tf, scale = F) 
plot(enetImp, top = 10)



# 1.3. naive bayes w tf_idf data -------------------------------------------------------

# set up tuning grid
search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = c(0,1), # laplace correction
  adjust = seq(0, 5, by = 1) # allows us to adjust the bandwidth of the 
  # kernel density (larger numbers mean more flexible density estimate)
)


set.seed(3)
start.time <- Sys.time()

model_tf_idf <- train(
  roma_felado ~ ., 
  data = dat_tf_idf_nzv, 
  trControl = train_control, 
  tuneGrid = search_grid,
  preProcess = "pca",
  method = "nb",
  metric = "ROC")

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken 

get_best_result(model_tf_idf)

caret::confusionMatrix(model_tf_idf)
tab <- round(caret::confusionMatrix(model_tf_idf)$table,0)
"A legjobb modellhez tartozo tevesztesi matrix:"
tab # szazalekban van 

roc = round(get_best_result(model_tf_idf)[[4]],2)
cat("A legjobb modellhez tartozo AUC ertek:", roc)


# feature importance

enetImp = varImp(model_tf_idf, scale = F) 
plot(enetImp, top = 10)