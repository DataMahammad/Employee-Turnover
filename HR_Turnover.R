library(tidyverse)
library(data.table)
library(rstudioapi)
library(skimr)
library(car)
library(h2o)
library(rlang)
library(glue)
library(highcharter)
library(lime)
library(inspectdf)


df <- fread("HR_turnover.csv")

df %>% dim()

df %>% inspect_na()

df %>% unique()

df %>% glimpse()

df$left <- df$left %>% as.factor()
  
df$Work_accident %>% unique()
df$promotion_last_5years %>% unique()

df$Work_accident <- df$Work_accident %>% as.factor()
df$promotion_last_5years <- df$promotion_last_5years %>% as.factor()
df$number_project <- df$number_project %>% as.factor()

boxplot(df[["satisfaction_level"]])
boxplot(df[["last_evaluation"]])
boxplot(df[["average_montly_hours"]])

df$number_project %>% unique()

df$left %>% table() %>% prop.table()


##########################---------MODELING-----------###############################
library(rJava)

Sys.setenv(JAVA_HOME= "C:\\Program Files\\Java\\jre1.8.0_271")
Sys.getenv("JAVA_HOME")


h2o.init(nthreads = -1, max_mem_size = '2g', ip = "127.0.0.1", port = 54321)


df_h2o <- df %>% as.h2o()

df_h2o <- df_h2o %>% h2o.splitFrame(ratio = 0.8, seed=123)
train <- df_h2o[[1]]
test <- df_h2o[[2]]


target <- 'left'
features <- df %>% select(-left) %>% names()

#?h2o.automl()
  
model <- h2o.automl(
  x = features,y = target,
  training_frame = train,
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "AUC",
  balance_classes = T,#in this case no need for imbalance
  nfolds = 10,seed=123,
  max_runtime_secs = 480
)


model@leaderboard %>% as.data.frame()

model@leader

#Prediction
pred <- model@leader %>% h2o.predict(test) %>% as.data.frame()

model@leader %>% 
  h2o.performance(test) %>% 
  h2o.find_threshold_by_max_metric('f1') -> treshold


model@leader %>% 
  h2o.confusionMatrix(test) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0, color = c("red", "darkgreen"),
               main = paste("Accuracy = ",
                            round(sum(diag(.))/sum(.)*100,1),"%"))

model@leader %>% 
  h2o.performance(test) %>% 
  h2o.metric() %>% 
  select(threshold,precision,recall,tpr,fpr) %>% 
  add_column(tpr_r=runif(nrow(.),min=0.001,max=1)) %>% 
  mutate(fpr_r=tpr_r) %>% 
  arrange(tpr_r,fpr_r) -> deep_metrics

model@leader %>% 
  h2o.performance(test) %>% 
  h2o.auc() %>% round(2) -> auc


#AUC Curve and value
highchart() %>% 
  hc_add_series(deep_metrics, "scatter", hcaes(y=tpr,x=fpr), color='green', name='TPR') %>%
  hc_add_series(deep_metrics, "line", hcaes(y=tpr_r,x=fpr_r), color='red', name='Random Guess') %>% 
  hc_add_annotation(
    labels = list(
      point = list(xAxis=0,yAxis=0,x=0.3,y=0.6),
      text = glue('AUC = {enexpr(auc)}'))
  ) %>%
  hc_title(text = "ROC Curve") %>% 
  hc_subtitle(text = "Model is performing much better than random guessing") 


#Check overfitting

model@leader %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)

#There's overfitting as model is perfectly fitted to train data
#Therefore,most probably we should apply Ridge or Lasso Regression




