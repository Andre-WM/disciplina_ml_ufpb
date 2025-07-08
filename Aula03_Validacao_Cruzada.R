# imports
library(tidymodels)
library(tidyverse)

tidymodels_prefer()

# Leave one out CV - Converge quase certamente para o AIC
# É deterministico

# K-fold - k = n -> k-fold = LOOCV
# probabilistico

# Podem ser testasdos os resultados com intervalos de confiança bootstrap

# nested cv - validação cruzada dentro da validação cruzada

# a validação cruzada serve para a seleção de modelos
# utilizar o conjunto de testes para comparar dois modelos é uma forma de lekage

# pacote em R
library(rsample)

# balanço entre vies e variancia
# As vezes aplica-se um pouco de viés ao modelo para controlar-se a variancia
# A interpretaçao dos parametros do modelo exige a representatividade dos dados
# não é necessariamente parte do paradigma de aprendizado de maquina
# modelos muito complexos tem muita variancia, sofrendo de hiperajustes
# prejudicando também sua generalizaçao para novos dados e aumentando-se muito
# o risco preditivo (overfitting)

# modelos muito simples podem apresentar muito viés e falta de ajuste
# underfitting

# conforme prediction - Area de estudo de interpretaçao de modelos de ML
# interpretaçao de parametros

# tuning parameters - regressao polinomial com tidymodels
load("datasets/dados_expectativa_renda.RData")
dados_expectativa_renda |> head()

set.seed(29110)

# split para treino (com cv) e teste
data_split <- rsample::initial_split(
  dados_expectativa_renda, 
  prop = 0.8, 
  strata = LifeExpectancy
)

not_testing <- rsample::training(data_split)
testing <- rsample::testing(data_split)

# para divisão de treino, validaçao e teste, sem cv
# dados_split <- rsample::initial_validation_split(df, prop = c(0.6, 0.2))
# treino <- rsample::training(dados_split)
# validacao <- rsample::validation(dados_split)
# teste <- rsample::testing(dados_split)

# receita e tuning
receita <- 
  recipe(LifeExpectancy ~ GDPercapita, data = not_testing) |>
  step_poly(GDPercapita, degree = tune("p"))

# verificar se a receita funciona
# receita <- receita |> prep() - deve ser retirado o tune

# modelo
# penalty (lambda) e mixture são argumentos para regularização
# podem ser usados na engine glmnet
modelo <- parsnip::linear_reg(
  mode = "regression",
  engine = "lm"
)

wflow <- workflow() |> 
  add_recipe(receita) |> 
  add_model(modelo)

# validacao cruzada k-fold (k=v)
cv <- rsample::vfold_cv(not_testing, v = 3, repeats = 1)
cv_tidy <- tidy(cv)

# visualizando a cv
cv_tidy |>
  ggplot(aes(x = Fold, y = Row, fill = Data)) +
  geom_tile() +
  scale_fill_brewer() +
  theme_minimal()

# Tunando o modelo
tune_res <- tune_grid(
  wf,
  resamples = cv,
  grid = 11,
  control = control_grid(save_pred = TRUE)
)

# visualizando os melhores hiperparametros
tune_res |> show_best(metric = "rmse")