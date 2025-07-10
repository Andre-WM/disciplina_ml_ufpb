# imports
library(tidymodels)
library(tidyverse)
library(finetune)

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

# leave one out cv (not supported in tune_grid)
loocv <- rsample::loo_cv(not_testing)

# visualizando a cv
cv_tidy |>
  ggplot(aes(x = Fold, y = Row, fill = Data)) +
  geom_tile() +
  scale_fill_brewer() +
  theme_minimal()

# Tunando o modelo
tune_res <- tune_grid(
  wflow,
  resamples = cv,
  grid = 10,
  control = control_grid(save_pred = TRUE),
  metrics = yardstick::metric_set(rmse)
)

# Métodos de otimização de hiperparâmetros
# 1. Grid Search: 
# Testa as possiveis combinações estabelecidas em um espço de busca fixado
# tune::tune_grid()
# determinístico
# custo alto por iteração

# 2. Randomized search:
# Escolhe aleatoriamente n combinações no espaço paramétrico dos hiperparametros
# tune::tune_grid(..., search = "random)
# pode ser mais eficiente em grandes espaços
# não garante o melhor ponto e é aleatório
# custo medio por iteração

# 3. Adaptative Selection
# Avalia estatisticamente os candidatos a melhor hiperparametro na validação
# descarta os piores e segue buscando dentre os melhores
# tune_grid(..., control = control_grid(adaptative = TRUE))
# pode descartar bons candidatos cedo
# custo baixo a medio por iteração

# 4. Otimização bayesiana
# Modela a função de avaliação como uma distribuição probabilistica e seleciona
# hiperparâmetros que maximizem uma função aquisição (ex: expected improvement)
# tune_bayes()
# as proximas mediçoes são otimizadas com as anteriores (otimização inteligente)
# custo alto por iteração

# 5. Simulated Annealing
# admite com certa probabilidade soluções levemente piores para escapar de 
# minimos locais. A probabilidade diminui com as iterações.
# ainda mais leve que um grid search, mas ainda assim lento 
# e com muitos parametros. Custo baixo por iteraçao.
# finetune::tune_sim_anneal()

# 6. Racing (Hyperband, Successive Halving)
# Avaliam muitos hiperparâmetros com menos recursos (ex: menos folds ou dados) 
# e vão alocando mais recursos apenas para os melhores candidatos.
# Muito eficiente e escala bem para grandes espaços
# Pode ser sensivel a escolhas iniciais
# finetune::tune_race_win_loss() ou finetune::tune_race_anova()
# custo muito baixo a medio por iteraçao

# intervalos de confiança para o rmse
tune::int_pctl(tune_res)

# visualizando os melhores hiperparametros
tune_res |> show_best(metric = "rmse")

# plot do tune
autoplot(tune_res) + theme_minimal()

# ajuste do modelo com o p escolhido com todo o conjunto not testing
# (remove a validaçao)
best_params <- select_best(tune_res, metric = "rmse")
final_wf <- finalize_workflow(wflow, best_params)

modelo_final <- tune::last_fit(final_wf, split = data_split)

# extrair o modelo para novas previsões
final_model <- extract_workflow(modelo_final)
predict(final_model, new_data = your_new_data)
