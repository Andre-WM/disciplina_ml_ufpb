# link para os slides da disciplina
# https://prdm0.github.io/curso_ml/#/title-slide

# meta pacotes
library(tidymodels)
library(tidyverse)

# Função risco preditivo: Função perda quadrática
# O erro quadratico medio converte q.c. quando i.i.d
# Para a sequencia dos eqm's calculados ser independente, o conjunto de testes 
# deve ser separado do de treino

# paralelismo
library(future)
plan("multisession")

# numero de cores
parallel::detectCores()

# hiperparametros: parametros não estimados a partir dos dados

# resolver dependencias no tidymodels
tidymodels_prefer() 