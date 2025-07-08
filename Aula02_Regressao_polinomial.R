# Regressão polinomial e importancia do data split

library(ggplot2)
library(dplyr)
library(purrr)
library(patchwork)
library(rsample)

# Generate data
gerando_dados <- function(n = 500L, ...) {
  f <- function(x, sd = 0.5, ...) {
    45 * tanh(x/1.9 - 7) + 57 + rnorm(n = length(x), sd = sd, ...)
  }

  tibble(x = runif(n = n, min = 0, max = 20)) |>
    mutate(y = f(x, ...))
}

set.seed(123)
dados <- gerando_dados(n = 10000L, mean = 0, sd = 5.5)

# Create a prediction function that includes all polynomial terms
predict_fun <- function(x) {
  newdata <- tibble(
    x = x,
    x2 = x^2,
    x3 = x^3,
    x4 = x^4,
    x5 = x^5,
    x6 = x^6,
    x7 = x^7
  )
  predict(modelo, newdata = newdata)
}

dados1 <- dados |>
  mutate(
    x2 = x^2,
    x3 = x^3,
    x4 = x^4,
    x5 = x^5,
    x6 = x^6,
    x7 = x^7
  )

# Fit model
modelo <- lm(y ~ ., data = dados1)

# Plot
ggplot(data = dados, aes(x = x, y = y)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "blue") +
  geom_function(fun = predict_fun, color = "red", size = 1) +
  labs(
    title = "Polynomial Regression (degree 7) vs Simple Linear Regression",
    subtitle = "Red: 7th degree polynomial fit | Blue: Simple linear regression"
  )

# alternativa para ajuste de uma regressão polinomial
poly_model <- lm(y ~ poly(x, degree = 2), data = dados)

# calcula o eqm para cada grau de polinomio p
# avaliação sem data split (holdout)
avaliacao_ingenua <- function(dados, p_max = 25) {
  avaliacao <- function(p) {
    regressao <- lm(y ~ poly(x, degree = p), data = dados)

    r <- dados |>
      mutate(y_chapeu = predict(regressao)) |>
      summarise(eqm = mean((y - y_chapeu)^2))

    tibble(p = p, eqm = r$eqm)
  }
  purrr::map_dfr(1:p_max, avaliacao)
}

# Avaliação com holdout (ajustada)
avaliacao_holdout <- function(dados, p_max = 25, seed = 123, ...) {
  split <- initial_split(dados, prop = 0.8, strata = y, ...)
  treino <- training(split)
  teste <- testing(split)
  avaliacao <- function(p) {
    regressao <- lm(y ~ poly(x, degree = p), data = treino)

    r <- teste |>
      mutate(y_chapeu = predict(regressao, newdata = teste)) |>
      summarise(eqm = mean((y - y_chapeu)^2))

    tibble(p = p, eqm = r$eqm)
  }
  purrr::map_dfr(1:p_max, avaliacao)
}

# Avaliações
tbl_ingenua <- avaliacao_ingenua(dados, p_max = 25)
tbl_holdout <- avaliacao_holdout(dados, p_max = 25)

# Gráficos
p1 <- tbl_ingenua |>
  ggplot(aes(x = p, y = eqm)) +
  geom_line() +
  labs(title = "Avaliação Ingênua (treino)")

p2 <- tbl_holdout |>
  ggplot(aes(x = p, y = log(eqm))) +
  geom_line() +
  geom_point(data = slice_min(tbl_holdout, eqm), color = "red", size = 3) +
  labs(title = "Avaliação Holdout (teste)", y = "log(EQM)") +
  scale_x_continuous(breaks = seq(1, 25, 2))

# Visualização dos dados
p_dados <- dados |>
  ggplot(aes(x = x, y = y)) +
  geom_point(alpha = 0.5) +
  labs(title = "Dados Observados")

# Layout final
(p_dados) / (p1 | p2)

# Plotando sobre p_dados o polinômio de melhor grau, polinômio de grau 6
melhor_grau <- which.min(tbl_holdout$eqm)
dados |>
  mutate(
    y_chapeu = predict(lm(y ~ poly(x, degree = melhor_grau), data = dados))
  ) |>
  ggplot(aes(x = x, y = y)) +
  geom_point(alpha = 0.5) +
  geom_line(aes(y = y_chapeu), color = "blue", size = 1) +
  labs(title = paste("Polinômio de grau", melhor_grau, "ajustado aos dados")) +
  theme_minimal()
