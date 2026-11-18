# A tiny R calculator used to test how KiLin reads code.
library(methods)

#' Double a number.
r_function_B <- function(x) {
  2 * x
}

#' Double a, then add b.
r_function_A <- function(a, b) {
  r_function_B(a) + b
}

RBaseClass <- setRefClass("RBaseClass", fields = list(value = "numeric"))

# Adds b to the stored value, doubled.
RChildClass <- setRefClass(
  "RChildClass",
  contains = "RBaseClass",
  methods = list(
    r_method_C = function(b) {
      r_function_A(value, b)
    }
  )
)

#' Apply r_method_C to every number in a list.
r_function_D <- function(values) {
  calculator <- RChildClass$new(value = 1)
  sapply(values, function(b) calculator$r_method_C(b))
}
