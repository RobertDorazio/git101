### EXAMPLE OF FITTING A REGRESSION MODEL OF BINOMIAL COUNTS

using Random
using Distributions
using StatsBase

using DataFrames
using GLM
using LinearAlgebra
using Optim
using ForwardDiff  # needed to evaluate the hessian


### Simulate data


## ... specify distribution for computing cdf

logis = Logistic()


Random.seed!(6789) # Set the seed for reproducibility


n = 100

x = randn(n)                 # Predictor variable

true_logit = -0.5 .+ 1.8 .* x  # True linear predictor: beta[0] + beta[1] * x

p = cdf(logis, true_logit)

N_set = collect(5:25)
N = sample(N_set, n, replace=true)

y = rand.(Binomial.(N, p))   # Binomial counts



### Create a DataFrame

data = DataFrame(success = y, failure = N .- y, x = x)


## ... name the weights column 'wts' (which is used by GLM.jl automatically (see below))
data[!, :wts] = data.success .+ data.failure

## ... Calculate the proportion (response variable)
data[!, :proportion] = data.success ./ (data.success .+ data.failure)



### Fit the model using glm() and its @formula macro


## ... this call to glm fails

# fit_glm = glm(@formula(proportion ~ x), data, Binomial(), LogitLink() wts=wts)



## ... this call to glm works

fit_glm = glm(@formula(proportion ~ x), data, Binomial(), LogitLink(), wts=data.wts )


##println(fit_glm)


### Compute estimates of regression coeffs and their vcov matrix

beta_glm = coef(fit_glm)

vcv_glm = vcov(fit_glm)

se_glm = sqrt.(diag(vcv_glm))

println("\n", "\n", "Analysis using GLM in Julia language", "\n")

println("Beta = ", beta_glm)

println("  se = ", se_glm)





### Fit the model using numerical maximization of the log-likelihood function


## ... define objective function

function negLogLike(param, y, N, x)

    beta = param

    eta = beta[1] .+ beta[2] .* x
   
    logL = dot(y, eta) - sum( N .* log.(1.0 .+ exp.(eta)) )

    return (-1.)*logL

end


## ... compute MLE using optimize()

paramGuess = [0., 0.]

fit_mle = optimize(param -> negLogLike(param, y, N, x),  paramGuess,  BFGS() )


println("\n", "\n", "Analysis using numerical optimization", "\n")


if Optim.converged(fit_mle)

    ## ... extract the estimated parameters
    beta_mle = Optim.minimizer(fit_mle)


    ## ... compute hessian and its inverse to obtain variance-covariance matrix

    obj_func_wrapper = param -> negLogLike(param, y, N, x)

    H = ForwardDiff.hessian(obj_func_wrapper, beta_mle)  # evaluate hessian at MLE

    vcv_mle = inv(H)  # invert hessian

    se_mle = sqrt.(diag(vcv_mle))


    println("Beta = ", beta_mle)

    println("  se = ", se_mle)

else

    println("Convergence failure during numerical optimization")
end






### Fit the model using R language's glm() function

using RCall

@rput data

R"""

fit_r = glm(cbind(success, failure) ~ x, data, family=binomial)
beta_r = fit_r$coefficients

vcv_r = vcov(fit_r)

se_r = sqrt(diag(vcv_r))
"""

@rget beta_r  se_r


### Compute estimates of regression coeffs and their vcov matrix

println("\n", "\n", "Analysis using GLM in R language", "\n")

println("Beta = ", beta_r)

println("  se = ", se_r)

