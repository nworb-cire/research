### A Pluto.jl notebook ###
# v0.19.9

#> [frontmatter]
#> author = "Eric Brown"
#> title = "AutoTune"
#> date = "2022-06-24"
#> tags = ["julia", "machine learning", "hyperparameters", "optimization", "optimisation", "data science", "mathematics", "applied mathematics"]
#> description = "A system for automatic hyperparameter tuning in Julia"
#> contact = "eric@ebrown.net"

using Markdown
using InteractiveUtils

# ╔═╡ b5a66c84-f3e2-11ec-397f-63f315fee830
using Optim, Zygote, MLDatasets, Flux, DataFrames, ForwardDiff

# ╔═╡ bb4fcd36-c83a-41b5-ab71-87271631482e
using Random: shuffle

# ╔═╡ 69792a6e-e520-40be-afcb-0027b3d5f39a
using ProgressLogging

# ╔═╡ bbd3d94e-c8d1-4cdd-8330-9b4003792f5a
using Statistics: mean

# ╔═╡ c15b83ea-a4e0-440e-bab0-a61c9ee478f7
using DiffResults: DiffResult

# ╔═╡ 162622ba-dcf8-4e81-9726-da2ffc311ccc
md"""
# AutoTune.jl
### A system for automatic hyperparameter tuning

#### Created by Eric Brown
#### Contact: [ebrown1@snapfinance.com](mailto:ebrown1@snapfinance.com) / [eric@ebrown.net](mailto:eric@ebrown.net)
"""

# ╔═╡ 2bf3f4a0-2e1d-4710-b938-7c680b2ea795
html"<button onclick='present()'>present</button>"

# ╔═╡ 748cd332-07fe-4095-8138-f7bbf8634506
test_range = 1e-2:3e-2:1 ;

# ╔═╡ 3fd3f317-d292-4572-a335-75ca91bc7793
begin
	U = Union{T, ForwardDiff.Dual{K,T,N}} where {K,T<:Real,N}
	V = Union{Vector{T}, Vector{U}} where T<:Real
end ;

# ╔═╡ c698281c-23b1-48c1-90cc-0b844bedd367
begin
	idxs = shuffle(collect(1:150))
	train_idxs = idxs[1:100]
	val_idxs = idxs[101:130]
	test_idxs = idxs[131:150]
end;

# ╔═╡ 2a8cd318-2c70-463e-9ecc-053c076ca5bc
md"""
# Introduction

The hyperparameter problem can be phrased as follows:

$\lambda^* = \min_\lambda \mathscr L_{test}(\lambda)$
where

$$\mathscr L_{test}(\lambda) = \min_i \mathscr L_{val}(\theta^{(i)}, \lambda)$$
where $\{\theta^{(i)}\}_{i=1}^N$ is a sequence obtained from an iterative optimization procedure (e.g. ADAM) giving

$\theta^{(i+1)} = step(\mathscr L, \mathbf x, \mathbf y,\theta^{(i)}, \lambda)$
for fixed dataset $\mathbf x, \mathbf y$, loss (objective) function $\mathscr L$ and given $\theta^{(1)}$.

Generally, $\mathscr L_{val}$ and $\mathscr L_{test}$ are identical to $\mathscr L$ but with different datasets $(\mathbf x_{val}, \mathbf y_{val})$ and $(\mathbf x_{test}, \mathbf y_{test})$.
"""

# ╔═╡ eb864c8e-e8db-4863-b0dc-a258c5de65b6
md"# Proof of concept"

# ╔═╡ da773c77-db69-4595-bab2-2f6ac3653645
md"""
Define a function $f: \mathbb R \times \mathbb R^n \to \mathbb R$ as follows:

$f_\lambda(\Theta) = \|\Theta-\lambda\|_2^2 + |\lambda|^2$
"""

# ╔═╡ 9aadd03d-619a-4628-b50e-9f3eb4d60e5d
f(Θ ; λ) = sum(abs2, Θ .- λ) + abs2(λ) ;

# ╔═╡ ae06dfcd-7123-4286-a9cb-01cf2597dbc3
md"For all values of $\lambda$, this function attains a minimum at $\theta_1=\theta_2=\cdots=\lambda$. Clearly the global minimum is at $\lambda=0$."

# ╔═╡ 3264e605-d411-4681-87df-2e8d4517dd23
md"""
## Minimizer

For any given value of $\lambda$, we can compute the minimizer $\Theta^\star_\lambda = \arg\min_\Theta f_\lambda(\Theta)$ using any optimization method. Here we use the first-order gradient descent method.
"""

# ╔═╡ fc940c6c-5517-4eb2-b724-99960f6b9e93
begin
function argmin_f(λ ; Θ₀::AbstractVector)
    opt = Descent()
    Θ = copy(Θ₀)
    for _ in 1:2_050
        g = Zygote.gradient(Θ) do Θ
            f(Θ ; λ)
        end
        Flux.update!(opt, Θ, g...)
    end
    return Θ
end
argmin_f(λ, N::Integer) = argmin_f(λ ; Θ₀ = Array{U}(randn(N))) 
argmin_f(λ) = argmin_f(λ, 1)
end ;

# ╔═╡ c78facd5-d78f-4f7a-9e67-75fa1983b407
# ╠═╡ show_logs = false
let λ = randn()
	N = 10
	@assert all(argmin_f(λ, N) .≈ λ)
end

# ╔═╡ 15e9045f-cf61-4c9c-9ea2-e0bf4a63e9e8
md"""
##

Let $g(\lambda) = \min_\Theta f_\lambda(\Theta) = f_\lambda(\Theta^\star_\lambda)$
"""

# ╔═╡ 9d629ee8-6530-4216-9083-a8fc558f8fb7
begin
g(λ, N) = f(argmin_f(λ, N) ; λ)
g(λ) = g(λ, 1)
end ;

# ╔═╡ c8b10d6e-ba40-4cb2-8c20-ff70f1322b3d
md"""
Recall that

$f_\lambda(\Theta) = \|\Theta-\lambda\|_2^2 + |\lambda|^2$

From our definition of $f_\lambda$ above, we know that $g(\lambda) = \lambda^2$ analytically. We verify this numerically."""

# ╔═╡ 8359f15b-0a3c-468f-8494-8715062f811a
# ╠═╡ show_logs = false
@assert all(g.(test_range) .≈ test_range.^2)

# ╔═╡ 2e444ede-6db6-4b63-84aa-21b399ce696b
md"""Using forward-mode differentiation, we can now compute the derivative of $g$ with respect to $\lambda$:"""

# ╔═╡ 3a06f2ab-b917-4657-8000-c29b8224d027
md"This should be equal to $2\lambda$, again this is verified numerically."

# ╔═╡ 3c856bcf-29f9-4753-a2bb-985580fccc89
md"""##
This works for higher dimensions as well:"""

# ╔═╡ d37298c8-ba86-41ec-9b7a-c00e2177b464
begin
∂g(λ, N) = ForwardDiff.derivative(λ) do λ; g(λ, N) end
∂g(λ) = ∂g(λ, 1)
end ;

# ╔═╡ 0601375e-7151-4968-a126-c79b5efa30be
@assert all(∂g.(test_range) .≈ 2test_range)

# ╔═╡ 7957e09a-ecad-4cc2-9a89-5c4c87923c25
let λ = rand(), N = 5
	@assert g(λ, N) ≈ λ^2
	@assert ∂g(λ, N) ≈ 2λ
end

# ╔═╡ 36d1685c-ca88-4665-94a4-4393987adb5d
md"""
# Application to neural networks

In the above proof of concept, we can consider the function $f_\lambda$ as an objective (loss) function over $\mathbb R^n$ with hyperparameter(s) $\lambda$. 
$\Theta$ are the network parameters and $\Theta^\star_\lambda$ is the optimum value given $\lambda$. 
"""

# ╔═╡ 1dc7bb61-c071-420f-bd31-aa98de06faa7
md"""In the example below, we use a bilayer feedforward network to predict entries in the Iris dataset."""

# ╔═╡ deddd641-d0c4-4635-9dea-e8055675386b
begin
	dataset = Iris()
	dataset
end

# ╔═╡ c5d821e1-7d8c-4118-8991-2c0138161a86
describe(dataset.dataframe)

# ╔═╡ a2d0f440-152a-44ac-913d-9485c085908d
classes = Array(unique(dataset.targets)) |> vec

# ╔═╡ 76a1e2d9-58a5-4761-88de-f793733b97b7
begin
	X_train = [Array(dataset[i].features) for i ∈ train_idxs]
	Y_train = [Flux.onehot(dataset[i].targets[1], classes) |> float for i ∈ train_idxs]

	X_val = [Array(dataset[i].features) for i ∈ val_idxs]
	Y_val = [Flux.onehot(dataset[i].targets[1], classes) |> float for i ∈ val_idxs]
	
	X_test = [Array(dataset[i].features) for i ∈ test_idxs]
	Y_test = [Flux.onehot(dataset[i].targets[1], classes) |> float for i ∈ test_idxs]
end ;

# ╔═╡ 508ace1e-afc2-4482-91f2-9d308dc5c5f6
md"In this example, we will call the features $X$ and the labels $Y$."

# ╔═╡ 5cb12a32-8636-4e1d-9fcb-b823b5d6fad5
# TODO: Static starting matrices
begin
	hidden_dim = 100
	model = Chain(
		Dense(4, hidden_dim, relu),
		Dense(hidden_dim, 3),
		softmax
	)
	model
end

# ╔═╡ f74c8718-8061-4071-a6c0-3879c47d9960
md"""## Model definition

Create a bilayer feedforward neural network NN with parameters $\Theta$. The network has a hidden dimension $hidden_dim.
"""

# ╔═╡ 108516ca-5a64-4aea-a078-437eb55f7f28
Θ, NN = Flux.destructure(model) ;

# ╔═╡ e4377d81-593f-4ea3-8364-20a65a1b1275
md"""
## Loss function

The loss function (corresponding to the function $f_\lambda$ in the previous example) is standard crossentropy loss with $\ell_1$ parameter regularization:

$\mathscr L_{xe}(\hat y, y) = -\sum y\log(\hat y)$

$\mathscr L(x, y, \lambda) = \mathscr L_{xe}(NN_\Theta(x), y) + \frac{\lambda}{\#\Theta} \|\Theta\|_1$
"""

# ╔═╡ 639a1bb0-1a29-48ab-b35c-e56b3f57b52c
crossentropy(ŷ, y, ϵ = eps(Float32)) = -sum(y .* log.(ŷ .+ ϵ)) ;

# ╔═╡ f77ab122-3aaf-47b1-ae46-edf17c20fc5c
function loss(Θ, X::T, Y::T ; λ) where T
	errs = map(zip(X, Y)) do (xᵢ, yᵢ)
		ŷᵢ = NN(Θ)(xᵢ)
		crossentropy(ŷᵢ, yᵢ)
	end
	λ = exp(λ)  # To make λ > 0
	return mean(errs) + λ*mean(abs, Θ)
end ;

# ╔═╡ 36ec1f9c-8038-46eb-aaee-747d4c97bb18
maxiters = 4_000 ;

# ╔═╡ 8d8671ec-86e9-407b-88dc-710258398a86
md"""
## Training

We train the neural network with a standard gradient descent training loop. For the sake of time, we put a cap at $maxiters iterations, but any stopping criteria will work just as well.
"""

# ╔═╡ a9cea07e-948c-43be-b3dd-dae29a78aeb5
function train(λ ; Θ′ = copy(Θ), maxiters = maxiters)
	η = 1e-4
	tol = 1e-7
	min_val_loss = Inf
	last_val = Inf
    @progress for i in 1:maxiters
        (l,ΔΘ) = Zygote.withgradient(Θ′) do Θ
            loss(Θ, X_train, Y_train ; λ)
		end
		val_loss = loss(Θ, X_val, Y_val ; λ)

		# Convergence checks
		if val_loss > min_val_loss
			@info "Validation increasing"
			break
		else
			min_val_loss = val_loss
		end
		if i > 10 && abs(last_val - l) < tol
			@info "Converged after $(i-1) iterations with abs tol $tol" last=last_val curr=l
			@show last_val
			@show l
			break
		else
			last_val = l
		end

        Θ′ = Θ′ - η * ΔΘ[1]
    end
    return Θ′
end ;

# ╔═╡ ba997393-48dc-4277-86d7-2d62155d6b8d
md"This is a standard training loop and behaves as expected."

# ╔═╡ 76078acb-bdc9-4122-99a7-0169138aef99
let λ = randn()
	before = loss(Θ, X_train, Y_train ; λ)
	Θᵒᵖᵗ = train(λ ; maxiters = 10)
	after = loss(Θᵒᵖᵗ, X_train, Y_train ; λ)

	@info "Loss difference: $(before-after)" before=before after=after
	@assert after < before
end

# ╔═╡ 17b82aa8-39a9-4c4e-a41f-5dfd3bf9ce3a
md"""## 
After training, we can evaluate the test loss.
"""

# ╔═╡ 38960a16-c49f-49f7-9cf7-319bc3e4e272
function test_loss(λ ; maxiters = maxiters)
	Θᵒᵖᵗ = train(λ ; maxiters)
	return loss(Θᵒᵖᵗ, X_test, Y_test ; λ)
end ;

# ╔═╡ 701a1065-23f5-4ba9-a565-67eaccd74c2c
# ╠═╡ show_logs = false
test_loss(randn() ; maxiters = 10)

# ╔═╡ 8a85ffa5-2de4-4b8c-a322-5b68d84f7f17
md"""

Just like before, we can use forward mode differentiation to differentiate the validation loss w.r.t. $\lambda$!
"""

# ╔═╡ 40ca36f8-ce8e-47cb-9a93-9a585bb815ab
begin
∂test_loss(λ ; kw...) = ForwardDiff.derivative(λ) do λ; test_loss(λ ; kw...) end 
∂test_loss(λ) = ∂test_loss(λ ; maxiters = maxiters)
end;

# ╔═╡ 580f758f-f443-4ddb-a040-1aafd9d4b758
∂test_loss(randn() ; maxiters = 10)

# ╔═╡ fffb27d2-a354-4b50-b0ae-cec66d0c2a39
md"""
##

We can now run an optimizer on the hyperparameter $\lambda$.
"""

# ╔═╡ c1e02298-0e2c-41e8-8ddf-ce1d4647859d
tol_λ = 1e-6

# ╔═╡ 2d681f34-18b7-44a4-bd35-da8899a556ad
let λ = 0.
	η = 0.1

	before = test_loss(λ)

	last_λ = Inf
	for i ∈ 1:15
		Δλ = ∂test_loss(λ)
		# Gradient descent step
		λ = λ - η*Δλ
		if i ≥ 3 && abs(λ - last_λ) < tol_λ
			@info "Hyperparameter tuning converged"
			break
		else
			last_λ = λ
		end
	end
	
	after = test_loss(λ)
	@info "Difference in test loss is $(before-after)" before=before after_=after λ_final=λ λ_exp = exp(λ)
end

# ╔═╡ b9280b33-8da5-4d17-9124-28b10c42aa43
md"This process is painfully slow, because we have to wait for one full training cycle between each update to $\lambda$. If only there were a way of accelerating this..."

# ╔═╡ 2477e412-89cc-4970-99bd-ee0c58b5052c
md"""
> Because of the magic of dual numbers, we can actually compute the second derivative and run Newton's method with *no* performance penalty, only a small bit of memory overhead. This has the added benefit of dramatically reducing the number of training cycles.
>
> The same goes for general $n$-th order derivatives, if we wanted to run an $n$-th order optimization method.
"""

# ╔═╡ a02fe1c6-ec7f-450a-a125-2da5f31c1611
begin
∂2test_loss!(res::DiffResult, λ ; maxiters) = ForwardDiff.derivative!(res, x->∂test_loss(x ; maxiters), λ)
∂2test_loss!(res::DiffResult, λ) = ∂2test_loss!(res, λ ; maxiters = maxiters)
end ;

# ╔═╡ 10d48c95-3c85-475c-b700-5e87c9aea22d
let λ = 0.
	η = 0.1
	ϵ = eps(λ)

	before = test_loss(λ)
	
	dr = DiffResult(0., 0.,)
	last_λ = Inf
	
	for i ∈ 1:10
		dr = ∂2test_loss!(dr, λ)
		λ′ = dr.value
		λ″ = dr.derivs[1]

		# Newton update
		λ = λ - η*(λ′/(λ″+ϵ))
		
		if i ≥ 3 && abs(λ - last_λ) < tol_λ
			@info "Hyperparameter tuning converged"
			break
		else
			last_λ = λ
		end
	end
	
	after = test_loss(λ)
	@info "Difference in test loss is $(before-after)" before=before after_=after λ_final=λ λ_exp = exp(λ)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
DataFrames = "~1.3.4"
DiffResults = "~1.0.3"
Flux = "~0.13.3"
ForwardDiff = "~0.10.30"
MLDatasets = "~0.7.2"
Optim = "~1.7.0"
ProgressLogging = "~0.1.4"
Zygote = "~0.6.40"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "63117898045d6d9e5acbdb517e3808a23aa26436"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.14"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterface]]
deps = ["ArrayInterfaceCore", "Compat", "IfElse", "LinearAlgebra", "Static"]
git-tree-sha1 = "1d062b8ab719670c16024105ace35e6d32988d4f"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.18"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "5e732808bcf7bbf730e810a9eaafc52705b38bb5"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.13"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[deps.BufferedStreams]]
git-tree-sha1 = "bb065b14d7f941b8617bc323063dbe79f55d16ea"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.1.0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "e4e5ece72fa2f108fb20c3c5538a5fa9ef3d668a"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.11.0"

[[deps.ChainRules]]
deps = ["ChainRulesCore", "Compat", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics"]
git-tree-sha1 = "97fd0a3b7703948a847265156a41079730805c77"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.36.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "8ccaa8c655bc1b83d2da4d569c9b28254ababd6e"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataDeps]]
deps = ["BinaryProvider", "HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "e299d8267135ef2f9c941a764006697082c1e7e8"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.8"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "4391d3ed58db9dc5a9883b23a0578316b4798b1f"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.0"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "9267e5f50b0e12fdfd5a2455534345c4cf2c7f7a"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.14.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FiniteDiff]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "ee13c773ce60d9e95a6c6ea134f25605dce2eda3"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.13.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "ArrayInterface", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "Optimisers", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Test", "Zygote"]
git-tree-sha1 = "62350a872545e1369b1d8f11358a21681aa73929"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.3"

[[deps.FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "eb8e1989b9028f7e0985b4268dabe94682249025"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.1"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[deps.Functors]]
git-tree-sha1 = "223fffa49ca0ff9ce4f875be001ffe173b2b7de4"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "73a4c9447419ce058df716925893e452ba5528ad"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.4.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "4078d3557ab15dd9fe6a0cf6f65e3d4937e98427"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "47f63159f7cb5d0e5e0cfd2f20454adea429bec9"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.16.1"

[[deps.GZip]]
deps = ["Libdl"]
git-tree-sha1 = "039be665faf0b8ae36e089cd694233f5dee3f7d6"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.5.1"

[[deps.Glob]]
git-tree-sha1 = "4df9f7e06108728ebf00a0a11edee4b29a482bb2"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "9ffc57b9bb643bf3fce34f3daf9ff506ed2d8b7a"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.10"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "bab67c0d1c4662d2c4be8c6007751b0b6111de5c"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.1+0"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "af14a478780ca78d5eb9908b263023096c2b9d64"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.6"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageShow]]
deps = ["Base64", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "b563cf9ae75a635592fc73d3eb78b86220e55bd8"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.6"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "81b9477b49402b47fbe7f7ae0b252077f53e4a08"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.22"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "fd6f0cae36f42525567108a42c1c674af2ac620d"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.5"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "e7e9184b0bf0158ac4e4aa9daf00041b5909bf1a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.14.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "971be550166fe3f604d28715302b58a3f7293160"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.3"

[[deps.MLDatasets]]
deps = ["CSV", "DataDeps", "DataFrames", "DelimitedFiles", "FileIO", "FixedPointNumbers", "GZip", "Glob", "HDF5", "ImageShow", "JLD2", "JSON3", "LazyModules", "MAT", "MLUtils", "NPZ", "Pickle", "Requires", "SparseArrays", "Tables"]
git-tree-sha1 = "58c658a59543839acba298a199770e69b728a53d"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.7.2"

[[deps.MLStyle]]
git-tree-sha1 = "2041c1fd6833b3720d363c3ea8140bffaf86d9c4"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.12"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "DelimitedFiles", "FLoops", "FoldsThreads", "Random", "ShowCases", "Statistics", "StatsBase", "Transducers"]
git-tree-sha1 = "025a4295ace07f35244597a98b392170b959ff48"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.2.7"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "1a80840bcdb73de345230328d49767ab115be6f2"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.8"

[[deps.NNlibCUDA]]
deps = ["CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "e161b835c6aa9e2339c1e72c3d4e39891eac7a4f"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.3"

[[deps.NPZ]]
deps = ["Compat", "FileIO", "ZipFile"]
git-tree-sha1 = "45f77b87cb9ed5b519f31e1590258930f3b840ee"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.2"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "ec2e30596282d722f018ae784b7f44f3b88065e4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.6"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9a36165cf84cff35851809a40a928e1103702013"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.16+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "7a28efc8e34d5df89fc87343318b0a8add2c4021"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "013596dcee5e55eb36ff56b8d4df888df01e040d"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.6"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pickle]]
deps = ["DataStructures", "InternedStrings", "Serialization", "SparseArrays", "Strided", "StringEncodings", "ZipFile"]
git-tree-sha1 = "8e4ba4cb57bedd0289865c65ffedeee910d6a8b6"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "afeacaecf4ed1649555a19cb2cad3c141bbc9474"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.5.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "a9e798cae4867e3a41cae2dd9eb60c047f1212db"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.6"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "11f1b69a28b6e4ca1cc18342bfab7adb7ff3a090"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.7.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "9f8a5dc5944dc7fbbe6eb4180660935653b0a9d9"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.0"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6edcea211d224fa551ec8a85debdc6d732f155dc"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2c11d7290036fe7aac9038ff312d3b3a2a5bf89e"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.4.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "642f08bf9ff9e39ccc7b710b2eb9a24971b52b1a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.17"

[[deps.Strided]]
deps = ["LinearAlgebra", "TupleTools"]
git-tree-sha1 = "632001471b256ceac6912c3e64d4b5c65154b216"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "1.2.2"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "50ccd5ddb00d19392577902f0079267a72c5ab04"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.5"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "d24a825a95a6d98c385001212dc9020d609f2d4f"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.8.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "464d64b2510a25e6efe410e7edab14fffdc333df"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.20"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a49267a2e5f113c7afe93843deea7461c0f6b206"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.40"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─162622ba-dcf8-4e81-9726-da2ffc311ccc
# ╟─2bf3f4a0-2e1d-4710-b938-7c680b2ea795
# ╟─b5a66c84-f3e2-11ec-397f-63f315fee830
# ╟─bb4fcd36-c83a-41b5-ab71-87271631482e
# ╟─69792a6e-e520-40be-afcb-0027b3d5f39a
# ╟─748cd332-07fe-4095-8138-f7bbf8634506
# ╟─bbd3d94e-c8d1-4cdd-8330-9b4003792f5a
# ╟─3fd3f317-d292-4572-a335-75ca91bc7793
# ╟─c698281c-23b1-48c1-90cc-0b844bedd367
# ╟─76a1e2d9-58a5-4761-88de-f793733b97b7
# ╟─c15b83ea-a4e0-440e-bab0-a61c9ee478f7
# ╟─2a8cd318-2c70-463e-9ecc-053c076ca5bc
# ╟─eb864c8e-e8db-4863-b0dc-a258c5de65b6
# ╟─da773c77-db69-4595-bab2-2f6ac3653645
# ╠═9aadd03d-619a-4628-b50e-9f3eb4d60e5d
# ╟─ae06dfcd-7123-4286-a9cb-01cf2597dbc3
# ╟─3264e605-d411-4681-87df-2e8d4517dd23
# ╠═fc940c6c-5517-4eb2-b724-99960f6b9e93
# ╠═c78facd5-d78f-4f7a-9e67-75fa1983b407
# ╟─15e9045f-cf61-4c9c-9ea2-e0bf4a63e9e8
# ╠═9d629ee8-6530-4216-9083-a8fc558f8fb7
# ╟─c8b10d6e-ba40-4cb2-8c20-ff70f1322b3d
# ╠═8359f15b-0a3c-468f-8494-8715062f811a
# ╟─2e444ede-6db6-4b63-84aa-21b399ce696b
# ╟─3a06f2ab-b917-4657-8000-c29b8224d027
# ╠═0601375e-7151-4968-a126-c79b5efa30be
# ╟─3c856bcf-29f9-4753-a2bb-985580fccc89
# ╠═d37298c8-ba86-41ec-9b7a-c00e2177b464
# ╠═7957e09a-ecad-4cc2-9a89-5c4c87923c25
# ╟─36d1685c-ca88-4665-94a4-4393987adb5d
# ╟─1dc7bb61-c071-420f-bd31-aa98de06faa7
# ╟─deddd641-d0c4-4635-9dea-e8055675386b
# ╟─c5d821e1-7d8c-4118-8991-2c0138161a86
# ╟─a2d0f440-152a-44ac-913d-9485c085908d
# ╟─508ace1e-afc2-4482-91f2-9d308dc5c5f6
# ╟─f74c8718-8061-4071-a6c0-3879c47d9960
# ╟─5cb12a32-8636-4e1d-9fcb-b823b5d6fad5
# ╟─108516ca-5a64-4aea-a078-437eb55f7f28
# ╟─e4377d81-593f-4ea3-8364-20a65a1b1275
# ╠═639a1bb0-1a29-48ab-b35c-e56b3f57b52c
# ╠═f77ab122-3aaf-47b1-ae46-edf17c20fc5c
# ╟─8d8671ec-86e9-407b-88dc-710258398a86
# ╠═a9cea07e-948c-43be-b3dd-dae29a78aeb5
# ╠═36ec1f9c-8038-46eb-aaee-747d4c97bb18
# ╟─ba997393-48dc-4277-86d7-2d62155d6b8d
# ╠═76078acb-bdc9-4122-99a7-0169138aef99
# ╟─17b82aa8-39a9-4c4e-a41f-5dfd3bf9ce3a
# ╠═38960a16-c49f-49f7-9cf7-319bc3e4e272
# ╠═701a1065-23f5-4ba9-a565-67eaccd74c2c
# ╟─8a85ffa5-2de4-4b8c-a322-5b68d84f7f17
# ╠═40ca36f8-ce8e-47cb-9a93-9a585bb815ab
# ╠═580f758f-f443-4ddb-a040-1aafd9d4b758
# ╟─fffb27d2-a354-4b50-b0ae-cec66d0c2a39
# ╠═c1e02298-0e2c-41e8-8ddf-ce1d4647859d
# ╠═2d681f34-18b7-44a4-bd35-da8899a556ad
# ╟─b9280b33-8da5-4d17-9124-28b10c42aa43
# ╟─2477e412-89cc-4970-99bd-ee0c58b5052c
# ╠═a02fe1c6-ec7f-450a-a125-2da5f31c1611
# ╠═10d48c95-3c85-475c-b700-5e87c9aea22d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
