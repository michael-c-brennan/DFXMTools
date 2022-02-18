
using Pkg
Pkg.activate(".")
using Optim
using ForwardDiff
using JLD
using LineSearches
using AdvancedMH
using AdaptiveMCMC
using MCMCChains

using Distributions
using StatsBase
using Base.Threads
nthreads()
using Printf
using Plots
using Plots.PlotMeasures
using LaTeXStrings
using Distributions
using StatsPlots



@time begin

#########################################################################################################################
# User inputs 
#########################################################################################################################

# Goniometer settings (must be set before including set_up.jl file as they enter into model definitions)
TwoDeltaTheta = 0
phi = 2.5E-3
chi = 0.015*pi/180

include("set_up.jl")

# Define lattice vector
miller_vec = [0;0;1]

# Define bounds of prior prior_bounds[char,:] = (width of prior, height of prior) in nm in the lab frame
prior_bounds = [800 800; 500 1000; 500 1000; 800 800; 
                1500 1500; 1500 1500; 1000 1500; 1000 1500; 
                500 1200; 1000 1000; 1000 1000; 500 1200
]


# Set character of dislocation
char = 2  #parse(Int64,ARGS[1])
dis_lab = get_dis_lab(char, phi, chi)

# Set noise parameters
background_percent = 0.05 #parse(Float64,ARGS[2])  # magnitude of background noise as percentage of image intensity
noise_level = 0.01 #parse(Float64,ARGS[3])  # magnitude of electic readout noise as percentage of image intensity

# Set prior center in image frame
loc_center = [0.;0.]

# Define the prior based on the character
lower = loc_center - prior_bounds[char,:]/2
upper = loc_center + prior_bounds[char,:]/2
prior = Product(Uniform.(lower, upper))

# Define name for result files (plot generation at bottom of script)
save_str = "example_data//inference_example_char_$(char)_background_$(background_percent)_noise_$(noise_level)"
save_str = replace(save_str, "."=>"p")

# Grid sizes for computing the log-posterior contour plots
gridsize_1 = 50 
gridsize_2 = 5 

#########################################################################################################################
# Generate a noisy image to act as an evaluation image
#########################################################################################################################

# Draw the true location from the prior and transform it to the dislocation frame
true_loc_im = rand(prior)
true_loc_dis = image_to_dis(true_loc_im, dis_lab)

# Compute the noise-free image
im_true = compute_image(char, true_loc_dis, phi, chi, TwoDeltaTheta, miller_vec)


# Define noise model based on noise parameters
noise_std = noise_level*maximum(im_true)
noise_var = noise_std^2

background = background_percent*maximum(im_true)

pixel_var = im_true .+ background .+ noise_var

# Compute and save noisy evaluation image
im_obs = im_true .+ background + sqrt.(pixel_var).*randn(Npixels,Npixels)
save(save_str*"_im_obs.jld","im_obs", im_obs)


#########################################################################################################################
# MAP optimization and Laplace approximation
#########################################################################################################################

# Define negative log-likelihood function of dislocation position (in dislocation frame)
function nlogl(locs)

        -1*compute_loglikelihood(im_obs, noise_var, char, locs, phi, chi, TwoDeltaTheta, miller_vec, background)
 
end


# Define (unnormalized) log posterior through log-likelihood (the prior is uniform and so does not contribution)
logpi(locs) = -nlogl(locs)


# optimization through Optim routine
loc_opt = optimize(nlogl,
                   loc_center,
                   GradientDescent(linesearch=LineSearches.BackTracking(order=2)),
                   Optim.Options(extended_trace=true, 
                                        store_trace=true,
                                        allow_f_increases=true,
                                        iterations=50,
                                        show_trace=true,
                                        g_abstol = 1E-5,
                                        x_abstol = 0.01)
)


# Unpack optimization results we want to save
num_iter = size(hcat(Optim.f_trace(loc_opt)...),2)
map_point = Optim.minimizer(loc_opt)
f_trace = hcat(Optim.f_trace(loc_opt)...)
loc_trace = hcat(Optim.x_trace(loc_opt)...)
runtime = Optim.time_run(loc_opt)

# Compute the Laplace approximation covariance (the negative inverse hessian of the posterior log-density at the MAP point)
map_cov = -inv(ForwardDiff.hessian(logpi, map_point))

# Save the data
save(save_str*"_opt_data.jld","true_loc_dis", true_loc_dis, "loc_trace", loc_trace, "f_trace", f_trace, "map_point", map_point, "num_iter", num_iter, "runtime", runtime, "map_cov", map_cov)



#########################################################################################################################
# MCMC sampling
#########################################################################################################################

# Define a distribution for drawing chain initialization points centered at the MAP point (alt. one could start chains at the MAP point)
mcmc_init = Product([Uniform(map_point[1] - 5, map_point[1] + 5),Uniform(map_point[2] - 5, map_point[2] + 5)])


# Define number and length of chains to compute [1000 length chain takes my laptop ~7-8 minutes to run]
nchains = 1
n_samples = 100 
adaptive_rwhm_chains = zeros(n_samples, 2, nchains)
times = zeros(nchains)
Threads.@threads for i in 1:nchains
        θ_init = rand(mcmc_init)
        samples = @timed adaptive_rwm(θ_init, logpi, n_samples; algorithm=:asm, b=1)
        adaptive_rwhm_chains[:,:,i] = Array(samples.value.X)'
        times[i] = samples.time
end

save(save_str*"_mcmc.jld","chains", adaptive_rwhm_chains)

adaptive_rwhm_chains = load(save_str*"_mcmc.jld","chains")
all_chains = Chains(adaptive_rwhm_chains)
all_samples = Array(all_chains)

#########################################################################################################################
# Computing log-posterior contour for plotting
#########################################################################################################################


#####################################################################
# A course sampling of a region larger than the prior in image plane
#####################################################################

xs_im = 2*lower[1]:gridsize_1:2*upper[1]
ys_im = 2*lower[2]:gridsize_1:2*upper[2]
density_matrix_im = zeros(length(ys_im),length(xs_im))
dis_xyz = []
@threads for i = 1:length(ys_im)
        y = ys_im[i]

        for j = 1:length(xs_im)
                x = xs_im[j]
                loc = [x;y]

                append!(dis_xyz, image_to_dis(loc, dis_lab, dims=3))
                density_matrix_im[i,j] = nlogl(image_to_dis(loc,dis_lab))
        end
end

save(save_str*"_mat_im.jld","mat", density_matrix_im, "xs_im", xs_im, "ys_im", ys_im)

dis_xyz = reshape(dis_xyz, (3, Int(length(dis_xyz)/3)))

xs_dis = minimum(dis_xyz[1,:]):gridsize_1:maximum(dis_xyz[1,:])
ys_dis = minimum(dis_xyz[2,:]):gridsize_1:maximum(dis_xyz[2,:])

density_matrix_dis = zeros(length(ys_dis),length(xs_dis))
@threads for i = 1:length(ys_dis)
        y = ys_dis[i]

        for j = 1:length(xs_dis)
                x = xs_dis[j]
                loc = [x;y]

                density_matrix_dis[i,j] = nlogl(loc)
        end
end

save(save_str*"_mat_dis.jld","mat", density_matrix_dis, "xs_dis", xs_dis, "ys_dis", ys_dis)
#########################################################################
# Refined density data near the MAP point in the dislocation frame
#########################################################################

xs_0 = minimum(all_samples[:,1])
xs_0 = minimum([xs_0, true_loc_dis[1]]) - 10
xs_1 = maximum(all_samples[:,1])
xs_1 = maximum([xs_1, true_loc_dis[1]]) + 10
xs_mid = 0.5*(xs_1 + xs_0)
xs_length = xs_1 - xs_0

ys_0 = minimum(all_samples[:,2])
ys_0 = minimum([ys_0, true_loc_dis[2]]) - 10
ys_1 = maximum(all_samples[:,2])
ys_1 = maximum([ys_1, true_loc_dis[2]]) + 10
ys_mid = 0.5*(ys_1 + ys_0)
ys_length = ys_1 - ys_0
max_length = maximum([xs_length,ys_length])

xs_f_dis = xs_mid - max_length/2:gridsize_2:xs_mid + max_length/2
ys_f_dis = ys_mid - max_length/2:gridsize_2:ys_mid + max_length/2
density_matrix_f_dis = zeros(length(ys_f_dis),length(xs_f_dis))
@threads for i = 1:length(ys_f_dis)
        y = ys_f_dis[i]

        for j = 1:length(xs_f_dis)
                x = xs_f_dis[j]
                loc = [x;y]
                density_matrix_f_dis[i,j] = nlogl(loc)
        end
end

save(save_str*"_mat_f_dis.jld","mat", density_matrix_f_dis, "xs_f_dis", xs_f_dis, "ys_f_dis", ys_f_dis)

##############################################################
# Refined density data near the MAP point in the image plane
##############################################################

all_samples_im = reshape(dis_to_image(all_samples',dis_lab),size(all_samples'))'

xs_0 = minimum(all_samples_im[:,1])
xs_0 = minimum([xs_0, true_loc_im[1]]) - 10
xs_1 = maximum(all_samples_im[:,1])
xs_1 = maximum([xs_1, true_loc_im[1]]) + 10
xs_mid = 0.5*(xs_1 + xs_0)
xs_length = xs_1 - xs_0

ys_0 = minimum(all_samples_im[:,2])
ys_0 = minimum([ys_0, true_loc_im[2]]) - 10
ys_1 = maximum(all_samples_im[:,2])
ys_1 = maximum([ys_1, true_loc_im[2]]) + 10
ys_mid = 0.5*(ys_1 + ys_0)
ys_length = ys_1 - ys_0
max_length = maximum([xs_length,ys_length])

xs_f_im = xs_mid - max_length/2:gridsize_2:xs_mid + max_length/2
ys_f_im = ys_mid - max_length/2:gridsize_2:ys_mid + max_length/2
density_matrix_f_im = zeros(length(ys_f_im),length(xs_f_im))
@threads for i = 1:length(ys_f_im)
        y = ys_f_im[i]

        for j = 1:length(xs_f_im)
                x = xs_f_im[j]
                loc = [x;y]
                density_matrix_f_im[i,j] = nlogl(image_to_dis(loc,dis_lab))
        end
end

save(save_str*"_mat_f_im.jld","mat", density_matrix_f_im, "xs_f_im", xs_f_im, "ys_f_im", ys_f_im)

# ############################################################################################################################
# # Plotting
# ############################################################################################################################

include("plotting.jl")
load_name = save_str

figure_dir = "example_figures"


# Load data 
im_obs = load(load_name*"_im_obs.jld","im_obs")
d = load(load_name*"_opt_data.jld")
true_loc_dis = d["true_loc_dis"]
true_loc_im = dis_to_image(true_loc_dis, dis_lab)
loc_trace_dis =d["loc_trace"]
loc_trace_im = reshape(dis_to_image(loc_trace_dis,dis_lab),size(loc_trace_dis))
f_trace =d["f_trace"]
map_point_dis =d["map_point"]
map_point_im = dis_to_image(map_point_dis,dis_lab)
num_iter = d["num_iter"]
runtime = d["runtime"]
map_cov = d["map_cov"]

adaptive_rwhm_chains = load(load_name*"_mcmc.jld","chains")
all_chains = Chains(adaptive_rwhm_chains)
all_samples = Array(all_chains)
all_samples_im = reshape(dis_to_image(all_samples',dis_lab),size(all_samples'))'

d = load(load_name*"_mat_im.jld")
xs_im = d["xs_im"]
ys_im = d["ys_im"]
density_matrix_im = d["mat"]

d = load(load_name*"_mat_dis.jld")
xs_dis = d["xs_dis"]
ys_dis = d["ys_dis"]
density_matrix_dis = d["mat"]

d = load(load_name*"_mat_f_dis.jld")
xs_f_dis = d["xs_f_dis"]
ys_f_dis = d["ys_f_dis"]
density_matrix_f_dis = d["mat"]

d = load(load_name*"_mat_f_im.jld")
xs_f_im = d["xs_f_im"]
ys_f_im = d["ys_f_im"]
density_matrix_f_im = d["mat"]

# Plot evaluation image
dfxm_image_plot_base(im_obs, char, true_loc_dis, figure_dir*"\\obs", prior=true, width=500)


# Figure 5c
contourf(xs_im, ys_im, density_matrix_im, levels=50, aspect_ratio=:equal, color=:reds)
rect = prior_bounds[char,:]
plot!(rectangle(rect...), opacity=.5, c=:gray, label=false, legend=:bottomright)
for t = 1:num_iter - 1
        plot!([loc_trace_im[1,t], loc_trace_im[1,t+1]], [loc_trace_im[2,t], loc_trace_im[2,t+1]],label=false,linewidth=3,c=:black)
end
scatter!(loc_trace_im[1,:],loc_trace_im[2,:], c=:orange, markershape = :utriangle, markersize = 4, label="iterates")
scatter!([map_point_im[1]],[map_point_im[2]],c=:lightgreen, markershape = :star5, markersize = 8, label=L"\xi^{\mathrm{MAP}}")
scatter!([true_loc_im[1]],[true_loc_im[2]], c=:cyan, markershape = :star7, markersize = 8, label=L"\xi^{\mathrm{true}}", legend=false)# outertop)
xlims!(-500,500)
ylims!(-600,600)
xticks!([-500,0,500])
yticks!([-500,0,500])
xlabel!(L"y_{\ell} \ (\mathrm{nm})", xtickfontsize=12, xguidefontsize=18, margin=5mm)
ylabel!(L"x_{\ell} \ (\mathrm{nm})", ytickfontsize=12, yguidefontsize=18)
plot!(size=(450,450),right_margin=15mm)
Plots.savefig(figure_dir*"\\iters.pdf")


# Figure 5b
prior_corners = [250; 500; -250; 500; -250; -500; 250; -500]
prior_corners_dis = reshape(image_to_dis(prior_corners, dis_lab), (2,4))
s_prior = Shape(prior_corners_dis[1,:],prior_corners_dis[2,:])

contourf(xs_dis, ys_dis, density_matrix_dis, levels=10, aspect_ratio=:equal, color=cgrad(:grays,rev=true))
plot!(s_prior, opacity=.5, c=:gray, linecolor=:white, label=false)
#plot!(s_region, c=false, linecolor=:white, label=false)
for t = 1:num_iter - 1
        plot!([loc_trace_dis[1,t], loc_trace_dis[1,t+1]], [loc_trace_dis[2,t], loc_trace_dis[2,t+1]],label=false,linewidth=3,c=:black)
end
scatter!(loc_trace_dis[1,:],loc_trace_dis[2,:], c=:orange, markershape = :utriangle, markersize = 4, label="iterates")
scatter!([map_point_dis[1]],[map_point_dis[2]],c=:lightgreen, markershape = :star5, markersize = 8, label=L"\xi^{\mathrm{MAP}}")
scatter!([true_loc_dis[1]],[true_loc_dis[2]], c=:cyan, markershape = :star7, markersize = 8, label=L"\xi^{\mathrm{true}}", legend=false)# outertop)
xlabel!(L"x_{d} \mathrm{ \ (nm)}", xtickfontsize=12, xguidefontsize=18, margin=5mm)
ylabel!(L"y_{d} \mathrm{ \ (nm)}", ytickfontsize=12, yguidefontsize=18)
xlims!(xs_dis[1],xs_dis[end])
ylims!(ys_dis[1],ys_dis[end])
xticks!([-500,0,500])
yticks!([-400,0,400])
plot!(size=(450,450),right_margin=15mm)
Plots.savefig(figure_dir*"\\iters_dis.pdf")


# Figure 6a
contourf(xs_f_dis,ys_f_dis,density_matrix_f_dis, levels=50, aspect_ratio=:equal, color=cgrad(:grays,rev=true))
scatter!(all_samples[:,1],all_samples[:,2],c=:plum1, alpha=.1, label="MCMC samples")
covellipse!(vec(map_point_dis), map_cov, n_std=2, aspect_ratio=1, color=(:yellow), label=false, alpha=.5)
scatter!([map_point_dis[1]],[map_point_dis[2]],c=:lightgreen, markershape = :star5, markersize = 8, label=L"\xi^{\mathrm{MAP}}")
scatter!([true_loc_dis[1]],[true_loc_dis[2]], c=:cyan, markershape = :star7, markersize = 8, label=L"\xi^{\mathrm{true}}", legend=false)# outertop)
xlabel!(L"x_{d} \mathrm{ \ (nm)}", xtickfontsize=14, xguidefontsize=18, margin=5mm)
ylabel!(L"y_{d} \mathrm{ \ (nm)}", ytickfontsize=14, yguidefontsize=18)
plot!(size=(450,450), right_margin=15mm)
Plots.savefig(figure_dir*"\\uq_dis.pdf")

# Figure 6b
vals,vecs = eigen(map_cov)
vecs_im = reshape(dis_to_image(vecs,dis_lab),(2,2))
map_cov_im = vecs_im*diagm(vals)*vecs_im'

contourf(xs_f_im,ys_f_im,density_matrix_f_im, levels=50,aspect_ratio=:equal, color=:reds)
scatter!(all_samples_im[:,1],all_samples_im[:,2], alpha=.2, c=:plum1, label="MCMC samples")
covellipse!(vec(map_point_im), map_cov_im, n_std=2, aspect_ratio=1, color=(:yellow), label=false, alpha=.5)
scatter!([map_point_im[1]],[map_point_im[2]],c=:lightgreen, markershape = :star5, markersize = 8, label=L"\xi^{\mathrm{MAP}}")
scatter!([true_loc_im[1]],[true_loc_im[2]], c=:cyan, markershape = :star7, markersize = 8, label=L"\xi^{\mathrm{true}}", legend=false)
xlabel!(L"y_{\ell} \mathrm{ \ (nm)}", xtickfontsize=14, xguidefontsize=18, margin=5mm)
ylabel!(L"x_{\ell} \mathrm{ \ (nm)}", ytickfontsize=14, yguidefontsize=18)
plot!(size=(450,450), right_margin=15mm)
Plots.savefig(figure_dir*"\\uq_im.pdf")

end