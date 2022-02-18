
using Pkg
Pkg.activate(".")

using Plots
using Plots.PlotMeasures
using LaTeXStrings
using Printf
using JLD
using StatsPlots
using Base.Threads
nthreads()
using AdvancedMH
using AdaptiveMCMC
using MCMCChains
using Distributions

begin

include("plotting.jl")

#########################################################################################################################

TwoDeltaTheta = 0
phi = 2.5E-3
chi = 0.015*pi/180
include("set_up.jl")

# Define lattice vector
miller_vec = [0;0;1]


dir_str = "paper_figures\\"


# ================================================================
# Figure 2
# ================================================================
char = 2


# Find coordinates in dislocation frame for a 500nm off-set in image plane 
bnt_sample, bnt_im, bnt_lab = get_bnt(phi, chi, TwoDeltaTheta, char)
dis_lab = get_dis_lab(char, phi, chi)

loc_d = [1;0]
loc_d_3 = -(dis_lab[1:2,3]'*loc_d)/dis_lab[3,3]

loc_lab = dis_lab'*[loc_d;loc_d_3]
v = [loc_lab[2]; loc_lab[1]/tan(2*theta)]
v = v/norm(v)

image_loc = 500*v

loc_lab = [image_loc[2]*tan(2*theta); image_loc[1]]

loc_d = dis_lab*[loc_lab;0]

# Figure 2a
im_true = compute_image(char, [0; 0], phi, chi, TwoDeltaTheta, [0;0;1])
dfxm_image_plot_base(im_true, char, [0; 0], dir_str*"base_char_2")

# Figure 2b
loc_center = [0; 0; 8*loc_d[1]; 0; -8*loc_d[1]; 0 ]
im_3 = compute_image(char, loc_center, phi, chi, TwoDeltaTheta, [0;0;1])
dfxm_image_plot_base(im_3, char, loc_center, dir_str*"base_char_2_3")

# Figure 2c
loc_center = [0; 0; 4*loc_d[1]; 0; -4*loc_d[1]; 0 ; 8*loc_d[1]; 0; -8*loc_d[1]; 0 ]
im_5 = compute_image(char, loc_center, phi, chi, TwoDeltaTheta, [0;0;1])
dfxm_image_plot_base(im_5, char, loc_center, dir_str*"base_char_2_5")

# Figure 2d
loc_center = [0; 0; 3*loc_d[1]; 0; -3*loc_d[1]; 0 ; 6*loc_d[1]; 0; -6*loc_d[1]; 0 ;9*loc_d[1]; 0; -9*loc_d[1]; 0 ]
im_7 = compute_image(char, loc_center, phi, chi, TwoDeltaTheta, [0;0;1])
dfxm_image_plot_base(im_7, char, loc_center, dir_str*"base_char_2_7")




# ================================================================
# Figure 3
# ================================================================
char = 6
im_true = compute_image(char, [0;0], phi, chi, TwoDeltaTheta, [0;0;1])

dfxm_image_plot_base(im_true, char, [0; 0], dir_str*"base_char_6")

noise_level = 0.01
background_percent = 0.05 

noise_std = noise_level*maximum(im_true)
noise_var = noise_std^2
background = background_percent*maximum(im_true)
pixel_var = im_true .+ background .+ noise_var
im_obs = im_true .+ background + sqrt.(pixel_var).*randn(Npixels,Npixels)

# Example 1 at max
pv = findmax(im_true)[1]
idx_1 = findmax(im_true)[2]
mu = pv + background
sig = sqrt(mu + noise_var)
Ndist1 = Normal(mu,sig)
xs = LinRange(0.,500.,1000)


# Example 2 at edge of blob
idx_2 = [26,idx_1[2]]
pv = im_true[idx_2[1], idx_2[2]]
mu = pv + background
sig = sqrt(mu + noise_var)
Ndist2 = Normal(mu,sig)


plot(xs, pdf.(Ndist1,xs), label="pdf at pixel A", lw=4, ls=:dot, color=:blue, legend=:topright)
plot!(xs, pdf.(Ndist2,xs), label="pdf at pixel B", lw=4, ls=:solid, color=:orange)
xlabel!("pixel value", xtickfontsize=14, xguidefontsize=18, ytickfontsize=14, margin=5mm, legendfontsize=18)
plot!(size=(450,450))
savefig(dir_str*"likelihood_pixel_dists.pdf")


heatmap(y_im_vec, z_im_vec, im_obs, clim=(0,maximum(im_obs)), aspect_ratio=:equal, bbox=:tight)
scatter!([y_im_vec[idx_1[2]]], [z_im_vec[idx_1[1]]], marker=(4,:blue), markerfacecolor="None", label="pixel A")
scatter!([y_im_vec[idx_2[2]]], [z_im_vec[idx_2[1]]], marker=(4,:orange,:square), markerfacecolor="None", label="pixel B",legend=:bottomright)
xlims!(y_im_vec[1],y_im_vec[end])
ylims!(z_im_vec[1],z_im_vec[end])
xticks!([-1,0,1])
yticks!([-4,-2,0,2,4])
xlabel!(L"y_{\ell} \ (\mu \mathrm{m})", xtickfontsize=14, xguidefontsize=18, margin=5mm)
ylabel!(L"x_{\ell} \ (\mu \mathrm{m})", ytickfontsize=14, yguidefontsize=18)
plot!(size=(450,450))
Plots.savefig(dir_str*"likelihood_im_obs.pdf")


# ================================================================
# Figure 4
# ================================================================
for char = 1:12 

    # noise-free image
    im_true = compute_image(char, [0; 0], phi, chi, TwoDeltaTheta, [0;0;1])
    
    # save with support of prior
    save_str = dir_str*"prior_char_$(char)"
    

    dfxm_image_plot_base(im_true, char, [0.,0.], save_str; prior=true)
    

end




# character and test number
char = 2
test = 1

dis_lab = get_dis_lab(char, phi, chi)

load_name = "paper_data//density_examples_char_$(char)_background_$(background_percent)_noise_$(noise_level)_test_$(test)"
load_name = replace(load_name, "."=>"p") 




# Load data 
im_obs = load(load_name*"_im_obs.jld","im_obs")
d = load(load_name*"_opt_data.jld")
true_loc_dis = d["true_loc"]
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
dfxm_image_plot_base(im_obs, char, true_loc_dis, dir_str*"\\obs", prior=true, width=500)


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
Plots.savefig(dir_str*"\\iters.pdf")


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
Plots.savefig(dir_str*"\\iters_dis.pdf")


# Figure 6a
contourf(xs_f_dis,ys_f_dis,density_matrix_f_dis, levels=50, aspect_ratio=:equal,clims=(15600, 16100), color=cgrad(:grays,rev=true))
scatter!(all_samples[:,1],all_samples[:,2],c=:plum1, alpha=.1, label="MCMC samples")
covellipse!(vec(map_point_dis), map_cov, n_std=2, aspect_ratio=1, color=(:yellow), label=false, alpha=.5)
scatter!([map_point_dis[1]],[map_point_dis[2]],c=:lightgreen, markershape = :star5, markersize = 8, label=L"\xi^{\mathrm{MAP}}")
scatter!([true_loc_dis[1]],[true_loc_dis[2]], c=:cyan, markershape = :star7, markersize = 8, label=L"\xi^{\mathrm{true}}", legend=false)# outertop)
xlims!(200,220)
ylims!(30, 50)
xticks!([200,210,220])
yticks!([30,40,50])
xlabel!(L"x_{d} \mathrm{ \ (nm)}", xtickfontsize=14, xguidefontsize=18, margin=5mm)
ylabel!(L"y_{d} \mathrm{ \ (nm)}", ytickfontsize=14, yguidefontsize=18)
plot!(size=(450,450), right_margin=15mm)
Plots.savefig(dir_str*"\\uq_dis.pdf")

# Figure 6b
vals,vecs = eigen(map_cov)
vecs_im = reshape(dis_to_image(vecs,dis_lab),(2,2))
map_cov_im = vecs_im*diagm(vals)*vecs_im'

contourf(xs_f_im,ys_f_im,density_matrix_f_im, levels=50,aspect_ratio=:equal,clims=(15600, 16100), color=:reds)
scatter!(all_samples_im[:,1],all_samples_im[:,2], alpha=.2, c=:plum1, label="MCMC samples")
covellipse!(vec(map_point_im), map_cov_im, n_std=2, aspect_ratio=1, color=(:yellow), label=false, alpha=.5)
scatter!([map_point_im[1]],[map_point_im[2]],c=:lightgreen, markershape = :star5, markersize = 8, label=L"\xi^{\mathrm{MAP}}")
scatter!([true_loc_im[1]],[true_loc_im[2]], c=:cyan, markershape = :star7, markersize = 8, label=L"\xi^{\mathrm{true}}", legend=false)
xlims!(110,150)
ylims!(440,480)
xticks!([110,130,150])
yticks!([440,460,480])
xlabel!(L"y_{\ell} \mathrm{ \ (nm)}", xtickfontsize=14, xguidefontsize=18, margin=5mm)
ylabel!(L"x_{\ell} \mathrm{ \ (nm)}", ytickfontsize=14, yguidefontsize=18)
plot!(size=(450,450), right_margin=15mm)
Plots.savefig(dir_str*"\\uq_im.pdf")




# figure 7 low noise histogram
background_percent = 0.05
noise_level = 0.01

offset_vec_dis_norms = []
offset_vec_im_norms = []
runtimes = []


for char = 1:12
    dis_lab = get_dis_lab(char, phi, chi)
    load_str = "paper_data\\map_results\\map_data_custom_prior_char_$(char)_background_$(background_percent)_noise_$(noise_level)"
    load_str = replace(load_str, "."=>"p")
    
    d = load(load_str*".jld")
    
    finishes_dis = d["finishes"]
    finishes_im = reshape(dis_to_image(finishes_dis, dis_lab), size(finishes_dis))
    
    true_locs_dis = d["true_locs"]
    true_locs_im = reshape(dis_to_image(true_locs_dis, dis_lab), size(true_locs_dis))


    offset_vec_dis = finishes_dis - true_locs_dis
    append!(offset_vec_dis_norms, [norm(offset_vec_dis[1:2,i]) for i=1:length(offset_vec_dis[1,:])])


    offset_vec_im = finishes_im - true_locs_im
    append!(offset_vec_im_norms, [norm(offset_vec_im[1:2,i]) for i=1:length(offset_vec_im[1,:])])

    append!(runtimes, d["runtimes"])

end

# =====================================================


mean_error = mean(offset_vec_dis_norms)
median_error = median(offset_vec_dis_norms)


histogram(offset_vec_dis_norms, nbins=100, label=false, alpha=.8, color=:gray, xtickfontsize=12,
ytickfontsize=12,xguidefontsize=16)

vline!([median_error], label="median error: $(@sprintf("%.1F",median_error)) nm", color=:blue, linewidth=4, linestyle=:solid)
vline!([mean_error], label="mean error: $(@sprintf("%.1F",mean_error)) nm", color=:green, linewidth=4, linestyle=:dash)
vline!([psize], label="pixel length: $(@sprintf("%.1F",psize)) nm", color=:red, linewidth=4, linestyle=:dot)
vline!([2*psize], label="image res: $(@sprintf("%.1F",2*psize)) nm", color=:black, linewidth=4, linestyle=:dashdotdot)

xlabel!("Estimation error (nm)")
xlims!(0,2*psize+10, legendfontsize=16)
save_str = dir_str*"\\error_histo_dis_background_$(background_percent)_noise_$(noise_level)"
save_str = replace(save_str, "."=>"p")
savefig(save_str*".pdf")


# figure 7 high noise histogram
background_percent = 0.1
noise_level = 0.05

offset_vec_dis_norms = []
offset_vec_im_norms = []
runtimes = []


for char = 1:12
    dis_lab = get_dis_lab(char, phi, chi)
    load_str = "paper_data\\map_results\\map_data_custom_prior_char_$(char)_background_$(background_percent)_noise_$(noise_level)"
    load_str = replace(load_str, "."=>"p")
    
    d = load(load_str*".jld")
    
    finishes_dis = d["finishes"]
    finishes_im = reshape(dis_to_image(finishes_dis, dis_lab), size(finishes_dis))
    
    true_locs_dis = d["true_locs"]
    true_locs_im = reshape(dis_to_image(true_locs_dis, dis_lab), size(true_locs_dis))


    offset_vec_dis = finishes_dis - true_locs_dis
    append!(offset_vec_dis_norms, [norm(offset_vec_dis[1:2,i]) for i=1:length(offset_vec_dis[1,:])])


    offset_vec_im = finishes_im - true_locs_im
    append!(offset_vec_im_norms, [norm(offset_vec_im[1:2,i]) for i=1:length(offset_vec_im[1,:])])

    append!(runtimes, d["runtimes"])

end

mean_error = mean(offset_vec_dis_norms)
median_error = median(offset_vec_dis_norms)

histogram(offset_vec_dis_norms, nbins=100, label=false, alpha=.8, color=:gray, xtickfontsize=12,
ytickfontsize=12,xguidefontsize=16)

vline!([median_error], label="median error: $(@sprintf("%.1F",median_error)) nm", color=:blue, linewidth=4, linestyle=:solid)
vline!([mean_error], label="mean error: $(@sprintf("%.1F",mean_error)) nm", color=:green, linewidth=4, linestyle=:dash)
vline!([psize], label="pixel length: $(@sprintf("%.1F",psize)) nm", color=:red, linewidth=4, linestyle=:dot)
vline!([2*psize], label="image res: $(@sprintf("%.1F",2*psize)) nm", color=:black, linewidth=4, linestyle=:dashdotdot)

xlabel!("Estimation error (nm)")
xlims!(0,2*psize+10, legendfontsize=16)
save_str = dir_str*"\\error_histo_dis_background_$(background_percent)_noise_$(noise_level)"
save_str = replace(save_str, "."=>"p")
savefig(save_str*".pdf")
end