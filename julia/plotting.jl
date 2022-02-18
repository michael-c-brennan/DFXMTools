function rectangle(w, h) 
    x = -w/2
    y = -h/2
    return Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
end

prior_bounds = [800 800; 500 1000; 500 1000; 800 800; 
                1500 1500; 1500 1500; 1000 1500; 1000 1500; 
                500 1200; 1000 1000; 1000 1000; 500 1200
]

function add_prior_region!(prior_bounds;label=true)

    rect = prior_bounds*1E-3
    if label==true
        plot!(rectangle(rect...), opacity=.5, c=:gray, linecolor=:white, label="Prior", legend=:outertop)
    else
        plot!(rectangle(rect...), opacity=.5, c=:gray, linecolor=:white, label=false, legend=:outertop)
    end

end


function add_trueloc!(true_loc, label, char)
    
    dis_lab = get_dis_lab(char, phi, chi)
    true_loc_im = dis_to_image(true_loc, dis_lab)
    num_dislocs = Int(length(true_loc)/2)
    for i = 1:num_dislocs
        scatter!([true_loc_im[2*i-1]*1E-3],[true_loc_im[2*i]*1E-3], c=:cyan, markersize = 8, markershape = :star7, legend=:outertop, legendfontsize=18, label=label)
    end

end


function dfxm_image_plot_base(im, char, true_loc, save_str; width=450, true_loc_label = false, tickfontsize=16, prior=false, prior_label=false)

    heatmap(y_im_vec, z_im_vec, im, clim=(0,maximum(im)), aspect_ratio=:equal, bbox=:tight)

    if prior 
        add_prior_region!(prior_bounds[char,:], label=prior_label)
    end
    
    add_trueloc!(true_loc,  true_loc_label, char)


    xlims!(y_im_vec[1],y_im_vec[end])
    ylims!(z_im_vec[1],z_im_vec[end])

    xticks!([-1,0,1])
    yticks!([-4,-2,0,2,4])
    
    xlabel!(L"y_{\ell} \ (\mu \mathrm{m})", xtickfontsize=tickfontsize, xguidefontsize=18, legendfontsize=18, margin=5mm)
    ylabel!(L"x_{\ell} \ (\mu \mathrm{m})", ytickfontsize=tickfontsize, yguidefontsize=18)
    
    plot!(size=(width,width))
    Plots.savefig(save_str*".pdf")

end
