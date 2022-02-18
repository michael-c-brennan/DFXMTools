using StatsBase
using LinearAlgebra
using QuadGK
using KernelDensity

# DFXM set-up 
# citations
# https://arxiv.org/pdf/2007.09475.pdf [1]
# https://arxiv.org/ftp/arxiv/papers/2009/2009.05083.pdf [2]


###############################################
# DEFINING THE INSTRUMENTAL RESOLUTION FUNCTION
###############################################

# Parameters
zeta_v_rms = 0.03*(pi/180)/2.355#; % [2-sup, pg 2] incoming divergence in vertical direction in rad (about 0.523 mrad)
zeta_h_rms = 1E-5/2.355; # [Henning's code] incoming divergence in horizontal direction, in rad 
NA_rms = 7.2E-4; # [2-sup, pg 2] NA of objective, in rad 
eps_rms = 1.4E-4/2.355; #[1, pg ] rms width of x-ray energy bandwidth (2.355 factor fwhm -> rms)
two_theta = 20.73 # [2-sup, pg 2] 2θ in degrees
D = 435; # [2-sup, pg 2] physical aperture of objective, in m
d1 = 0.274; # [2-sup, pg 2] sample-objective distance, in m

phys_aper = D/d1; 
theta = two_theta/2*pi/180; # theta in rad

# Monte Carlo simulation to sample delta_2theta and xi
Nrays = Int(1E5)
delta_2theta = zeros(Nrays)
xi = zeros(Nrays)
for k=1:Nrays

    temp = phys_aper/2 + 1
    while abs(temp) > phys_aper/2
        temp = randn()*NA_rms
    end
    delta_2theta[k] = temp

    temp = phys_aper/2 + 1
    while abs(temp) > phys_aper/2
        temp = randn()*NA_rms
    end
    xi[k] = temp

end

# compute zeta_v and zeta_h data
zeta_v = (rand(Nrays).-0.5).*zeta_v_rms*2.355
zeta_h = randn(Nrays).*zeta_v_rms
eps = randn(Nrays)*eps_rms

# compute q_{rock,roll,par} 
qrock = -zeta_v/2 - delta_2theta/2 .+ phi;
qroll = -zeta_h/(2*sin(theta)) -  xi/(2*sin(theta));
qpar = eps + cot(theta)*(-zeta_v/2 + delta_2theta/2)
qrock_prime = cos(theta).*qrock + sin(theta).*qpar

# Fit MC data and form kernel density approximation
hist = fit(Histogram,qrock_prime,nbins=10)
hist = normalize(hist,mode=:probability)
bin_max = maximum(hist.weights)
k = kde(qrock_prime)
ik = InterpKDE(k)

# Define resolution function as the normalized pdf 
resq_fn(q) = pdf(ik, q)
res_max = maximum(resq_fn(-5E-3:1E-6:1E-1))
resq_fn(q) = pdf(ik, q)/res_max


#Define FOV
Npixels = 50; # number of pixels across detector (same in both y and z) - sets the FOV.
              # even s0 that 0 is between at the corner of 4 pixels
psize = 75; # [1 pg 13, 2-sup pg 2] pixel size in units of nm, in the object plane
image_bound = Npixels*psize # for 50*75 = 3750 nm or 3.75 microns)

# experimental/physical constants
zl_fwhm = 600   # fwhm value of Gaussian beam profile
zl_rms = zl_fwhm/2.355

mu = theta 
ny = 0.334
b = 0.286 # length of Burgers vector 0.286 nm for Al 0.248 nm for Fe
bfactor = b/(4*pi*(1-ny));

# define quadrature nodes and weights for computing integral against Gaussian beam profile in z_lab direction
nodes,weights = gauss( z->exp(-0.5*(z/zl_rms)^2), 10, -.5*zl_rms, .5*zl_rms)

# 12 dislocations characters encoded in 12 choices for the grain->dislocation coordinate transformation
dis_grain_all = zeros(3,3,12)
dis_grain_all[:,:,1] = [1/sqrt(2) 1/sqrt(2) 0; -1/sqrt(3) 1/sqrt(3) 1/sqrt(3); 1/sqrt(6) -1/sqrt(6) 2/sqrt(6) ]
dis_grain_all[:,:,2] = [1/sqrt(2) 1/sqrt(2) 0; 1/sqrt(3) -1/sqrt(3) 1/sqrt(3); 1/sqrt(6) -1/sqrt(6) -2/sqrt(6) ]
dis_grain_all[:,:,3] = [1/sqrt(2) -1/sqrt(2) 0; -1/sqrt(3) -1/sqrt(3) -1/sqrt(3); 1/sqrt(6) 1/sqrt(6) -2/sqrt(6) ]
dis_grain_all[:,:,4] = [1/sqrt(2) -1/sqrt(2) 0; 1/sqrt(3) 1/sqrt(3) -1/sqrt(3); 1/sqrt(6) 1/sqrt(6) 2/sqrt(6) ]
dis_grain_all[:,:,5] = [1/sqrt(2) 0 -1/sqrt(2); 1/sqrt(3) 1/sqrt(3) 1/sqrt(3); 1/sqrt(6) -2/sqrt(6) 1/sqrt(6) ]
dis_grain_all[:,:,6] = [1/sqrt(2) 0 -1/sqrt(2); 1/sqrt(3) -1/sqrt(3) 1/sqrt(3); -1/sqrt(6) -2/sqrt(6) -1/sqrt(6) ]
dis_grain_all[:,:,7] = [1/sqrt(2) 0 1/sqrt(2); 1/sqrt(3) 1/sqrt(3) -1/sqrt(3); -1/sqrt(6) 2/sqrt(6) 1/sqrt(6) ]
dis_grain_all[:,:,8] = [1/sqrt(2) 0 1/sqrt(2); -1/sqrt(3) 1/sqrt(3) 1/sqrt(3); -1/sqrt(6) -2/sqrt(6) 1/sqrt(6) ]
dis_grain_all[:,:,9] = [0 1/sqrt(2) 1/sqrt(2); 1/sqrt(3) 1/sqrt(3) -1/sqrt(3); -2/sqrt(6) 1/sqrt(6) -1/sqrt(6) ]
dis_grain_all[:,:,10] = [0 1/sqrt(2) 1/sqrt(2); -1/sqrt(3) 1/sqrt(3) -1/sqrt(3); -2/sqrt(6) -1/sqrt(6) 1/sqrt(6) ]
dis_grain_all[:,:,11] = [0 1/sqrt(2) -1/sqrt(2); 1/sqrt(3) 1/sqrt(3) 1/sqrt(3); 2/sqrt(6) -1/sqrt(6) -1/sqrt(6) ]
dis_grain_all[:,:,12] = [0 1/sqrt(2) -1/sqrt(2); -1/sqrt(3) 1/sqrt(3) 1/sqrt(3); 2/sqrt(6) 1/sqrt(6) 1/sqrt(6) ]

# constant transformation matrices
sample_grain = I(3)
Omega = I(3)
Mu =  [cos(mu) 0 sin(mu); 0 1 0; -sin(mu) 0 cos(mu)]

# Define (square) voxel (stretched in image function)
y_lab_vec = -(Npixels-1)*psize/2:psize:(Npixels-1)*psize/2
x_lab_vec = -(Npixels-1)*psize/2:psize:(Npixels-1)*psize/2
z_lab_vec = -(Npixels-1)*psize/2:psize:(Npixels-1)*psize/2
Nz = length(z_lab_vec)

# transformed coordinates of image
z_im_vec = 1E-3*x_lab_vec/tan(2*theta)
y_im_vec = 1E-3*y_lab_vec


##############################################################
# SET UP FOR compute_image AND compute_loglikelihood FUNCTIONS
##############################################################

# returns coordinate change matrix 'dis_lab', such that R_{dis} = dis_lab*R_{lab}
function get_dis_lab(chars, phi, chi)

    # define rotation/change of frame matrices we'll need

    Chi = [1 0 0; 0 cos(chi) -sin(chi); 0 sin(chi) cos(chi)];
    Phi = [cos(phi) 0 -sin(phi); 0 1 0; sin(phi) 0 cos(phi)];

    lab_sample = Mu*Omega*Chi*Phi;
    grain_lab = (lab_sample*sample_grain)';
    char = chars[1]

    # define coordinate transforms that depend on dislocation character
    dis_grain = dis_grain_all[:,:,char]

    dis_lab = dis_grain*grain_lab;

    dis_lab
    
end

# returns Burgers, slip-plane normal and line vectors in sample, image, and lab frames
function get_bnt(phi, chi, TwoDeltaTheta, char)

    # define rotation/change of frame matrices we'll need
    theta_p = theta + TwoDeltaTheta;
    
    Theta = [cos(theta_p) 0 sin(theta_p); 0 1 0; -sin(theta_p) 0 cos(theta_p)];
    Theta2 = Theta*Theta;

    image_lab = Theta2;
    Chi = [1 0 0; 0 cos(chi) -sin(chi); 0 sin(chi) cos(chi)];
    Phi = [cos(phi) 0 -sin(phi); 0 1 0; sin(phi) 0 cos(phi)];

    lab_sample = Mu*Omega*Chi*Phi;
    image_sample = image_lab*lab_sample;

    dis_grain = dis_grain_all[:,:,char]

    # burgers, line, slip plane normal vectors
    bnt_sample = dis_grain'
    bnt_im = image_sample*bnt_sample
    bnt_lab = lab_sample*bnt_sample

    return bnt_sample, bnt_im, bnt_lab

end

# simulates an DFXM image for dislocations described by at locations in locs [real,real] of character char int[1-12]
# NOTE: we assume all dislocation characters are the same ( e.g., chars = [1;1;...], not chars = [1;2;...] )
image_scale_constant = 1.8  # constant so that images have realistic maximum intensities
function compute_image(chars, locs, phi, chi, TwoDeltaTheta, miller_vec)

    N_dislocs = Int(length(locs)/2)

    # define lattice vector
    Q_norm = norm(miller_vec) # We have assumed B_0 = I
    q_hkl = miller_vec/Q_norm

    # define rotation/change of frame matrices we'll need
    theta_p = theta + TwoDeltaTheta
    
    Theta = [cos(theta_p) 0 sin(theta_p); 0 1 0; -sin(theta_p) 0 cos(theta_p)]
    Theta2 = Theta*Theta

    image_lab = Theta2
    Chi = [1 0 0; 0 cos(chi) -sin(chi); 0 sin(chi) cos(chi)]
    Phi = [cos(phi) 0 -sin(phi); 0 1 0; sin(phi) 0 cos(phi)]

    lab_sample = Mu*Omega*Chi*Phi
    image_sample_x = image_lab[1,:]'*lab_sample
    grain_lab = (lab_sample*sample_grain)'
    char = chars[1] #

    # define coordinate transforms that depend on dislocation character
    dis_grain = dis_grain_all[:,:,char]

    dis_lab = dis_grain*grain_lab;
    dis_sample = dis_lab*lab_sample;

    image = zeros(Npixels,Npixels)
    for ii = 1:Npixels
        x_lab = x_lab_vec[ii]

        for jj = 1:Npixels
            y_lab = y_lab_vec[jj]
        
            # resolution function (the integrand)
            function res_fun(z_lab,locs)
                
                Fdis_sum = zeros(3,3)
                # Lab to dislocation coordinates
                for i = 1:N_dislocs
                    
                    # put out location and character of i-th dislocation
                    loc = locs[1+2*(i-1):2*i]
        
                    # lab -> dislocation frame
                    r_lab = [x_lab + z_lab/tan(2*theta_p); y_lab; z_lab]
        
                    r_dis = dis_lab[1:2,:]*r_lab;
                    xd = r_dis[1]-loc[1]
                    yd = r_dis[2]-loc[2];
        
                    # helper variables for strain computation
                    sqx = xd^2;
                    sqy = yd^2;
                    denom = (sqx + sqy)^2;
                    nyfactor = 2*ny*(sqx+sqy);
        
                    # Form strain tensor in dislocation frame
                    Fdis = [-yd*(3*sqx + sqy - nyfactor) xd*(3*sqx+sqy-nyfactor) 0;
                            -xd*(3*sqy+sqx-nyfactor) yd*(sqx-sqy-nyfactor) 0;
                            0 0 0];
                    Fdis_sum += bfactor*Fdis/denom;
        
                end

                Fdis_sum = Fdis_sum + I(3)
                
                # transform to sample frame
                Fsample = dis_sample'*Fdis_sum*dis_sample
                # Compute qi (qs -> qi)
                qs = (Fsample')\q_hkl - q_hkl;
                qs = qs + [phi - TwoDeltaTheta/2; chi; (TwoDeltaTheta/2)/tan(theta_p)]; # Eq 40 (also Eq. 20)
        
                # compute res(qi_1)
                resq_fn(image_sample_x*qs)
        
            end
        
            # compute integral
            image[ii,jj] = image_scale_constant*sum([res_fun(node, locs)*weight for (node,weight) in zip(nodes,weights)])

        end
    end

    return image

end

# computes the log-likelihood of dislocations described by locs,char for the observed imagine obs_im.
# NOTE: we assume all dislocation characters are the same ( e.g., chars = [1;1;...], not chars = [1;2;...] )
function compute_loglikelihood(obs_im, noise_var, chars, locs, phi, chi, TwoDeltaTheta, miller_vec, background)


    N_dislocs = Int(length(locs)/2)

    # define lattice vector
    Q_norm = norm(miller_vec); # We have assumed B_0 = I
    q_hkl = miller_vec/Q_norm;

    # define rotation/change of frame matrices we'll need
    theta_p = theta + TwoDeltaTheta;
    sample_grain = I(3);
    
    Mu =  [cos(mu) 0 sin(mu); 0 1 0; -sin(mu) 0 cos(mu)];
    Omega = I(3);
    Theta = [cos(theta_p) 0 sin(theta_p); 0 1 0; -sin(theta_p) 0 cos(theta_p)];
    Theta2 = Theta*Theta;

    image_lab = Theta2;
    Chi = [1 0 0; 0 cos(chi) -sin(chi); 0 sin(chi) cos(chi)];
    Phi = [cos(phi) 0 -sin(phi); 0 1 0; sin(phi) 0 cos(phi)];

    lab_sample = Mu*Omega*Chi*Phi;
    image_sample_x = image_lab[1,:]'*lab_sample;
    grain_lab = (lab_sample*sample_grain)';
    char = chars[1]

    # define coordinate transforms that depend on dislocation character
    dis_grain = dis_grain_all[:,:,char]

    dis_lab = dis_grain*grain_lab;
    dis_sample = dis_lab*lab_sample;

    noise_prec = 1/noise_var
    ll = 0
    for ii = 1:Npixels
        x_lab = x_lab_vec[ii]

        for jj = 1:Npixels
            y_lab = y_lab_vec[jj]
        
            # resolution function (the integrand)
            function res_fun(z_lab,locs)
                
                Fdis_sum = zeros(3,3)
                # Lab to dislocation coordinates
                for i = 1:N_dislocs
                    
                    # put out location and character of i-th dislocation
                    loc = locs[1+2*(i-1):2*i]
        
                    # lab -> dislocation frame
                    r_lab = [x_lab + z_lab/tan(2*theta_p); y_lab; z_lab]
        
                    r_dis = dis_lab[1:2,:]*r_lab;
                    xd = r_dis[1]-loc[1]
                    yd = r_dis[2]-loc[2];
        
                    # helper variables for strain computation
                    sqx = xd^2;
                    sqy = yd^2;
                    denom = (sqx + sqy)^2;
                    nyfactor = 2*ny*(sqx+sqy);
        
                    # Form strain tensor in dislocation frame
                    Fdis = [-yd*(3*sqx + sqy - nyfactor) xd*(3*sqx+sqy-nyfactor) 0;
                            -xd*(3*sqy+sqx-nyfactor) yd*(sqx-sqy-nyfactor) 0;
                            0 0 0];
                    Fdis_sum += bfactor*Fdis/denom;
        
                end

                Fdis_sum = Fdis_sum + I(3)
                
                # transform to sample frame
                Fsample = dis_sample'*Fdis_sum*dis_sample
                # Compute qi (qs -> qi)
                qs = (Fsample')\q_hkl - q_hkl;
                qs = qs + [phi - TwoDeltaTheta/2; chi; (TwoDeltaTheta/2)/tan(theta_p)]; # Eq 40 (also Eq. 20)
        
                # compute res(qi_1)
                resq_fn(image_sample_x*qs)
        
            end
        
            # compute integral
            mu = image_scale_constant*sum([res_fun(node, locs)*weight for (node,weight) in zip(nodes,weights)]) 
            pixel_var = mu + background + noise_var


            ll += -0.5*(obs_im[ii,jj] - mu)^2/pixel_var - 0.5*log(pixel_var)
           

        end
    end

    return ll

end


function image_to_dis_i(loc_im_i, dis_lab; dims=2)

    loc_lab_i = [loc_im_i[2]*tan(2*theta); loc_im_i[1]]

    loc_dis_i = dis_lab*[loc_lab_i;0]

    return loc_dis_i[1:dims]

end

function image_to_dis(loc_im, dis_lab; dims=2)

    num_dislocs = Int(length(loc_im)/2)
    loc_dis = zeros(dims*num_dislocs)
    for i in 1:num_dislocs

        loc_im_i = loc_im[2*i-1:2*i]
        loc_dis[dims*i-(dims-1):dims*i] = image_to_dis_i(loc_im_i, dis_lab; dims=dims)

    end
    
    return loc_dis
end


function dis_to_image_i(loc_dis_i, dis_lab)

    loc_dis_i_3 = -(dis_lab[1:2,3]'*loc_dis_i)/dis_lab[3,3]

    loc_lab_i = dis_lab'*[loc_dis_i;loc_dis_i_3]

    loc_im_i = [loc_lab_i[2]; loc_lab_i[1]/tan(2*theta)]

    return loc_im_i

end


function dis_to_image(loc_dis, dis_lab)

    num_dislocs = Int(length(loc_dis)/2)
    loc_im = zeros(2*num_dislocs)
    for i in 1:num_dislocs

        loc_dis_i = loc_dis[2*i-1:2*i]
        loc_im[2*i-1:2*i] = dis_to_image_i(loc_dis_i, dis_lab)

    end

    return loc_im
end



function vecvec_to_mat(vecvec)
    dim1 = length(vecvec)
    dim2 = length(vecvec[1])
    mat = zeros(Float64, dim1, dim2)
    for i in 1:dim1
            for j in 1:dim2
            mat[i,j] = vecvec[i][j]
            end
    end
    return mat
end