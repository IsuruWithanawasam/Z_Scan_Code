"""
Z-scan: PARAMETER EXTRACTION PROGRAM (YAG FINAL)
Features:
1. Handles Transparent Materials (Alpha=0, Beta=0)
2. Ignores faulty Open Aperture data
3. AUTOMATICALLY calculates Beam Waist (w0)
4. Uses MANUAL PEAK POWERS from experimental log
5. Generates correct plots (mm/um)
"""

# 📝 Modify user settings within the box 📝
#╒════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
#│          

data_filename = "Z_Scan_Data.csv" 

install_libs = False 

data_res_survey = True 

save_to_file = True 
                                    
# --- UPDATED PARAMETERS FOR YAG EXPERIMENT ---

k_ext = 0.0 # Transparent at 1060nm
L_sample = 2e-3 # 2mm thickness

# !!! VERIFY LASER SPECS !!!
tau = 90e-15 #s 
f_rep = 100e6 #Hz 

lda_0 = 1.06e-6 # 1060 nm

# Setup Length (Lens to Aperture). Verify on optical table.
L_setup = 300e-3 

f_lens = 150e-3 # 150 mm lens
n_prop_media = 1.0 # Air
a = 0.75e-3 # 1.5mm diameter pinhole

# Search range for YAG (approx 6e-20 m^2/W)
n_2_search = [1e-20, 15e-20] 

N_obj = 1024 # Even number for FFT

CALIBRATION_FACTOR_W_PER_NVS = 0.05714 

#│                                                                                                                                │
#╘════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

if(install_libs):
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'datetime'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'bokeh'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lmfit'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sklearn'])
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from bokeh.plotting import figure as bokeh_fig
from bokeh.plotting import show as bokeh_show
from scipy import stats
from scipy.optimize import curve_fit
from lmfit.models import LorentzianModel
from sklearn.linear_model import LinearRegression

# --- HELPER FUNCTIONS ---

def check_out_of_lin_range(x, lin_ranges):
    aux=1
    for ii in np.arange(len(lin_ranges)):
        if(x>=lin_ranges[ii,0]):
            if(x<=lin_ranges[ii,1]):
                aux*=0
    return bool(aux)

def error_percentage(x, x_ref):
    return abs((x-x_ref)/x_ref)

def linear_diff(a, x0, x): 
    return a*(x0-x)

def linear_f(a, b, x):
    return a*x+b

def lorentzian(A, mu, sigma, x):
    return (A/np.pi)*(sigma/((x-mu)**2+sigma**2))

def log_exp(P_in,F):
    return (1/F)*np.log(1+(F*P_in))

def T_diff(X):
    return max(X)-min(X)

def beam(X,Y, lda_0, P_PEAK, w_0, z_0, n_prop_media, z, t):
    k=n_prop_media*(2*pi/lda_0) 
    I_0=2*P_PEAK/(pi*(w_0**2)) 
    E_0_norm=np.sqrt(I_0/(2*c_light*epsilon_0*n_prop_media)) 
    omega=c_light*n_prop_media*k 
    w_z=w_0*np.sqrt(1+(z/z_0)**2) 
    R_z=z*(1+(z_0/z)**2) 
    r=np.sqrt(X**2+Y**2) 
    E_zrt=E_0_norm*(w_0/w_z)*np.exp(-((r/w_z)**2)+(I*k*(r**2)/(2*R_z)))*np.exp(I*((n_prop_media*k*z)-(omega*t))) 
    return E_zrt 

def beam_NL(L_obj, N_obj, lda_0, P_PEAK, w_0, z_0, n_prop_media, z, t, L_sample, alpha, L_eff, n_2, beta_TPA):
    dx = L_obj/N_obj 
    x = np.arange(-N_obj*dx/2,N_obj*dx/2,dx) 
    X, Y = np.meshgrid(x,x) 
    k_0=2*pi/lda_0 
    E_zrt=beam(X,Y, lda_0, P_PEAK, w_0, z_0, n_prop_media, z, t) 
    I_zrt=2*c_light*epsilon_0*n_prop_media*(abs(E_zrt)**2) 
    
    factor_1=np.exp(-0.5*alpha*L_sample)*E_zrt 
    
    # !!! FIXED SECTION FOR ZERO BETA !!!
    if beta_TPA == 0:
        factor_2 = np.exp(1j * k_0 * n_2 * L_eff * I_zrt)
    else:
        factor_2=(1+(beta_TPA*L_eff*I_zrt))**(-0.5+(1j*k_0*n_2/beta_TPA)) 
        
    E_after_sample=np.multiply(factor_1,factor_2) 
    return E_after_sample 

def propaFT(u1, lda_0, dist, dx):
    k = 2*np.pi/lda_0
    tam = len(u1)
    df = 1/(tam*dx)
    f = np.arange(-tam*df/2,tam*df/2,df)
    Fx , Fy = np.meshgrid(f,f)
    H = np.exp(-1j*np.pi*lda_0*dist*(Fx**2 + Fy**2))
    U1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(u1)))
    return np.exp(1j*k*dist)*np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(U1*H)))

def iris(L_obj, N_obj, u1,a):
    dx = L_obj/N_obj 
    x2 = np.arange(-N_obj*dx/2,N_obj*dx/2,dx)   
    X2, Y2 = np.meshgrid(x2,x2)  
    r=np.sqrt(X2**2+Y2**2) 
    aperture=np.heaviside(-(r-a),1) 
    return np.multiply(aperture,u1)

def detect(L_obj, N_obj, n_prop_media, u1):
    dx = L_obj/N_obj 
    u_aux=2*n_prop_media*c_light*epsilon_0*np.abs(u1)**2 
    integration=np.sum(u_aux)*(dx**2) 
    return integration

def Tpv_P_slope(n_2, beta_TPA, alpha, L_sample, L_eff, P_peak_max, lda_0, w_0, z_0, f_lens, L_setup, aa, n_prop_media, L_obj, N_obj,  t):
    dx = L_obj/N_obj 
    Tpv_vs_Power=[] 
    for Power in np.linspace(0.001*P_peak_max, P_peak_max, 3): 
        PD_oa=[]
        PD_ca=[]
        z_sample_pos=np.linspace(-2.01*z_0, 2*z_0, 15) 
        for Z in z_sample_pos:             
            BEAM_sampled=beam_NL(L_obj, N_obj, lda_0, Power, w_0, z_0, n_prop_media, Z, t, L_sample, alpha, L_eff, n_2, beta_TPA)
            BEAM_out=propaFT(BEAM_sampled,lda_0,(L_setup-f_lens)-Z,dx)     
            BEAM_irised=iris(L_obj, N_obj, BEAM_out, aa)     
            PD_oa.append(detect(L_obj, N_obj, n_prop_media, BEAM_out)) 
            PD_ca.append(detect(L_obj, N_obj, n_prop_media, BEAM_irised)) 
        
        # NORMALIZATION LOGIC (Modified for YAG)
        BEAM_sampled=beam_NL(L_obj, N_obj, lda_0, Power, w_0, z_0, n_prop_media, 0.99*(L_setup-f_lens), t, L_sample, alpha, L_eff, n_2, beta_TPA)
        BEAM_out=propaFT(BEAM_sampled,lda_0,0.01*(L_setup-f_lens),dx)     
        BEAM_irised=iris(L_obj, N_obj, BEAM_out, aa)     
        
        PD_REF_ca=detect(L_obj, N_obj, n_prop_media, BEAM_irised) 
        PD_ca=PD_ca/PD_REF_ca 
        
        PD_renorm = np.array(PD_ca) # IGNORE OA DATA
             
        Tpv_vs_Power.append(T_diff(PD_renorm)) 
    lin_regress = LinearRegression(fit_intercept=False).fit((np.linspace(0.001*P_peak_max, P_peak_max, 3)).reshape((-1, 1)), np.array(Tpv_vs_Power))
    slope=lin_regress.coef_[0]
    return slope

# --- INITIALIZATION ---

I=0+1j 
pi=np.pi 
t = 0 
c_light=299792458 
epsilon_0=8.854188e-12  

raw_data=pd.read_csv(data_filename) 

alpha=4*pi*k_ext/lda_0 
k_0=2*pi/lda_0 

if alpha == 0:
    L_eff = L_sample
else:
    L_eff=(1-np.exp(-alpha*L_sample))/alpha 

colormap =cm.get_cmap("magma")
bokehpalette = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]

# 📜📜 DATA PREPARATION 📜📜

number_of_power_sweeps = int((len(list(raw_data)) - 1) / 3) 
if 'CALIBRATION_FACTOR_W_PER_NVS' in locals() and CALIBRATION_FACTOR_W_PER_NVS != 1.0:
    print(f"\n📢 Applying nVs to W conversion with factor: {CALIBRATION_FACTOR_W_PER_NVS} W/nVs")
    for ii in np.arange(number_of_power_sweeps):
        P_header_oa = list(raw_data)[2+3*ii]
        P_header_ca = list(raw_data)[3+3*ii]
        raw_data[P_header_oa] = raw_data[P_header_oa] * CALIBRATION_FACTOR_W_PER_NVS
        raw_data[P_header_ca] = raw_data[P_header_ca] * CALIBRATION_FACTOR_W_PER_NVS

lin_data=raw_data.copy()
data=raw_data.copy()

fig2= bokeh_fig(width=1000, height=600)
for ii in np.arange(number_of_power_sweeps):    
    dataX=np.array(raw_data.iloc[:,1+3*ii])
    dataY=np.array(raw_data.iloc[:,3+3*ii])
    fig2.line(dataX, [float(i) for i in dataY], line_width=2, line_dash='solid', color='#52a736')
    fig2.title = "Closed aperture measurements"
bokeh_show(fig2)

print("\n PROCESSING DATA...")

lin_range_num=int(input("\nEnter number of linear ranges (typically 2):\n>>: "))
lin_ranges=np.zeros((lin_range_num,2))
for ii in np.arange(len(lin_ranges)):
    lin_ranges[ii,0]=float(input("\nEnter lower limit for range "+str(ii+1)+" (in mm):\n>>: "))
    lin_ranges[ii,1]=float(input("\nEnter upper limit for range "+str(ii+1)+" (in mm):\n>>: "))
    
for ii in np.arange(number_of_power_sweeps):
    Z_header=list(lin_data)[1+3*ii]
    P_header=list(lin_data)[2+3*ii]
    P2_header=list(lin_data)[3+3*ii]
    for jj in np.arange(len(lin_data)):
        if(check_out_of_lin_range(lin_data[Z_header][jj],lin_ranges)):
            lin_data.loc[jj, P_header]=float('nan')
            lin_data.loc[jj, P2_header]=float('nan')

for ii in np.arange(number_of_power_sweeps):
    Z_header=list(lin_data)[1+3*ii]
    lin_data[Z_header]=1e-3*lin_data[Z_header]
    data[Z_header]=1e-3*data[Z_header]

# Linear Params Calculation (With Manual Power Override)
lin_params=np.zeros((number_of_power_sweeps,6))
for ii in np.arange(number_of_power_sweeps):
    dataX=np.array(lin_data.iloc[:,1+3*ii])
    dataY=np.array(lin_data.iloc[:,2+3*ii])    
    mask = ~np.isnan(dataX) & ~np.isnan(dataY)
    lin_params[ii,0], lin_params[ii,1], lin_params[ii,2], lin_params[ii,3], lin_params[ii,4] = stats.linregress(dataX[mask], dataY[mask])

# !!! MANUAL OVERRIDE FOR PEAK POWERS !!!
print("\n⚠️ OVERRIDING PEAK POWER VALUES (Based on Graph) ⚠️")
# Values read from your uploaded image:
manual_peak_powers = [0.0037,0.00287,0.00234,0.00185,0.0013,0.000905] 

if len(manual_peak_powers) != number_of_power_sweeps:
    print(f"❌ ERROR: You provided {len(manual_peak_powers)} values, but data has {number_of_power_sweeps} sweeps.")
    P_peak_max = lin_params[-1,1]/(tau*f_rep) # Fallback
else:
    print(f"Using Manual Peak Powers: {manual_peak_powers}")
    P_peak_max = manual_peak_powers[-1]
    # Update lin_params to trick subsequent functions
    for ii in range(number_of_power_sweeps):
        lin_params[ii,1] = manual_peak_powers[ii] * (tau * f_rep)

lin_params2=np.zeros((number_of_power_sweeps,6))
for ii in np.arange(number_of_power_sweeps):
    dataX=np.array(lin_data.iloc[:,1+3*ii])
    dataY=np.array(lin_data.iloc[:,3+3*ii])    
    mask = ~np.isnan(dataX) & ~np.isnan(dataY)
    lin_params2[ii,0], lin_params2[ii,1], lin_params2[ii,2], lin_params2[ii,3], lin_params2[ii,4] = stats.linregress(dataX[mask], dataY[mask])
    
    Z_header=list(data)[1+3*ii]
    P_header=list(data)[2+3*ii]
    P2_header=list(data)[3+3*ii]
    
    # Normalize CA data by its OWN linear trend (OA IGNORED)
    data[P2_header]=data[P2_header]/linear_f(lin_params2[ii,0],lin_params2[ii,1],data[Z_header])

# Plot Renormalized Data
fig3= bokeh_fig(width=1000, height=600)
for ii in np.arange(number_of_power_sweeps):    
    dataX=np.array(data.iloc[:,1+3*ii])
    dataY=np.array(data.iloc[:,3+3*ii])
    fig3.line(dataX, [float(i) for i in dataY], line_width=2, line_dash='solid', color='#64b066')
    fig3.title = "Closed aperture RENORMALIZED (OA Ignored)"
bokeh_show(fig3)

negative_n_2=bool(int(input("\nSelect expected sign of n_2:\n(0) Positive (YAG).\n(1) Negative.\n>>: ")))

# 🔴🔴 BEAM WAIST CALCULATION (METHOD B) 🔴🔴

print("\n⚠️ AUTOMATIC BEAM WAIST CALCULATION ⚠️")
print("We need to find the Peak-Valley distance to calculate w0.")
print("Please look at your 'Renormalized CA' plot (in browser) and choose a clean Z-range.")

try:
    z_limit_min_mm = float(input(" >> Enter Minimum Z (mm) to consider for Peak/Valley search: "))
    z_limit_max_mm = float(input(" >> Enter Maximum Z (mm) to consider for Peak/Valley search: "))
except ValueError:
    print("Invalid input. Using default wide range (0 to 20mm)")
    z_limit_min_mm = 0
    z_limit_max_mm = 20

# Use the last power sweep (usually highest signal)
z_col = 1 + 3 * (number_of_power_sweeps - 1)
ca_col = 3 + 3 * (number_of_power_sweeps - 1)

z_data_m = np.array(data.iloc[:, z_col])
norm_data = np.array(data.iloc[:, ca_col]) # Already normalized above

# Filter by range
mask_z = (z_data_m >= z_limit_min_mm*1e-3) & (z_data_m <= z_limit_max_mm*1e-3)
z_subset = z_data_m[mask_z]
norm_subset = norm_data[mask_z]

if len(z_subset) > 2:
    idx_max = np.argmax(norm_subset)
    idx_min = np.argmin(norm_subset)
    z_peak = z_subset[idx_max]
    z_valley = z_subset[idx_min]
    
    delta_z_pv = abs(z_peak - z_valley)
    
    # Calculate w0
    z0_geom = delta_z_pv / 1.7
    w_0_calc = np.sqrt((z0_geom * lda_0) / np.pi)
    
    print(f" >> Found Peak Z: {z_peak*1e3:.2f} mm")
    print(f" >> Found Valley Z: {z_valley*1e3:.2f} mm")
    print(f" >> Delta Z (p-v): {delta_z_pv*1e3:.2f} mm")
    print(f" >> CALCULATED w0: {w_0_calc*1e6:.2f} microns")
    
    # UPDATE GLOBAL VARIABLE
    w_0 = w_0_calc
    z_0 = z0_geom
else:
    print("⚠️ Range selection failed. Defaulting to manual 25um.")
    w_0 = 25e-6
    z_0 = (pi * w_0**2) / lda_0

beta_TPA = 0.0 

print(f"--> Final w_0 used for fit: {w_0} m")
print(f"--> Final z_0 used for fit: {z_0} m")

# 📊📊 EXTRA GRAPHS GENERATION 📊📊

if data_res_survey:
    # 1. Peak-Valley Difference vs Peak Power (Using Manual Powers)
    peak_valley_diff=[]
    powers_peak = np.array(manual_peak_powers) # USE MANUAL POWERS
    
    for ii in np.arange(number_of_power_sweeps):
        peak_valley_diff.append(T_diff(np.array(data.iloc[:,3+3*ii])))
    
    fig_pv = bokeh_fig(width=1000, height=600, title="Peak-Valley Transmission Difference vs Peak Power")
    fig_pv.scatter(powers_peak, peak_valley_diff, color='#d62728', size=10, legend_label="Experimental Data")
    
    # Linear Fit Line
    slope, intercept, _, _, _ = stats.linregress(powers_peak, peak_valley_diff)
    y_fit = slope * powers_peak + intercept
    fig_pv.line(powers_peak, y_fit, line_width=2, line_dash='dashed', color='black', legend_label="Linear Fit")
    
    fig_pv.xaxis.axis_label = "Peak Input Power (W)"
    fig_pv.yaxis.axis_label = "Delta T (p-v)"
    bokeh_show(fig_pv)

    # --- BEAM PROFILE PLOTS ---
    # Create a coordinate grid in millimeters
    
    # A. BEAM AT APERTURE (Far Field)
    z_ap = L_setup - f_lens
    w_ap = w_0 * np.sqrt(1 + (z_ap/z_0)**2)
    L_sim_ap = 6 * w_ap  # Simulation window size (approx 6x beam width)
    
    dx_ap = L_sim_ap / N_obj
    x_ap = np.arange(-N_obj*dx_ap/2, N_obj*dx_ap/2, dx_ap)
    X_ap, Y_ap = np.meshgrid(x_ap, x_ap)
    
    BEAM_ap = beam(X_ap, Y_ap, lda_0, 1.0, w_0, z_0, n_prop_media, z_ap, 0)
    Irr_ap = np.abs(BEAM_ap)**2
    
    # Scale to mm for plotting
    L_sim_ap_mm = L_sim_ap * 1e3
    x_ap_mm = x_ap * 1e3
    
    # Plot 2D Aperture Profile
    fig_2d_ap = bokeh_fig(title="2D Beam Profile at APERTURE Plane (mm)", 
                          tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
    fig_2d_ap.x_range.range_padding = fig_2d_ap.y_range.range_padding = 0
    fig_2d_ap.image(image=[Irr_ap], x=-L_sim_ap_mm/2, y=-L_sim_ap_mm/2, dw=L_sim_ap_mm, dh=L_sim_ap_mm, palette="Viridis256", level="image")
    fig_2d_ap.xaxis.axis_label = "x (mm)"
    fig_2d_ap.yaxis.axis_label = "y (mm)"
    fig_2d_ap.xgrid.grid_line_color = None
    fig_2d_ap.ygrid.grid_line_color = None
    bokeh_show(fig_2d_ap)

    # Plot 1D Aperture Profile
    center_idx = int(N_obj/2)
    profile_1d_ap = Irr_ap[center_idx, :]
    fig_1d_ap = bokeh_fig(width=1000, height=600, title="1D Beam Profile at APERTURE Plane (Cross-section)")
    fig_1d_ap.line(x_ap_mm, profile_1d_ap, line_width=3, color='#1f77b4')
    fig_1d_ap.xaxis.axis_label = "Position (mm)"
    fig_1d_ap.yaxis.axis_label = "Intensity (arb. units)"
    bokeh_show(fig_1d_ap)

    # B. BEAM AT FOCUS (z approx 0)
    # Beam is tiny here (microns), so we need a much smaller grid
    L_sim_foc = 8 * w_0 
    dx_foc = L_sim_foc / N_obj
    x_foc = np.arange(-N_obj*dx_foc/2, N_obj*dx_foc/2, dx_foc)
    X_foc, Y_foc = np.meshgrid(x_foc, x_foc)
    
    # Use 1e-9 instead of 0 to avoid DivisionByZero in R(z)
    BEAM_foc = beam(X_foc, Y_foc, lda_0, 1.0, w_0, z_0, n_prop_media, 1e-9, 0)
    Irr_foc = np.abs(BEAM_foc)**2
    
    # Scale to microns (um) for plotting
    L_sim_foc_um = L_sim_foc * 1e6
    x_foc_um = x_foc * 1e6
    
    # Plot 2D Focus Profile
    fig_2d_foc = bokeh_fig(title="2D Beam Profile at FOCUS (microns)", 
                           tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
    fig_2d_foc.x_range.range_padding = fig_2d_foc.y_range.range_padding = 0
    fig_2d_foc.image(image=[Irr_foc], x=-L_sim_foc_um/2, y=-L_sim_foc_um/2, dw=L_sim_foc_um, dh=L_sim_foc_um, palette="Inferno256", level="image")
    fig_2d_foc.xaxis.axis_label = "x (um)"
    fig_2d_foc.yaxis.axis_label = "y (um)"
    fig_2d_foc.xgrid.grid_line_color = None
    fig_2d_foc.ygrid.grid_line_color = None
    bokeh_show(fig_2d_foc)

    # Plot 1D Focus Profile
    profile_1d_foc = Irr_foc[center_idx, :]
    fig_1d_foc = bokeh_fig(width=1000, height=600, title="1D Beam Profile at FOCUS (Cross-section)")
    fig_1d_foc.line(x_foc_um, profile_1d_foc, line_width=3, color='#d62728')
    fig_1d_foc.xaxis.axis_label = "Position (um)"
    fig_1d_foc.yaxis.axis_label = "Intensity (arb. units)"
    bokeh_show(fig_1d_foc)

# 🔎🕳 KERR EFFECT PARAMETER EXTRACTION 🔎🕳

# Continue with N2 extraction using the calculated w0...
L_obj = 30*(2*10*w_0) 
dx = L_obj/N_obj 

# Aperture estimation 
estimate_aperture=bool(int(input('\nSelect method for aperture setting:\n(0) Use value in source code.\n(1) Execute estimation (may be inaccurate).\n>>: ')))
if(estimate_aperture):
    extinction_objective_value=np.mean(np.array(lin_params2[:,1])/np.array(lin_params[:,1]))
    dx = L_obj/N_obj 
    x = np.arange(-N_obj*dx/2,N_obj*dx/2,dx)    
    X, Y = np.meshgrid(x,x)   
    max_guess_val=float(input('Provide a maximun value for the estimation (in m):\n>>: '))
    lower_lim_guess=0
    upper_lim_guess=max_guess_val
    iteration=0
    while(iteration<=10):
        iteration+=1
        err_iter_list=[]
        for a_guess in np.linspace(lower_lim_guess,upper_lim_guess,10):
            val=detect(L_obj, N_obj, n_prop_media, iris(L_obj, N_obj, beam(X,Y, lda_0, 1, w_0, z_0, n_prop_media, L_setup-f_lens, 0),a_guess))/detect(L_obj, N_obj, n_prop_media, beam(X,Y, lda_0, 1, w_0, z_0, n_prop_media, L_setup-f_lens, 0))
            err_iter_list.append(error_percentage(val,extinction_objective_value))
        step_guess=(upper_lim_guess-lower_lim_guess)/9
        best_guess=np.linspace(lower_lim_guess,upper_lim_guess,10)[err_iter_list.index(min(err_iter_list))]
        lower_lim_guess=best_guess-step_guess
        upper_lim_guess=best_guess+step_guess
        if(min(err_iter_list)<=0.01):
            break
    aa=best_guess
else:
    aa=a

lin_regress = LinearRegression(fit_intercept=False).fit((np.array(lin_params[:,1])/(tau*f_rep)).reshape((-1, 1)),np.array(peak_valley_diff))
pvd_slope_objective=lin_regress.coef_[0]

# n2 optimization
if(bool(int(input("\nSelect method for n_2 search initialization:\n(0) Use limits in source code.\n(1) Enter limits via console.\n>>: ")))):
    n_2_search=[]
    n_2_search.append(float(input("Enter lower limit for n_2 search interval:\n>>: ")))
    n_2_search.append(float(input("Enter upper limit for n_2 search interval:\n>>: ")))

preliminary_calculated_slopes=[]
for N_2 in np.linspace(n_2_search[0],n_2_search[1],3):
    preliminary_calculated_slopes.append(Tpv_P_slope(N_2, beta_TPA, alpha, L_sample, L_eff, P_peak_max, lda_0, w_0, z_0, f_lens, L_setup, aa, n_prop_media, L_obj, N_obj,  t))
lin_regress2 = LinearRegression(fit_intercept=False).fit((np.linspace(n_2_search[0],n_2_search[1],3)).reshape((-1, 1)), np.array(preliminary_calculated_slopes))
slope_vs_n2_relation=lin_regress2.coef_[0]
n_2_first_prediction=(1/slope_vs_n2_relation)*pvd_slope_objective 

if(negative_n_2):
    lower_lim_search=10*n_2_first_prediction
    upper_lim_search=0.1*n_2_first_prediction
else:
    lower_lim_search=0.1*n_2_first_prediction
    upper_lim_search=10*n_2_first_prediction

iteration=0
while(iteration<=10):
    iteration+=1
    err_iter_list=[]
    for N_2 in np.linspace(lower_lim_search, upper_lim_search, 5):
        val=Tpv_P_slope(N_2, beta_TPA, alpha, L_sample, L_eff, P_peak_max, lda_0, w_0, z_0, f_lens, L_setup, aa, n_prop_media, L_obj, N_obj,  t)
        err_iter_list.append(error_percentage(val,pvd_slope_objective))
    step_search=abs(upper_lim_search-lower_lim_search)/4 
    best_val=np.linspace(lower_lim_search, upper_lim_search, 5)[err_iter_list.index(min(err_iter_list))] 
    
    if(negative_n_2):
        lower_lim_search=best_val-step_search
        if((best_val+step_search)<0):
            upper_lim_search=best_val+step_search
        else:
            upper_lim_search=0
    else:
        upper_lim_search=best_val+step_search
        if((best_val-step_search)>0):
            lower_lim_search=best_val-step_search
        else:
            lower_lim_search=0
    
    if(min(err_iter_list)<=0.01):
        break
    
n_2=best_val 

# 📰💾 RESULTS PRESENTATION 📰💾

if(save_to_file):
    date=datetime.datetime.now()
    date_now=str(date.year)+'-'+str(date.month)+'-'+str(date.day)+'; '+str(date.hour)+'-'+str(date.minute)+'-'+str(date.second)
    if (not os.path.exists(date_now)):
        os.makedirs(date_now)
    params_file=open(date_now+'/'+str(data_filename[0:-4])+'--Nonlinear_extracted_parameters.csv','w')
    if(estimate_aperture):
        params_file.write('r_a, '+str(aa)+',')
        params_file.write('\n')
    params_file.write('w_0, '+str(w_0)+',')
    params_file.write('\n')
    params_file.write('n_2, '+str(n_2)+',')
    params_file.write('\n')
    params_file.write('beta_TPA, '+str(beta_TPA)+',')
    params_file.write('\n')
    params_file.close()

print("")
print("╓─── ▾▾▾")
print("║ Parameters extracted:")
if(estimate_aperture):
    print('║ r_a: '+str(aa)+' m')
print('║ w_0: '+str(w_0)+' m (CALCULATED FROM P-V)')
print("║ n_2: "+str(n_2)+" m^2/W")
print('║ beta_TPA: '+str(beta_TPA)+' m/W (FIXED, OA ignored)')
print("╙─── ▴▴▴")