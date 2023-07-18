#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits 
from astropy.visualization import simple_norm

from scipy.signal import savgol_filter


# ## get_counts (returns counts of a star)

# In[2]:


# this function takes an individual image files 
# this function takes an individual image files 
def get_final_counts(file, radii = np.arange(20), bg_radius = 20, N = 50, R = 100, plot=False): 
    
    """
    The get_final_counts function takes an image of a star and returns the counts being emitted from the star
    
    Parameters:
    -----------
    • file: image of the star taken by STC7 (fits file)
    
    Optional Parameters:
    ---------------------
    • radii: the range of radii needed to subtract background(set to 20)
    • bg_radius: radius of the circles used to subtract background(set to 20)
    • N: number of circles to subtract background(set to 50)
    • R: distance between the star and subtracting radius(set to 100)
    • plot: plots the final counts of star against the radius with the image of the circle around the star 

    Returns:
    --------
    The function returns one value:
    • final_counts: The average value of the final counts of the star
    
    Usage:
    ------
    final_counts = get_final_counts('Mizar_image.fit', np.arange(20), bg_radius = 20, N = 50, R = 100, plot=True)
    """
    
    this_data = get_data(file) #getting data for the file 
    center_y, center_x, t_array = get_center(this_data) #getting center values for the data 
    
    center_x, center_y = center_x[0], center_y[0]

    counts = np.zeros_like(radii) #declaring counts variable to be length of radii
    area = np.zeros_like(radii) #declaring area variable to be length of radii
    
    _counts = np.zeros((N, 1)) #declaring _counts variable to be variables used for background subtraction 
    _area = np.zeros((N, 1)) #declaring _area variable to be variables used for background subtraction
    
    dist = get_this_dist(center_x, center_y, t_array) #getting distance values from center to individual points

    
    for i in radii: 
        mask = dist < i
        counts[i] = np.sum(t_array[mask])  #calculating the counts of the star
        area[i] = np.sum(mask) #calculating the area of the star
    
    for i in range(0, N): 
        rand_num = np.random.randint(0, high=R) 
        y = (R**2 - rand_num**2)**(1/2)
        _counts[i], _area[i] = get_counts(rand_num, y, t_array, bg_radius) #calculating background counts and area
    
    average_bg = np.average(_counts/_area) # subtracting the background counts from the original circle 
    mult_area = area*average_bg
    final_counts = counts - mult_area #getting final counts
    
    if plot==True: #plotting the counts against area 
        
        norm = simple_norm(this_data, 'log', log_a=1e11)
        
        y_max, x_max = np.where(this_data==np.max(this_data))
        x_max = x_max[0]
        y_max = y_max[0]
        
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 3))
        
        axes[0].imshow(this_data, cmap = 'Greys', norm=norm, origin='lower')
        axes[0].scatter(x_max, y_max, color="yellow", fc='none', ec='white', s=2000) 
        axes[0].set_xlim(x_max-100, x_max+100) 
        axes[0].set_ylim(y_max-100, y_max+100) #plotting an image of the circle around the star for which counts have been calculated
        
        axes[1].scatter(radii, final_counts, color='navy')
        axes[1].set_xlabel("Radii of Background")
        axes[1].set_ylabel("Counts of the Star")#plotting the counts against the radii
        
    
    return np.mean(final_counts[-5:])


# In[3]:


# function which gets the data of the file
def get_data(file):
    data = fits.getdata(file) #getting data using fits 
    return data # returns 2D array 


# In[4]:


# function which gets the center value of the 2D array; returns center_vals and temp_array
def get_center(data): 
    
    lim_val = np.array([100])
    lim_val = lim_val[0] #limiting value of the image (for ease of taking background)
    
    for i in range(len(data)):
        ymax, xmax = np.where(data==np.max(data))
        temp_array = data[int(ymax[0])-lim_val: int(ymax[0])+lim_val, int(xmax[0])-lim_val:int(xmax[0])+lim_val]#creating a temporary array for which distance calculations can be made 
        center_y, center_x = np.where(temp_array==np.max(temp_array)) #getting center values for the data 
        
    return center_y, center_x, temp_array


# In[5]:


# gets the counts of that particular star 
def get_counts(x, y, array, rad): #takes center x value, center y value, temp_array and radius
    distances = get_this_dist(x, y, array) #calculating the distance from the center to various values  
    count = np.sum(array[distances < rad]) #calculating counts where distance is less than the given radius
    areas = np.sum(distances < rad) #calculating area over which counts have been calculated
    return count, areas


# In[6]:


def get_this_dist(x, y, temp): #takes center x and y values and temporary array 
     
    dist = np.zeros((len(temp[0]), len(temp[0]))) #creating a 2-D distances array 
   
    for i in range(0, len(temp[0])):
        for j in range(0, len(temp[0])):
            dist[i, j] = np.sqrt((i - x)**2 + (j - y)**2) #store distance from center of the star to every other pixel
    return dist 


# ## get_star_mags (relative RGB magnitudes for STC7)

# In[7]:


# defining a function to integrate over a curve using Reimann sum method
def integrate(x, y):
    dx = np.diff(x)
    return np.sum(y[:-1]*dx)


# In[8]:


# creating interpolating fucntions for each filter response
def ired(x): 
    return np.interp(x, this_wave[0], tr_[0])
def igreen(x):
    return np.interp(x, this_wave[1], tr_[1])
def iblue(x):
    return np.interp(x, this_wave[2], tr_[2])
def ilum(x):
    return np.interp(x, this_wave[3], tr_[3])
def iqe(x):
    return np.interp(x, this_qe, qe)


# In[9]:


def t_lambda(wavelength, flux):
    
    R, G, B, L = 0, 1, 2, 3
    
    qe_interp = iqe(wavelength) #calling interpolating functions using wavelength from star_data for qe 
    red_interp = ired(wavelength)#calling interpolating functions using wavelength from star_data for red 
    green_interp = igreen(wavelength)#calling interpolating functions using wavelength from star_data for green 
    blue_interp = iblue(wavelength)#calling interpolating functions using wavelength from star_data for blue 
    lum_interp = ilum(wavelength)#calling interpolating functions using wavelength from star_data for lum
    
    mult_val_ref = qe_interp*f_lambda(wavelength)*wavelength #defining multiplied value using f_lambda value as flux
    mult_val_calc = qe_interp*flux*wavelength #defining multiplied value using star flux as flux
   
    red_ref = mult_val_ref*red_interp  #multiplying reference value by interpolated value in red
    green_ref = mult_val_ref*green_interp #multiplying reference value by interpolated value in green
    blue_ref = mult_val_ref*blue_interp #multiplying reference value by interpolated value in blue
    lum_ref = mult_val_ref*lum_interp #multiplying reference value by interpolated value in lum
    
    ref[:,R] = red_ref # adding red values in reference array 
    ref[:,G] = green_ref # adding green values in reference array 
    ref[:,B] = blue_ref # adding blue values in reference array 
    ref[:,L] = lum_ref # adding lum values in reference array 
    

    red_calc = mult_val_calc*red_interp #multiplying calculated values by interpolated value in red 
    green_calc = mult_val_calc*green_interp #multiplying calculated values by interpolated value in green 
    blue_calc = mult_val_calc*blue_interp #multiplying calculated values by interpolated value in blue 
    lum_calc = mult_val_calc*lum_interp #multiplying calculated values by interpolated value in lum 
    
    calc[:,R] = red_calc # adding red values in calculated array 
    calc[:,G] = green_calc # adding green values in calculated array 
    calc[:,B] = blue_calc # adding blue values in calculated array 
    calc[:,L] = lum_calc # adding lum values in calculated array 
    
    return ref, calc 


# In[108]:


# flux for standard values 
def f_lambda(x):
    return 0.10885/(x**2) 


# In[109]:


# getting quantum efficiency data if the camera is stc7 else return 1
def get_qe(name):
     # return quantum efficiency data if we have stc7
    wave_qe, qe = np.loadtxt("Quantum_Efficiency_Dataset.csv", delimiter=',', unpack=True)
    return wave_qe, qe #all wavelengths have to be in Angstrom


# In[39]:


def get_star_mags(ref_counts, targ_counts, targ_hr_num, useLum=False): 
    
    """
    The get_star_mags function takes counts for the reference star, target star and the hr number of target star. 
    Returns relative RGB magnitudes of the target star.
    
    Parameters:
    -----------
    • ref_counts: array of reference star counts in RGB or RGBL(in that order)
    • targ_counts: array of target star counts in RGB or RGBL (in that order)
    • targ_hr_num: hr number of the target star
    
    Optional Parameters:
    -----------
    • useLum: set this to True if luminance filter values are in the array 

    Returns:
    --------
    The function returns 3(or 4)values (in this order):
    • m_red: magnitude of the target star in red filter 
    • m_green: magnitude of the target star in green filter 
    • m_blue: magnitude of the target star in blue filter 
    • m_lum: magnitude of the target star in luminance filter (if useLum=True)
    
    Usage:
    ------
    red, green, blue, lum = get_star_mags(ref_counts, targ_counts, targ_hr_num, useLum=True)
    """
    matching_hrs = np.where(hr==targ_hr_num)[0]
    
    hr_index = matching_hrs[0]
    
    stcblue = loaded['STC_B'] #loading stc red value onto variable
    stcgreen = loaded['STC_G'] #loading stc green value onto variable
    stcred = loaded['STC_R'] #loading stc blue value onto variable
    stclum  = loaded['STC_L'] #loading stc lum value onto variable
    
    if not useLum:
        R, G, B = 0, 1, 2
        ref_red = ref_counts[R]
        ref_green = ref_counts[G]
        ref_blue = ref_counts[B]

        targ_red = targ_counts[R]
        targ_green = targ_counts[G]
        targ_blue = targ_counts[B]

        m_red = stcred[hr_index] -2.5*np.log10(targ_red/ref_red)
        m_green = stcgreen[hr_index] -2.5*np.log10(targ_green/ref_green)
        m_blue = stcblue[hr_index]-2.5*np.log10(targ_blue/ref_blue)

        return m_red, m_green, m_blue
        
    else:
        R, G, B, L = 0, 1, 2, 3
        
        ref_red = ref_counts[R]
        ref_green = ref_counts[G]
        ref_blue = ref_counts[B]
        ref_lum = ref_counts[L]

        targ_red = targ_counts[R]
        targ_green = targ_counts[G]
        targ_blue = targ_counts[B]
        targ_lum = targ_counts[L]

        m_red = stcred[hr_index] -2.5*np.log10(targ_red/ref_red)
        m_green = stcgreen[hr_index] -2.5*np.log10(targ_green/ref_green)
        m_blue = stcblue[hr_index] -2.5*np.log10(targ_blue/ref_blue)
        m_lum = stclum[hr_index] -2.5*np.log10(targ_lum/ref_lum)

        return m_red, m_green, m_blue, m_lum 


# ## get_temp (return average temperature)

# In[15]:


# defining a function to load catalogue 
def load_catalogue():
    catalogue = np.zeros((1346, 10))
    df = pd.read_csv("RGB_Data_Fulltable.csv")
    return df


# In[16]:


loaded = load_catalogue()
hr = loaded['HR']
plx = loaded['Parallax']


# In[17]:


camera_name='stc7'


# In[18]:


# loading necessary information from catalogue into variables 
def get_catalogue_data(num, allstd=False, allstc=False):
    
   
    stdblue = loaded['standard_B'] # loading standard red value onto variable
    stdgreen = loaded['standard_G']# loading standard green value onto variable
    stdred = loaded['standard_R']# loading standard blue value onto variable
    stcblue = loaded['STC_B'] #loading stc red value onto variable
    stcgreen = loaded['STC_G'] #loading stc green value onto variable
    stcred = loaded['STC_R'] #loading stc blue value onto variable
    stclum  = loaded['STC_L'] #loading stc lum value onto variable
    
    n_filters = 4
    n_stc = n_filters
    n_std = n_filters - 1
    
    R, G, B, L = np.arange(n_filters)
    
    stc = np.zeros((len(hr), n_stc))
    std = np.zeros((len(hr), n_std))
    
    stc[:,R] = stcred
    stc[:,G] = stcgreen
    stc[:,B] = stcblue
    stc[:,L] = stclum
    
    
    std[:,R] = stdred
    std[:,G] = stdgreen
    std[:,B] = stdblue
    
    if(allstd):
        return std
    if(allstc):
        return stc
    
    
    matching_hrs = np.where(hr==num)[0]
    
    if(len(matching_hrs)==0):
        raise ValueError("HR number cannot be found in catalogue.")
        
    elif(len(matching_hrs) > 1):
        print("Multiple HR number matches in catalogue, using the first match")
        
    hr_index = matching_hrs[0]
    
    stc_star = np.zeros(n_stc)
    std_star = np.zeros(n_std)
    
    
    std_star[R] = stdred[hr_index]
    std_star[G] = stdgreen[hr_index]
    std_star[B] = stdblue[hr_index]

    stc_star[R] = stcred[hr_index]
    stc_star[G] = stcgreen[hr_index]
    stc_star[B] = stcblue[hr_index]
    stc_star[L] = stclum[hr_index]

    return std_star, stc_star    


# In[19]:


loaded = load_catalogue()
hr = loaded['HR'] # getting HR numbers 


# In[20]:


# function which gets filter data according to 'camera_name'
def get_filter_data(name):
    if name=='website':
        wavelength_R, tr_R = np.loadtxt('Red_Data.txt', unpack=True) # loading standard filter response in red
        wavelength_G, tr_G = np.loadtxt('Green_Data.txt', unpack=True)# loading standard filter response in green
        wavelength_B, tr_B = np.loadtxt('Blue_Data.txt', unpack=True)# loading standard filter response in blue
        
        #(??) how to get a wavelength array with a range of the shortest wavelength in blue and longest in red
        wavelengthRGB = np.array([wavelength_R, wavelength_G, wavelength_B])
        trRGB = np.array([tr_R, tr_G, tr_B])
        return wavelengthRGB, trRGB
    
    elif name=='stc7':
        
        wavelength_L, tr_L = np.loadtxt('Luminance_Filter_Response_Digitized.csv', delimiter=',', unpack=True)# loading stc filter response in red
        wavelength_R, tr_R = np.loadtxt('Red_Filter_Response_Digitized.csv', delimiter=',', unpack=True)# loading stc filter response in green
        wavelength_G, tr_G = np.loadtxt('Green_Filter_Response_Digitized.csv', delimiter=',', unpack=True)# loading stc filter response in blue
        wavelength_B, tr_B = np.loadtxt('Blue_Filter_Response_Digitized.csv', delimiter=',', unpack=True)# loading stc filter response in lum
        
        #(??) how to get a wavelength array with a range of the shortest wavelength in blue and longest in red
        wavelengthLRGB = np.array([wavelength_R, wavelength_G, wavelength_B, wavelength_L])
        trLRGB = np.array([tr_R, tr_G, tr_B, tr_L])
        return wavelengthLRGB, trLRGB
    
    else:
        raise NameError("Camera Name does not exist")
        return -1, -1 


# In[21]:


allstc = get_catalogue_data(3, allstc=True)


# In[22]:


R, G, B, L = 0, 1, 2, 3


# In[23]:


B_R = allstc[:,B] - allstc[:,R]
G_R = allstc[:,G] - allstc[:,R]
B_G = allstc[:,B] - allstc[:,G]


# In[24]:


# defining a function to get temperature from the catalogues
def get_temp():
    T_eff = np.zeros(len(hr))
    for i in range(len(hr)):
        T_eff[i] = loaded['Teff(K)'][i]
    return T_eff


# In[25]:


# getting temperature values and storing data onto variable
T_eff = get_temp()


# In[26]:


# sort the calculated values of BR, GR, BG wrt temperature 
def sorted_list():
    _T_eff, _BR = zip(*sorted(zip(T_eff, B_R)))
    _T_eff, _GR = zip(*sorted(zip(T_eff, G_R)))
    _T_eff, _BG = zip(*sorted(zip(T_eff, B_G)))
    return _T_eff, _BR, _GR, _BG 


# In[27]:


def fitted_vals(plot=False):
    T_eff_new, calc_BR_new, calc_GR_new, calc_BG_new = sorted_list()
    
    # changing backto original variable names 
    T_eff_this = np.array(T_eff_new)
    calc_BR = np.array(calc_BR_new)
    calc_GR = np.array(calc_GR_new)
    calc_BG = np.array(calc_BG_new)
    
    # fitted curves using savgol_filter 
    fitted_BR = savgol_filter(calc_BR, 201, 1) 
    fitted_GR = savgol_filter(calc_GR, 201, 1)
    fitted_BG = savgol_filter(calc_BG, 201, 1)
    
    if(plot):
        plt.plot(T_eff_this, fitted_BR, color='blue', label='B-R')
        plt.plot(T_eff_this, fitted_GR, color='green', label='G-R')
        plt.plot(T_eff_this, fitted_BG, color='purple', label='B-G')
        plt.xlabel("Effective Temperature")
        plt.ylabel("Colour(mag)")
        plt.title("Colour Magnitudes versus Temperature Graph for STC7")
        plt.legend()
        plt.show()  

    return fitted_BR, fitted_GR, fitted_BG, T_eff_this


# In[28]:


# defining a function to get all the data from a star
def get_temp(R, G, B, plot=False, color_temp=False):
    
    """
    The get_temp function takes red, green and blue magnitudes and returns the average temperature of the star
    
    Parameters:
    -----------
    • R: magnitude of the star in red
    • G: magnitude of the star in green
    • B: magnitude of the star in blue
    
    Optional Parameters:
    -----------
    • color_temp: if true prints the temperature value at each colour magnitude 
    • plot: plots the color_mag vs temp graph along with the BR, GR and BG values of the star

    Returns:
    --------
    The function returns 1 value:
    • avg_temp: average temperature of the star 
    
    Usage:
    ------
    avg_temp = get_magnitudes(ref_stars_array, targ_stars_array, colour_temp=True)
    """
    BRstar = B - R
    GRstar = G - R
    BGstar = B - G
    
    fitted_BR, fitted_GR, fitted_BG, T_eff_func = fitted_vals()
    
    # finding the minimum temperature
    posBR = np.min(T_eff_func[np.where(fitted_BR < BRstar)])
    posGR = np.min(T_eff_func[np.where(fitted_GR < GRstar)])
    posBG = np.min(T_eff_func[np.where(fitted_BG < BGstar)])
    
    
    if(color_temp):
        print("BR value: ", posBR,"K")
        print("GR value: ", posGR,"K")
        print("BG value: ", posBG,"K")
    
    if plot==True:
         # plot B-R curve, horizontal line for BR and vertical line for temperature
        plt.plot(T_eff_func, fitted_BR, color='blue', label='B-R')
        plt.axhline(BRstar, color='blue', linestyle='--')
        plt.axvline(posBR, color='blue', ls='dashdot')

        # plot G-R curve, horizontal line for GR and vertical line for temperature
        plt.plot(T_eff_func, fitted_GR, color='green', label='G-R')
        plt.axhline(GRstar, color='green', linestyle='--')
        plt.axvline(posGR, color='green', ls='dashdot')

        # plot B-G curve, horizontal line for BG and vertical line for temperature
        plt.plot(T_eff_func, fitted_BG, color='purple', label='B-G')
        plt.axhline(BGstar, color='purple', linestyle='--')
        plt.axvline(posBG, color='purple', ls='dashdot')

        plt.xlabel("Effective Temperature(K)")
        plt.ylabel("Colour(mag)")

        plt.title("Graph to find the temperature of star")
        plt.legend()
        
        
    
    avg_temperature = (posBR+posGR+posBG)/3
    
    return avg_temperature


# ## get_abs_mag (relative magnitude to absolute magnitude conversion)

# In[30]:


def get_abs_mag(r, g, b, num):
    
    """
    The get_abs_mags function takes red, green and blue magnitudes and returns absolute magnitudes in RGB of the star

    
    Parameters:
    -----------
    • R: magnitude of the star in red
    • G: magnitude of the star in green
    • B: magnitude of the star in blue
    • num: HR number of the star

    Returns:
    --------
    The function returns 3 values(in this order):
    • abs_r: absolute magnitude in the red filter 
    • abs_g: absolute magnitude in the green filter 
    • abs_b: absolute magnitude in the red filter 
    
    Usage:
    ------
    abs_r, abs_g, abs_b = get_abs_mag(red, green, blue, num)
    """
    
    matching_hrs = np.where(hr==num)[0]
    
    hr_index = matching_hrs[0]
    
    parallax = plx[hr_index]
    
    d = 1000/parallax
    
    abs_mag_r = r - np.log10((d/10)**2)
    abs_mag_g = g - np.log10((d/10)**2)
    abs_mag_b = b - np.log10((d/10)**2)
    
    return abs_mag_r, abs_mag_g, abs_mag_b


# ## get_std_mags (getting standard magnitudes from stc magnitudes)

# In[31]:


def f_l(X, M):
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    x4 = X[3]
    return M[0]*x1 + M[1]*x2 + M[2]*x3 + M[3]*x4 + M[4]

def f(X, M):
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    
    return M[0]*x1 + M[1]*x2 + M[2]*x3 + M[3]


# In[34]:


def get_std_mags(r, g, b, l):
    
    """
    The get_std_mags function takes red, green, blue and lum magnitudes for STC7 and returns standard magnitudes in RGB of the star

    
    Parameters:
    -----------
    • R: magnitude of the star in red
    • G: magnitude of the star in green
    • B: magnitude of the star in blue
    • num: HR number of the star 

    Returns:
    --------
    The function returns 3 values(in this order):
    • std_r: standard magnitude in the red filter 
    • std_g: standard magnitude in the green filter 
    • std_b: standard magnitude in the red filter 
    
    Usage:
    ------
    std_r, std_g, std_b = get_abs_mag(red, green, blue, num)
    """
 
    pars_r = np.array([0.70903393,  0.18504481, -0.04698287,  0.15311282, -0.01430194])
    pars_g = np.array([0.10710746,  0.72279424, -0.00248302,  0.17273757, -0.0162348])
    pars_b = np.array([-0.08742958,  0.63625232,  0.57443408, -0.12261884, -0.02879737])

    mags = np.array([r, g, b, l])

    std_r = f_l(mags, pars_r)
    std_g = f_l(mags, pars_g)
    std_b = f_l(mags, pars_b)

    return std_r, std_g, std_b

