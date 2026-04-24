from __future__ import print_function

from pathlib import Path
import importlib.resources as resources

import numpy as np
from scipy.ndimage import median_filter, zoom, gaussian_filter1d, binary_dilation, binary_erosion
from scipy.signal import fftconvolve
# from .estimate_sky import estimate_background, get_interpolated_cubic
# from ..procedures.destriper import destriper, stack128, stack64, get_stripe_pattern64
# from ..igrins_libs.resource_helper_igrins import ResourceHelper
from astropy.io import fits
#import glob
import copy

from numpy.polynomial import Polynomial

from scipy.ndimage import median_filter, gaussian_filter


#Use a series of median filters to isolate the sky lines while ignoring everything else
def isolate_sky_lines(data):
	mask = data > 4.5
	mask = binary_erosion(mask, iterations=1)
	data[mask] = np.nan
	data = data - np.nanmedian(data, 0) #Remove pattern
	data = data - np.nanmedian(data, 1)[:,np.newaxis]
	data = median_filter(data - median_filter(data, [1, 35]), [7, 1]) #median filters to try to isolate sky lines
	data[mask] = np.nan #Reapply mask to keep masked pixels masked after applying median filters
	data = data - np.nanmedian(data, 0) #Remove pattern
	std = np.nanstd(data) #Zero out negative residuals from science signal
	data[data < -std] = np.nan
	return data


def get_date_and_band(obsset):
	date, band = obsset.get_resource_spec()
	return date, band

def get_exptime(obsset):
	return obsset.recipe_entry['exptime']


def roll_along_axis(array_to_correct, correction, axis=0): #Apply flexure correction by numpy rolling along an axis and averaging between two rolled arrays to account for sub-pixel shifts
    axis = int(axis)
    integer_correction = np.round(correction) #grab whole number component of correction
    fractional_correction = correction - float(integer_correction) #Grab fractional component of correction (remainder after grabbing whole number out)
    rolled_array =  np.roll(array_to_correct, int(integer_correction), axis=axis) #role array the number of pixels matching the integer correction
    if fractional_correction > 0.: #For a positive correction
        rolled_array_plus_one = np.roll(array_to_correct, int(integer_correction+1), axis=axis) #Roll array an extra one pixel to the right
    else: #For a negative correction
        rolled_array_plus_one = np.roll(array_to_correct, int(integer_correction-1), axis=axis) #Roll array an extra one pixel to the left
    corrected_array = rolled_array*(1.0-np.abs(fractional_correction)) + rolled_array_plus_one*np.abs(fractional_correction) #interpolate over the fraction of a pixel
    return corrected_array


def cross_correlate(reference, data, zoom_amount=1000, maximum_pixel_search=10):
	masked_reference = copy.deepcopy(reference)
	masked_data = copy.deepcopy(data)
	mask = (reference == 0) | np.isnan(reference) | (data == 0) | np.isnan(data)

	masked_reference[mask] = np.nan
	masked_data[mask] = np.nan
	interp_order = 1
	fx1 = zoom(np.nansum(masked_reference, 0), zoom_amount, order=interp_order) #Zoom and collapse into 1D
	fx2 = zoom(np.nansum(masked_data, 0), zoom_amount, order=interp_order) #Zoom and collapse into 1D
	fx1[np.isnan(fx1)] = 0 #Zero out remaining nan pixels
	fx2[np.isnan(fx2)] = 0 #Zero out remaining nan pixels
	fft_result_x = fftconvolve(fx1, fx2[::-1]) #Perform FFIT cross correlation only in x and y
	delta_sub_pixels = 0.5*maximum_pixel_search*zoom_amount #Cut FFT cross-correlation result to be within a maximum number of pixels from zero offset, this cuts out possible extraneous minima screwing up the maximum in the FFT result characterizing the true offset
	x1 = int((fft_result_x.shape[0]/2) - delta_sub_pixels)
	x2 = int((fft_result_x.shape[0]/2) + delta_sub_pixels)
	fft_sub_result_x = fft_result_x[x1:x2]

	#Mask out large trends
	n = len(fft_sub_result_x)
	fit_x_array = np.arange(n)
	fft_sub_result_mask = (fit_x_array < n*0.25) | (fit_x_array > n*0.75)
	pfit = Polynomial.fit(fit_x_array[fft_sub_result_mask], fft_sub_result_x[fft_sub_result_mask], 2)
	fft_sub_result_x = fft_sub_result_x - pfit(fit_x_array)

	find_shift_from_maximum_x = np.unravel_index(np.argmax(fft_sub_result_x), fft_sub_result_x.shape[0]) #Find pixels with strongest correlation
	fft_dx_result = (find_shift_from_maximum_x[0] -  (fft_sub_result_x.shape[0]/2))/zoom_amount #Calcualte the offset from the pixels with the strongest correlation

	if abs(fft_dx_result) == maximum_pixel_search/2: #If flexure measured has hit the minimum or maximum shift checked, something went wrong, throw an exception
		raise Exception("Flexure correction failed to find a good cross correlation between sky frame and this exposure.  Check sky frame is good and and sky emission lines are visible in this exposure.")


	return fft_dx_result #Returns the difference in x pixels and y pixels between the reference and data frames


#Check 
def check_telluric_shift(obsset, datalist):
    date, band = get_date_and_band(obsset) #Grab date and band we are working in
    xr = [400,1900]
    zoom_amount = 1000.0
    interp_order=1
    maximum_pixel_search = 10
    if band == 'H':
        orders = [119, 120, 121]
    elif band == 'K':
        orders = [72, 73, 74]
    #filename = glob.glob('calib/primary/'+date+'/SKY_SDC'+band+'_'+date+'*order_map.fits')[0] #Load order map
    #order_map = fits.getdata(filename)
    order_map = obsset.load_resource_for("ordermap")[0].data
    filtered_data1 = datalist[0] - np.nanmedian(datalist[0], 0) - np.nanmedian(datalist[0], 1)[:,np.newaxis] #Cross correlate each dataframe with the first data frame in the list
    filtered_data1 -= median_filter(filtered_data1, [35,1])
    for j in range(1, len(datalist)):
        filtered_data2 =  datalist[j] - np.nanmedian(datalist[j], 0) - np.nanmedian(datalist[j], 1)[:,np.newaxis]
        filtered_data2 -= median_filter(filtered_data2, [35,1])
        dx_results = []

        for i in orders:
            cut_data1 = copy.deepcopy(filtered_data1) #Collapse data in an order into 1D
            cut_data2 = copy.deepcopy(filtered_data2)
            cut_data1[~(order_map == i)] = np.nan
            cut_data2[~(order_map == i)] = np.nan
            collapsed_data1 = np.nansum(cut_data1[:,xr[0]:xr[1]], 0)
            collapsed_data2 = np.nansum(cut_data2[:,xr[0]:xr[1]], 0)
            collapsed_data1 /= np.nansum(collapsed_data1) #Normalize both frames
            collapsed_data2 /= np.nansum(collapsed_data2)
            fx1 = zoom(collapsed_data1, zoom_amount, order=interp_order) #Zoom and collapse into 1D
            fx2 = zoom(collapsed_data2, zoom_amount, order=interp_order) #Zoom and collapse into 1D
            fx1[np.isnan(fx1) | (fx1 < -np.std(fx1))] = 0 #Zero out remaining nan pixels
            fx2[np.isnan(fx2) | (fx2 < -np.std(fx2))] = 0 #Zero out remaining nan pixels
            fft_result_x = fftconvolve(fx1, fx2[::-1]) #Perform FFIT cross correlation only in x and y
            delta_sub_pixels = 0.5*maximum_pixel_search*zoom_amount #Cut FFT cross-correlation result to be within a maximum number of pixels from zero offset, this cuts out possible extraneous minima screwing up the maximum in the FFT result characterizing the true offset
            x1 = int((fft_result_x.shape[0]/2) - delta_sub_pixels)
            x2 = int((fft_result_x.shape[0]/2) + delta_sub_pixels)
            fft_sub_result_x = fft_result_x[x1:x2]
            find_shift_from_maximum_x = np.unravel_index(np.argmax(fft_sub_result_x), fft_sub_result_x.shape[0]) #Find pixels with strongest correlation
            fft_dx_result = (find_shift_from_maximum_x[0] -  (fft_sub_result_x.shape[0]/2))/zoom_amount #Calc
            dx_results.append(fft_dx_result)
        dx = np.nanmedian(dx_results)

        #if abs(dx) > 0.0: #threshold for telluric shift
        if True:
            outdata_path = obsset.rs.storage.get_section('OUTDATA_PATH')
            with open(outdata_path+"/telluric_shift_"+band+".csv", "a") as f: #Output flexure corrections to the textfile flexure.txt 
                f.write(str(obsset.obsids[0])+', '+str(obsset.obsids[j])+', '+str(dx)+'\n')


def get_band(adlist):
    ad = adlist[0]
    band = ad[0].band() # phu["BAND"]

    return band


def estimate_flexure(adlist, ad_sky, exptime):
    flexure_corrected_data = [] #Create a list to store the flexure corrected data

    refframe = ad_sky[0].FLEXCORR

    # FIXME we need a way to load packaged calibration data

	# if exptime >= 20.0: #Load mask to isolate sky lines , for long exposures estimate flexure for each frame seperately
    # master_cal_dir = obsset.rs.master_ref_loader.config.master_cal_dir
	#mask = (fits.getdata(master_cal_dir+'/'+band+'-band_sky_mask.fits') == 1.0)

    band = get_band(adlist)
    mskname = f'{band}-band_sky_mask_igrins2.fits'
    mask_path = resources.files('igrinsdr.igrins.lookups.ref_data').joinpath(mskname)
    with resources.as_file(mask_path) as p:
        mask = (fits.getdata(p) == 1.0)
    refframe[~mask] = np.nan

	#for dataframe in data:
    # for i in range(len(data)):
    #     dataframe = data[i]
    for ad in adlist:
        dataframe = ad[0].data
        cleaned_dataframe = isolate_sky_lines(dataframe/exptime) #Apply median filters to isolate sky lines from other signal and normalize by exposure time
        #if obsset.obsids[i] == 99:
        #	breakpoint()
        cleaned_dataframe[~mask] = np.nan #Apply mask to isolate sky lines on detector
        #dx, dy = cross_correlate(refframe, cleaned_dataframe) #Estimate delta-x and delta-y difference in pixels between the reference and data frames
        dx = cross_correlate(refframe, cleaned_dataframe) #Estimate delta-x and delta-y difference in pixels between the reference and data frames

        #shifted_dataframe = roll_along_axis(dataframe, dy, axis=0)
        #shifted_dataframe = roll_along_axis(shifted_dataframe, dx, axis=1) #Apply flexure correction
        shifted_dataframe = roll_along_axis(dataframe, dx, axis=1)

        ad[0].data = shifted_dataframe
        # flexure_corrected_data.append(shifted_dataframe)

        #print('dx =', dx, 'dy =', dy)

        # if False:
        #     outdata_path = obsset.rs.storage.get_section('OUTDATA_PATH')
        #     with open(outdata_path+"/flexure_"+band+".csv", "a") as f: #Output flexure corrections to the textfile flexure.txt 
        #         f.write(band+', '+str(obsset.obsids[i])+', '+str(dx)+'\n')

    return adlist # flexure_corrected_data

def estimate_flexure_orig(obsset, data, exptime):
	#exptime = get_exptime(obsset)
    date, band = get_date_and_band(obsset) #Grab date and band we are working in
    flexure_corrected_data = [] #Create a list to store the flexure corrected data

    refframe = copy.deepcopy(obsset.load_resource_for("flexcorr")[0].data)

	# if exptime >= 20.0: #Load mask to isolate sky lines , for long exposures estimate flexure for each frame seperately
    master_cal_dir = obsset.rs.master_ref_loader.config.master_cal_dir
	#mask = (fits.getdata(master_cal_dir+'/'+band+'-band_sky_mask.fits') == 1.0)
    mask = (fits.getdata(master_cal_dir+'/'+band+'-band_sky_mask_igrins2.fits') == 1.0)
    refframe[~mask] = np.nan
	#for dataframe in data:
    for i in range(len(data)):
        dataframe = data[i]
        cleaned_dataframe = isolate_sky_lines(dataframe/exptime) #Apply median filters to isolate sky lines from other signal and normalize by exposure time
        #if obsset.obsids[i] == 99:
        #	breakpoint()		
        cleaned_dataframe[~mask] = np.nan #Apply mask to isolate sky lines on detector
        #dx, dy = cross_correlate(refframe, cleaned_dataframe) #Estimate delta-x and delta-y difference in pixels between the reference and data frames
        dx = cross_correlate(refframe, cleaned_dataframe) #Estimate delta-x and delta-y difference in pixels between the reference and data frames

        #shifted_dataframe = roll_along_axis(dataframe, dy, axis=0)			
        #shifted_dataframe = roll_along_axis(shifted_dataframe, dx, axis=1) #Apply flexure correction
        shifted_dataframe = roll_along_axis(dataframe, dx, axis=1)	

        flexure_corrected_data.append(shifted_dataframe)
        #print('dx =', dx, 'dy =', dy)
        outdata_path = obsset.rs.storage.get_section('OUTDATA_PATH')
        with open(outdata_path+"/flexure_"+band+".csv", "a") as f: #Output flexure corrections to the textfile flexure.txt 
            f.write(band+', '+str(obsset.obsids[i])+', '+str(dx)+'\n')

    return flexure_corrected_data

def estimate_flexure_short_exposures(obsset, data_a, data_b, exptime):
    date, band = get_date_and_band(obsset) #Grab date and band we are working in
    refframe = copy.deepcopy(obsset.load_resource_for("flexcorr")[0].data)
    master_cal_dir = obsset.rs.master_ref_loader.config.master_cal_dir
    #mask = (fits.getdata(master_cal_dir+'/'+band+'-band_limited_sky_mask.fits') == 1.0)  #(note we use a more conservative mask for short exposures)
    mask = (fits.getdata(master_cal_dir+'/'+band+'-band_limited_sky_mask_igrins2.fits') == 1.0)  #(note we use a more conservative mask for short exposures)
    refframe[~mask] = np.nan
    combined_data = data_a + data_b - np.abs(data_a - data_b)
    cleaned_combined_data = isolate_sky_lines(combined_data / (exptime * len(obsset.get_obsids()))) #Apply median filters to isolate sky lines from other signal and normalize by exposure time
    cleaned_combined_data[~mask] = np.nan #Apply mask to isolate sky lines on detector	
    #dx, dy = cross_correlate(refframe, cleaned_combined_data) #Estimate delta-x and delta-y difference in pixels between the reference and data frames
    dx = cross_correlate(refframe, cleaned_combined_data) #Estimate delta-x and delta-y difference in pixels between the reference and data frames
    shifted_data_a = roll_along_axis(data_a, dx, axis=1) #Apply flexure correction
    #shifted_data_a = roll_along_axis(shifted_data_a, dy, axis=0)
    shifted_data_b = roll_along_axis(data_b, dx, axis=1) #Apply flexure correction
    #shifted_data_b = roll_along_axis(shifted_data_b, dy, axis=0)

    outdata_path = obsset.rs.storage.get_section('OUTDATA_PATH')
    with open(outdata_path +"/flexure_"+band+".csv", "a") as f: #Output flexure corrections to the textfile flexure.txt 
        #f.write(band+', '+str(obsset.obsids[0])+', '+str(dx)+', '+str(dy)+'\n')
        f.write(band+', '+str(obsset.obsids[0])+', '+str(dx)+'\n')

    return shifted_data_a, shifted_data_b

if __name__ == "__main__":
    pass
