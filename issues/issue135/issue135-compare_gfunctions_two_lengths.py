"""

"""

import json
import time as tim
import numpy as np

import pygfunction as gt

def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------
    filename_reference_12 = '../../data/similarities_12_segments_two_lengths.json'
    filename_reference_64 = '../../data/similarities_64_segments_two_lengths.json'
    filename_equal_12 = 'similarities_12_segments_two_lengths.json'
    filename_unequal_8 = 'similarities_8_segments_two_lengths.json'
    fieldname = 'two_lengths'

    with open(filename_reference_12, 'r') as outfile:
        data_reference_12 = json.load(outfile)
    with open(filename_reference_64, 'r') as outfile:
        data_reference_64 = json.load(outfile)
    with open(filename_equal_12, 'r') as outfile:
        data_equal_12 = json.load(outfile)
    with open(filename_unequal_8, 'r') as outfile:
        data_unequal_8 = json.load(outfile)

    # -------------------------------------------------------------------------
    # Compare 12 segments g-functions with reference
    # -------------------------------------------------------------------------
    # Calculation time
    nBoreholes = data_reference_12[fieldname]['nBoreholes']
    calc_time_reference_12 = data_reference_12[fieldname]['calc_time']
    calc_time_reference_64 = data_reference_64[fieldname]['calc_time']
    calc_time_equal_12 = data_equal_12[fieldname]['calc_time']
    calc_time_unequal_8 = data_unequal_8[fieldname]['calc_time']

    # g-function
    lntts = data_reference_12[fieldname]['lntts']
    gfunc_reference_12 = data_reference_12[fieldname]['gfunc']
    gfunc_reference_64 = data_reference_64[fieldname]['gfunc']
    gfunc_equal_12 = data_equal_12[fieldname]['gfunc']
    gfunc_unequal_8 = data_unequal_8[fieldname]['gfunc']

    # RMSE
    RMSE_reference_12 = RMSE(gfunc_reference_64, gfunc_reference_12)
    RMSE_equal_12 = RMSE(gfunc_reference_64, gfunc_equal_12)
    RMSE_unequal_8 = RMSE(gfunc_reference_64, gfunc_unequal_8)

    # Figure
    fig = gt.utilities._initialize_figure()
    fig.suptitle('Field of 6 by 6 boreholes with outer ring')
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'ln(t/ts)')
    ax1.set_ylabel(r'g-function')
    gt.utilities._format_axes(ax1)
    ax1.plot(lntts, gfunc_reference_12, label=data_reference_12['label'] + ' ({}) :\n nSegments={}, calc_time = {:.3f} sec, RMSE = {:.3f}'.format(data_reference_12['commit'], 12, calc_time_reference_12, RMSE_reference_12))
    ax1.plot(lntts, gfunc_equal_12, label=data_equal_12['label'] + ' ({}) :\n nSegments={}, calc_time = {:.3f} sec, RMSE = {:.3f}'.format(data_equal_12['commit'], 12, calc_time_equal_12, RMSE_equal_12))
    ax1.plot(lntts, gfunc_unequal_8, label=data_unequal_8['label'] + ' ({}) :\n nSegments={}, calc_time = {:.3f} sec, RMSE = {:.3f}'.format(data_unequal_8['commit'], 8, calc_time_unequal_8, RMSE_unequal_8))
    ax1.plot(lntts, gfunc_reference_64, 'ko', label=data_reference_64['label'] + ' ({}) :\n nSegments={}, calc_time = {:.3f} sec'.format(data_reference_64['commit'], 64, calc_time_reference_64))
    ax1.legend()
    fig.tight_layout()

    return


def RMSE(reference, predicted):
    rmse = np.linalg.norm(np.array(predicted) - np.array(reference)) / len(reference)
    return rmse


# Main function
if __name__ == '__main__':
    out = main()