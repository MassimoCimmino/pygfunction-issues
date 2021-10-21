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
    filename_reference_12 = '../../data/rectangular-UBWT/similarities_12_segments.json'
    filename_reference_64 = '../../data/rectangular-UBWT/similarities_64_segments.json'
    filename_equal_12 = 'similarities_12_segments.json'
    filename_unequal_8 = 'similarities_8_segments.json'

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
    N_min = 1
    N_max = 20

    nBoreholes = [data_reference_12['{}_by_{}'.format(N_1, N_2)]['nBoreholes'] for N_1 in range(N_min, N_max+1) for N_2 in range(N_1, N_max+1)]
    calc_time_reference_12 = [data_reference_12['{}_by_{}'.format(N_1, N_2)]['calc_time'] for N_1 in range(N_min, N_max+1) for N_2 in range(N_1, N_max+1)]
    calc_time_equal_12 = [data_equal_12['{}_by_{}'.format(N_1, N_2)]['calc_time'] for N_1 in range(N_min, N_max+1) for N_2 in range(N_1, N_max+1)]
    calc_time_unequal_8 = [data_unequal_8['{}_by_{}'.format(N_1, N_2)]['calc_time'] for N_1 in range(N_min, N_max+1) for N_2 in range(N_1, N_max+1)]

    fig = gt.utilities._initialize_figure()
    fig.suptitle('Rectangular fields up to {} by {} : Calculation time'.format(N_max, N_max))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'Number of boreholes')
    ax1.set_ylabel(r'Calculation time [sec]')
    gt.utilities._format_axes(ax1)
    ax1.loglog(nBoreholes, calc_time_reference_12, 'ko', label=data_reference_12['label'] + ' ({}) : nSegments={}'.format(data_reference_12['commit'], 12))
    ax1.loglog(nBoreholes, calc_time_equal_12, 'o', label=data_equal_12['label'] + ' ({}) : nSegments={}'.format(data_equal_12['commit'], 12))
    ax1.loglog(nBoreholes, calc_time_unequal_8, 'o', label=data_unequal_8['label'] + ' ({}) : nSegments={}'.format(data_unequal_8['commit'], 8))
    ax1.legend()
    fig.tight_layout()

    # RMSE
    N_min = 1
    N_max = 20

    nBoreholes = [data_reference_12['{}_by_{}'.format(N_1, N_2)]['nBoreholes'] for N_1 in range(N_min, N_max+1) for N_2 in range(N_1, N_max+1)]
    RMSE_12_12 = [RMSE(data_reference_12['{}_by_{}'.format(N_1, N_2)]['gfunc'], data_equal_12['{}_by_{}'.format(N_1, N_2)]['gfunc']) for N_1 in range(N_min, N_max+1) for N_2 in range(N_1, N_max+1)]

    fig = gt.utilities._initialize_figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'Number of boreholes')
    ax1.set_ylabel(r'RMSE')
    fig.suptitle('Rectangular fields up to {} by {} : Comparison of 12 segments g-functions'.format(N_max, N_max))
    gt.utilities._format_axes(ax1)
    ax1.loglog(nBoreholes, RMSE_12_12, 'o', label=data_equal_12['label'] + ' ({}) : nSegments={}'.format(data_equal_12['commit'], 12))
    ax1.legend()
    fig.tight_layout()

    # RMSE
    N_min = 1
    N_max = 10

    nBoreholes = [data_reference_64['{}_by_{}'.format(N_1, N_2)]['nBoreholes'] for N_1 in range(N_min, N_max+1) for N_2 in range(N_1, N_max+1)]
    RMSE_12_64 = [RMSE(data_reference_64['{}_by_{}'.format(N_1, N_2)]['gfunc'], data_equal_12['{}_by_{}'.format(N_1, N_2)]['gfunc']) for N_1 in range(N_min, N_max+1) for N_2 in range(N_1, N_max+1)]
    RMSE_8_64 = [RMSE(data_reference_64['{}_by_{}'.format(N_1, N_2)]['gfunc'], data_unequal_8['{}_by_{}'.format(N_1, N_2)]['gfunc']) for N_1 in range(N_min, N_max+1) for N_2 in range(N_1, N_max+1)]

    fig = gt.utilities._initialize_figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'Number of boreholes')
    ax1.set_ylabel(r'RMSE')
    fig.suptitle('Rectangular fields up to {} by {} : Comparison against 64 segments g-functions'.format(N_max, N_max))
    gt.utilities._format_axes(ax1)
    ax1.loglog(nBoreholes, RMSE_12_64, 'o', label=data_equal_12['label'] + ' ({}) : nSegments={}'.format(data_equal_12['commit'], 12))
    ax1.loglog(nBoreholes, RMSE_8_64, 'o', label=data_unequal_8['label'] + ' ({}) : nSegments={}'.format(data_unequal_8['commit'], 8))
    ax1.legend()
    fig.tight_layout()

    return


def RMSE(reference, predicted):
    rmse = np.linalg.norm(np.array(predicted) - np.array(reference)) / len(reference)
    return rmse


# Main function
if __name__ == '__main__':
    out = main()