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

    # Borehole dimensions
    D = 4.0             # Borehole buried depth (m)
    H = 150.0           # Borehole length (m)
    r_b = 0.075         # Borehole radius (m)
    B = 7.5             # Borehole spacing (m)

    # Thermal properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 25                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)
    lntts = np.log(time/ts)

    # g-Function calculation options
    nSegments = 12      # Number of segments per borehole
    method = 'similarities'
    options = {'nSegments': 12, 'disp': True}
    N_min = 1
    N_max = 2

    # -------------------------------------------------------------------------
    # Configure results file
    # -------------------------------------------------------------------------
    filename = '{}_{}_segments.json'.format(method, nSegments)
    data = {}
    data['label'] = 'master'
    data['commit'] = '73cb929'
    data['author'] = 'MassimoCimmino'

    # -------------------------------------------------------------------------
    # g-Functions for all fields from N_min by N_min to N_max by N_max
    # -------------------------------------------------------------------------
    for N_1 in range(N_min, N_max+1):
        for N_2 in range(N_1, N_max+1):
            print(' Field of {} by {} boreholes '.format(N_1, N_2).center(60, '='))
            boreholes = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)
            tic = tim.time()
            gfunc = gt.gfunction.gFunction(
                boreholes, alpha, time=time, method=method, options=options)
            toc = tim.time()
            fieldname = '{}_by_{}'.format(N_1, N_2)
            data[fieldname] = {'time': time.tolist(),
                               'lntts': lntts.tolist(),
                               'N_1': N_1,
                               'N_2': N_2,
                               'nBoreholes': len(boreholes),
                               'gfunc': gfunc.gFunc.tolist(),
                               'calc_time': toc - tic}
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
            

    fig = gt.utilities._initialize_figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'Number of boreholes')
    ax1.set_ylabel(r'Calculation time [sec]')
    ax1.set_title('Rectangular fields up to {} by {}'.format(N_max, N_max))
    gt.utilities._format_axes(ax1)
    nBoreholes = [data['{}_by_{}'.format(N_1, N_2)]['nBoreholes'] for N_1 in range(N_min, N_max+1) for N_2 in range(N_1, N_max+1)]
    calc_time = [data['{}_by_{}'.format(N_1, N_2)]['calc_time'] for N_1 in range(N_min, N_max+1) for N_2 in range(N_1, N_max+1)]
    ax1.loglog(nBoreholes, calc_time, 'o',)
    ax1.legend()
    fig.tight_layout()
    return


# Main function
if __name__ == '__main__':
    out = main()