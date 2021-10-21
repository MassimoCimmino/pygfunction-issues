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
    H1 = 150.0          # Borehole length (m)
    H2 = 85.0           # Borehole length (m)
    r_b = 0.075         # Borehole radius (m)
    B = 7.5             # Borehole spacing (m)

    # Thermal properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 25                         # Number of time steps
    time = gt.utilities.time_geometric(dt, tmax, Nt)

    # g-Function calculation options
    method = 'similarities'

    # -------------------------------------------------------------------------
    # Field of 6 by 6 boreholes with outer ring of boreholes of
    # reduced length (totalling 10 by 10 boreholes)
    # -------------------------------------------------------------------------
    boreholes = gt.boreholes.rectangle_field(6, 6, B, B, H1, D, r_b)
    boreholes1 = gt.boreholes.box_shaped_field(8, 8, B, B, H2, D, r_b)
    boreholes2 = gt.boreholes.box_shaped_field(10, 10, B, B, H2, D, r_b)
    for b in boreholes1:
        b.x = b.x - B
        b.y = b.y - B
    for b in boreholes2:
        b.x = b.x - 2*B
        b.y = b.y - 2*B
    boreholes = boreholes + boreholes1 + boreholes2
    H_mean = np.mean([b.H for b in boreholes])
    ts = H_mean**2/(9*alpha)
    lntts = np.log(time/ts)

    # -------------------------------------------------------------------------
    # Uniform segments
    # -------------------------------------------------------------------------
    nSegments = 12      # Number of segments per borehole
    options = {'nSegments': nSegments,
               'disp': True}

    filename = '{}_{}_segments_two_lengths.json'.format(method, nSegments)
    data_equal = {}
    data_equal['label'] = 'PR-136'
    data_equal['commit'] = 'd37872c'
    data_equal['author'] = 'MassimoCimmino'

    tic = tim.time()
    gfunc = gt.gfunction.gFunction(
        boreholes, alpha, time=time, method=method, options=options)
    toc = tim.time()
    fieldname = 'two_lengths'
    data_equal[fieldname] = {'time': time.tolist(),
                             'lntts': lntts.tolist(),
                             'nBoreholes': len(boreholes),
                             'gfunc': gfunc.gFunc.tolist(),
                             'calc_time': toc - tic}

    with open(filename, 'w') as outfile:
        json.dump(data_equal, outfile)

    # -------------------------------------------------------------------------
    # Non-uniform segments
    # -------------------------------------------------------------------------
    nSegments = 8      # Number of segments per borehole
    end_length_ratio = 0.02
    segment_ratios = gt.utilities.segment_ratios(
        nSegments, end_length_ratio=end_length_ratio)
    options = {'nSegments': nSegments,
               'segment_ratios': segment_ratios,
               'disp': True}

    filename = '{}_{}_segments_two_lengths.json'.format(method, nSegments)
    data_unequal = {}
    data_unequal['label'] = 'PR-136 (unequal segments)'
    data_unequal['commit'] = 'd37872c'
    data_unequal['author'] = 'MassimoCimmino'

    tic = tim.time()
    gfunc = gt.gfunction.gFunction(
        boreholes, alpha, time=time, method=method, options=options)
    toc = tim.time()
    fieldname = 'two_lengths'
    data_unequal[fieldname] = {'time': time.tolist(),
                               'lntts': lntts.tolist(),
                               'nBoreholes': len(boreholes),
                               'gfunc': gfunc.gFunc.tolist(),
                               'calc_time': toc - tic}

    with open(filename, 'w') as outfile:
        json.dump(data_unequal, outfile)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig = gt.utilities._initialize_figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'ln(t/ts)')
    ax1.set_ylabel(r'g-function')
    ax1.set_title('Field of 6 by 6 boreholes with outer ring')
    gt.utilities._format_axes(ax1)
    lntts = data_equal['two_lengths']['lntts']
    gfunc = data_equal['two_lengths']['gfunc']
    calc_time = data_equal['two_lengths']['calc_time']
    ax1.plot(lntts, gfunc, '-', label='Equal segments : calc_time = {:.3f} sec'.format(calc_time))
    lntts = data_unequal['two_lengths']['lntts']
    gfunc = data_unequal['two_lengths']['gfunc']
    calc_time = data_unequal['two_lengths']['calc_time']
    ax1.plot(lntts, gfunc, '-', label='Unequal segments : calc_time = {:.3f} sec'.format(calc_time))
    ax1.legend()
    fig.tight_layout()
    return


# Main function
if __name__ == '__main__':
    out = main()