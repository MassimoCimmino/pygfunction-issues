# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from matplotlib.ticker import AutoMinorLocator
from scipy import pi

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

    # Pipe dimensions
    r_out = 0.0211      # Pipe outer radius (m)
    r_in = 0.0147       # Pipe inner radius (m)
    D_s = 0.052         # Shank spacing (m)
    epsilon = 1.0e-6    # Pipe roughness (m)

    # Pipe positions
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_pipes = [(-D_s, 0.), (D_s, 0.)]

    # Ground properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)
    k_s = 2.0           # Ground thermal conductivity (W/m.K)

    # Grout properties
    k_g = 1.0           # Grout thermal conductivity (W/m.K)

    # Pipe properties
    k_p = 0.4           # Pipe thermal conductivity (W/m.K)

    # Fluid properties
    m_flow_borehole = 0.25  # Total fluid mass flow rate per borehole (kg/s)
    # The fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    cp_f = fluid.cp     # Fluid specific isobaric heat capacity (J/kg.K)
    rho_f = fluid.rho   # Fluid density (kg/m3)
    mu_f = fluid.mu     # Fluid dynamic viscosity (kg/m.s)
    k_f = fluid.k       # Fluid thermal conductivity (W/m.K)

    # g-Function calculation options
    nSegments = 48
    nSegments2 = 8
    nSegments3 = nSegments2
    factor = 2.5
    segment_ratios2 = expanding_segments_factor(nSegments2, factor=factor)
    min_length = 0.02
    segment_ratios3 = expanding_segments_minlength(nSegments3, min_length=min_length)
    options = {'nSegments':nSegments, 'disp':True}
    options2 = {'nSegments':nSegments2, 'segment_ratios':segment_ratios2, 'disp':True}
    options3 = {'nSegments':nSegments3, 'segment_ratios':segment_ratios3, 'disp':True}
    method='equivalent'
    method2='equivalent'
    method3=method2

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 25                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    # Field of 6x4 (n=24) boreholes
    N_1 = 10
    N_2 = 10
    boreField = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)
    nBoreholes = len(boreField)

    # -------------------------------------------------------------------------
    # Initialize pipe model
    # -------------------------------------------------------------------------

    # Pipe thermal resistance
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_in, r_out, k_p)
    # Fluid to inner pipe wall thermal resistance (Single U-tube)
    m_flow_pipe = m_flow_borehole
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe, r_in, mu_f, rho_f, k_f, cp_f, epsilon)
    R_f = 1.0/(h_f*2*pi*r_in)

    # Single U-tube, same for all boreholes in the bore field
    UTubes = []
    for borehole in boreField:
        SingleUTube = gt.pipes.SingleUTube(pos_pipes, r_in, r_out,
                                           borehole, k_s, k_g, R_f + R_p)
        UTubes.append(SingleUTube)
    m_flow_network = m_flow_borehole*nBoreholes
    network = gt.networks.Network(
        boreField, UTubes, m_flow_network=m_flow_network, cp_f=cp_f,
        nSegments=nSegments)

    # -------------------------------------------------------------------------
    # Evaluate the g-functions for the borefield
    # -------------------------------------------------------------------------

    # Calculate the g-function for uniform heat extraction rate
    gfunc_uniform_Q = gt.gfunction.gFunction(
        boreField, alpha, time=time, method=method, boundary_condition='UHTR', options=options)
    gfunc_uniform_Q2 = gt.gfunction.gFunction(
        boreField, alpha, time=time, method=method2, boundary_condition='UHTR', options=options2)
    gfunc_uniform_Q3 = gt.gfunction.gFunction(
        boreField, alpha, time=time, method=method3, boundary_condition='UHTR', options=options3)

    # Calculate the g-function for uniform borehole wall temperature
    gfunc_uniform_T = gt.gfunction.gFunction(
        boreField, alpha, time=time, method=method, boundary_condition='UBWT', options=options)
    gfunc_uniform_T2 = gt.gfunction.gFunction(
        boreField, alpha, time=time, method=method2, boundary_condition='UBWT', options=options2)
    gfunc_uniform_T3 = gt.gfunction.gFunction(
        boreField, alpha, time=time, method=method3, boundary_condition='UBWT', options=options3)

    # Calculate the g-function for equal inlet fluid temperature
    gfunc_equal_Tf_in = gt.gfunction.gFunction(
        network, alpha, time=time, method=method, boundary_condition='MIFT', options=options)
    gfunc_equal_Tf_in2 = gt.gfunction.gFunction(
        network, alpha, time=time, method=method2, boundary_condition='MIFT', options=options2)
    gfunc_equal_Tf_in3 = gt.gfunction.gFunction(
        network, alpha, time=time, method=method3, boundary_condition='MIFT', options=options3)

    # -------------------------------------------------------------------------
    # Plot g-functions
    # -------------------------------------------------------------------------

    ax = gfunc_uniform_Q.visualize_g_function().axes[0]
    ax.plot(np.log(time/ts), gfunc_uniform_T.gFunc)
    ax.plot(np.log(time/ts), gfunc_equal_Tf_in.gFunc)
    ax.plot(np.log(time/ts), gfunc_uniform_Q2.gFunc, 'ko')
    ax.plot(np.log(time/ts), gfunc_uniform_Q3.gFunc, 'k--')
    ax.legend(['Uniform heat extraction rate : nSegments={}'.format(nSegments),
               'Uniform borehole wall temperature : nSegments={}'.format(nSegments),
               'Equal inlet temperature : nSegments={}'.format(nSegments),
               '"Eskilson" : nSegments={}, factor={:.3f}'.format(nSegments2, factor),
               '"Cook" : nSegments={}, min_length={}'.format(nSegments3, min_length)])
    ax.plot(np.log(time/ts), gfunc_uniform_T2.gFunc, 'ko')
    ax.plot(np.log(time/ts), gfunc_uniform_T3.gFunc, 'k--')
    ax.plot(np.log(time/ts), gfunc_equal_Tf_in2.gFunc, 'ko')
    ax.plot(np.log(time/ts), gfunc_equal_Tf_in3.gFunc, 'k--')
    plt.tight_layout()

    return


def expanding_segments_factor(nSegments, factor=np.sqrt(2)):
    if is_even(nSegments):
        nz = int(nSegments / 2)
        minlength = 0.5 * (factor - 1) / (factor**nz - 1)
        dz = [factor**i * minlength for i in range(nz)]
        segment_ratios = np.concatenate((dz, dz[::-1]))
    else:
        nz = int((nSegments - 1) / 2)
        minlength = 1 / (2 * (factor**nz - 1) / (factor - 1) + factor**nz)
        dz = [factor**i * minlength for i in range(nz)]
        segment_ratios = np.concatenate((dz, np.array([factor**nz]) * minlength, dz[::-1]))
    return segment_ratios


def expanding_segments_minlength(nSegments, min_length=0.05):
    if is_even(nSegments):
        nz = int(nSegments / 2)
        coefs = np.zeros(nz+1)
        coefs[0] = 1 - 2 * min_length
        coefs[1] = -1
        coefs[-1] = 2 * min_length
        roots = poly.Polynomial(coefs).roots()
        factor = max(np.real(roots))
        dz = [factor**i * min_length for i in range(nz)]
        segment_ratios = np.concatenate((dz, dz[::-1]))
    else:
        nz = int((nSegments - 1) / 2)
        coefs = np.zeros(nz+2)
        coefs[0] = 1 - 2 * min_length
        coefs[1] = -1
        coefs[-2] = min_length
        coefs[-1] = min_length
        roots = poly.Polynomial(coefs).roots()
        factor = max(np.real(roots))
        dz = [factor**i * min_length for i in range(nz)]
        segment_ratios = np.concatenate((dz, np.array([factor**nz]) * min_length, dz[::-1]))
    return segment_ratios


def is_even(num):
    return not(num & 0x1)


# Main function
if __name__ == '__main__':
    main()
