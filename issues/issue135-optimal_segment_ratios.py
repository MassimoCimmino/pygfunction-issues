# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from matplotlib.ticker import AutoMinorLocator
from scipy import pi
from scipy.optimize import minimize
from time import time as tic

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

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 25                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)

    # g-Function calculation options
    nSegments = 128     # Number of segments for the reference solution
    options = {'nSegments':nSegments, 'disp':True}
    nSegments_min = 5
    nSegments_max = 20
    nSegments_test = range(nSegments_min, nSegments_max + 1)
    factor = np.sqrt(2)
    min_length = 0.05
    factor_revised = 2.5
    min_length_revised = 0.02
    method = 'equivalent'
    boundary_condition = 'UBWT'

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    # Field of Nx4 (n=24) boreholes
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
    # Evaluate errors on the g-function (default parameters)
    # -------------------------------------------------------------------------

    gfunc_reference = gt.gfunction.gFunction(
        boreField, alpha, time=time, method=method, boundary_condition=boundary_condition, options=options)
    rmse_uniform = []
    time_uniform = []
    rmse_Eskilson = []
    time_Eskilson = []
    rmse_Cook = []
    time_Cook = []
    for n in nSegments_test:
        options_uniform = {'nSegments':n, 'disp':True}
        tic0 = tic()
        gfunc_uniform = gt.gfunction.gFunction(
            boreField, alpha, time=time, method=method, boundary_condition=boundary_condition, options=options_uniform)
        toc0 = tic()
        rmse = np.sqrt(((gfunc_uniform.gFunc - gfunc_reference.gFunc) ** 2).mean())
        rmse_uniform.append(rmse)
        time_uniform.append(toc0 - tic0)

        segment_ratios = expanding_segments_factor(n, factor=factor)
        options_Eskilson = {'nSegments':n, 'segment_ratios':segment_ratios, 'disp':True}
        tic1 = tic()
        gfunc_Eskilson = gt.gfunction.gFunction(
            boreField, alpha, time=time, method=method, boundary_condition=boundary_condition, options=options_Eskilson)
        toc1 = tic()
        rmse = np.sqrt(((gfunc_Eskilson.gFunc - gfunc_reference.gFunc) ** 2).mean())
        rmse_Eskilson.append(rmse)
        time_Eskilson.append(toc1 - tic1)

        segment_ratios = expanding_segments_minlength(n, min_length=min_length)
        options_Cook = {'nSegments':n, 'segment_ratios':segment_ratios, 'disp':True}
        tic2 = tic()
        gfunc_Cook = gt.gfunction.gFunction(
            boreField, alpha, time=time, method=method, boundary_condition=boundary_condition, options=options_Cook)
        toc2 = tic()
        rmse = np.sqrt(((gfunc_Cook.gFunc - gfunc_reference.gFunc) ** 2).mean())
        rmse_Cook.append(rmse)
        time_Cook.append(toc2 - tic2)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    # Configure figure and axes
    fig = gt.utilities._initialize_figure()

    ax1 = fig.add_subplot(211)
    # Axis labels
    ax1.set_xlabel('nSegments')
    ax1.set_ylabel('RMSE')
    gt.utilities._format_axes(ax1)
    # Draw
    ax1.plot(nSegments_test, rmse_uniform)
    ax1.plot(nSegments_test, rmse_Eskilson)
    ax1.plot(nSegments_test, rmse_Cook)

    ax2 = fig.add_subplot(212)
    # Axis labels
    ax2.set_xlabel('nSegments')
    ax2.set_ylabel('Calc. time [sec]')
    gt.utilities._format_axes(ax2)
    # Draw
    ax2.plot(nSegments_test, time_uniform)
    ax2.plot(nSegments_test, time_Eskilson)
    ax2.plot(nSegments_test, time_Cook)
    ax2.legend(['Equal ratios',
                '"Eskilson" : factor={:.3f}'.format(factor),
                '"Cook" : min_length={}'.format(min_length)])

    # Adjust to plot window
    fig.suptitle('RMSE and calculation time using "default" parameters')
    fig.tight_layout()

    # -------------------------------------------------------------------------
    # Evaluate errors on the g-function (revised parameters)
    # -------------------------------------------------------------------------

    gfunc_reference = gt.gfunction.gFunction(
        boreField, alpha, time=time, method=method, boundary_condition=boundary_condition, options=options)
    rmse_uniform = []
    time_uniform = []
    rmse_Eskilson = []
    time_Eskilson = []
    rmse_Cook = []
    time_Cook = []
    for n in nSegments_test:
        options_uniform = {'nSegments':n, 'disp':True}
        tic0 = tic()
        gfunc_uniform = gt.gfunction.gFunction(
            boreField, alpha, time=time, method=method, boundary_condition=boundary_condition, options=options_uniform)
        toc0 = tic()
        rmse = np.sqrt(((gfunc_uniform.gFunc - gfunc_reference.gFunc) ** 2).mean())
        rmse_uniform.append(rmse)
        time_uniform.append(toc0 - tic0)

        segment_ratios = expanding_segments_factor(n, factor=factor_revised)
        options_Eskilson = {'nSegments':n, 'segment_ratios':segment_ratios, 'disp':True}
        tic1 = tic()
        gfunc_Eskilson = gt.gfunction.gFunction(
            boreField, alpha, time=time, method=method, boundary_condition=boundary_condition, options=options_Eskilson)
        toc1 = tic()
        rmse = np.sqrt(((gfunc_Eskilson.gFunc - gfunc_reference.gFunc) ** 2).mean())
        rmse_Eskilson.append(rmse)
        time_Eskilson.append(toc1 - tic1)

        segment_ratios = expanding_segments_minlength(n, min_length=min_length_revised)
        options_Cook = {'nSegments':n, 'segment_ratios':segment_ratios, 'disp':True}
        tic2 = tic()
        gfunc_Cook = gt.gfunction.gFunction(
            boreField, alpha, time=time, method=method, boundary_condition=boundary_condition, options=options_Cook)
        toc2 = tic()
        rmse = np.sqrt(((gfunc_Cook.gFunc - gfunc_reference.gFunc) ** 2).mean())
        rmse_Cook.append(rmse)
        time_Cook.append(toc2 - tic2)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    # Configure figure and axes
    fig = gt.utilities._initialize_figure()

    ax1 = fig.add_subplot(211)
    # Axis labels
    ax1.set_xlabel('nSegments')
    ax1.set_ylabel('RMSE')
    gt.utilities._format_axes(ax1)
    # Draw
    ax1.plot(nSegments_test, rmse_uniform)
    ax1.plot(nSegments_test, rmse_Eskilson)
    ax1.plot(nSegments_test, rmse_Cook)

    ax2 = fig.add_subplot(212)
    # Axis labels
    ax2.set_xlabel('nSegments')
    ax2.set_ylabel('Calc. time [sec]')
    gt.utilities._format_axes(ax2)
    # Draw
    ax2.plot(nSegments_test, time_uniform)
    ax2.plot(nSegments_test, time_Eskilson)
    ax2.plot(nSegments_test, time_Cook)
    ax2.legend(['Equal ratios',
                '"Eskilson" : factor={:.3f}'.format(factor_revised),
                '"Cook" : min_length={}'.format(min_length_revised)])

    # Adjust to plot window
    fig.suptitle('RMSE and calculation time using revised parameters')
    fig.tight_layout()

    # -------------------------------------------------------------------------
    # Find optimal parameters
    # -------------------------------------------------------------------------

    rmse_Eskilson = []
    time_Eskilson = []
    factor_Eskilson = []
    min_length_Cook = []
    rmse_Cook = []
    time_Cook = []
    for n in nSegments_test:
        options = {'nSegments':n, 'disp':True}
        res = minimize(error_Eskilson,
                        factor,
                        args=(gfunc_reference, boreField, alpha, time, method, boundary_condition, options),
                        bounds=[(1.000001, None)])

        segment_ratios = expanding_segments_factor(n, factor=res.x[0])
        options_Eskilson = {'nSegments':n, 'segment_ratios':segment_ratios, 'disp':True}
        tic1 = tic()
        gfunc_Eskilson = gt.gfunction.gFunction(
            boreField, alpha, time=time, method=method, boundary_condition=boundary_condition, options=options_Eskilson)
        toc1 = tic()
        rmse = np.sqrt(((gfunc_Eskilson.gFunc - gfunc_reference.gFunc) ** 2).mean())
        rmse_Eskilson.append(rmse)
        time_Eskilson.append(toc1 - tic1)
        factor_Eskilson.append(res.x[0])

        res = minimize(error_Cook,
                        min_length,
                        args=(gfunc_reference, boreField, alpha, time, method, boundary_condition, options),
                        bounds=[(1e-6, 1/n)])

        segment_ratios = expanding_segments_minlength(n, min_length=res.x[0])
        options_Cook = {'nSegments':n, 'segment_ratios':segment_ratios, 'disp':True}
        tic2 = tic()
        gfunc_Cook = gt.gfunction.gFunction(
            boreField, alpha, time=time, method=method, boundary_condition=boundary_condition, options=options_Cook)
        toc2 = tic()
        rmse = np.sqrt(((gfunc_Cook.gFunc - gfunc_reference.gFunc) ** 2).mean())
        rmse_Cook.append(rmse)
        time_Cook.append(toc2 - tic2)
        min_length_Cook.append(res.x[0])

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    # Configure figure and axes
    fig = gt.utilities._initialize_figure()

    ax1 = fig.add_subplot(411)
    # Axis labels
    ax1.set_xlabel('nSegments')
    ax1.set_ylabel('RMSE')
    gt.utilities._format_axes(ax1)
    # Draw
    ax1.plot(nSegments_test, rmse_uniform)
    ax1.plot(nSegments_test, rmse_Eskilson)
    ax1.plot(nSegments_test, rmse_Cook)

    ax2 = fig.add_subplot(412)
    # Axis labels
    ax2.set_xlabel('nSegments')
    ax2.set_ylabel('Calc. time [sec]')
    gt.utilities._format_axes(ax2)
    # Draw
    ax2.plot(nSegments_test, time_uniform)
    ax2.plot(nSegments_test, time_Eskilson)
    ax2.plot(nSegments_test, time_Cook)
    ax2.legend(['Equal ratios',
                '"Eskilson"'.format(factor),
                '"Cook"'.format(min_length)])

    ax3 = fig.add_subplot(413)
    # Axis labels
    ax3.set_xlabel('nSegments')
    ax3.set_ylabel('factor')
    gt.utilities._format_axes(ax3)
    # Draw
    ax3.plot(nSegments_test, factor_Eskilson)

    ax4 = fig.add_subplot(414)
    # Axis labels
    ax4.set_xlabel('nSegments')
    ax4.set_ylabel('min_length')
    gt.utilities._format_axes(ax3)
    # Draw
    ax4.plot(nSegments_test, min_length_Cook)

    # Adjust to plot window
    fig.suptitle('RMSE and calculation time using optimal parameters')
    fig.tight_layout()

    return


def error_Eskilson(factor, gfunc_reference, boreField, alpha, time, method, boundary_condition, options):
    segment_ratios = expanding_segments_factor(options['nSegments'], factor=factor[0])
    print(factor[0], segment_ratios)
    options_Eskilson = {'nSegments':options['nSegments'], 'segment_ratios':segment_ratios, 'disp':options['disp']}
    gfunc_Eskilson = gt.gfunction.gFunction(
        boreField, alpha, time=time, method=method, boundary_condition=boundary_condition, options=options_Eskilson)
    rmse = np.sqrt(((gfunc_Eskilson.gFunc - gfunc_reference.gFunc) ** 2).mean())
    print(rmse)
    return rmse


def error_Cook(min_length, gfunc_reference, boreField, alpha, time, method, boundary_condition, options):
    segment_ratios = expanding_segments_minlength(options['nSegments'], min_length=min_length[0])
    print(min_length[0], segment_ratios)
    options_Cook = {'nSegments':options['nSegments'], 'segment_ratios':segment_ratios, 'disp':options['disp']}
    gfunc_Cook = gt.gfunction.gFunction(
        boreField, alpha, time=time, method=method, boundary_condition=boundary_condition, options=options_Cook)
    rmse = np.sqrt(((gfunc_Cook.gFunc - gfunc_reference.gFunc) ** 2).mean())
    print(rmse)
    return rmse


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
