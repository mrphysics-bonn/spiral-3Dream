""" This function adds a 2D GRE reference scan to a sequence for calculation of sensitivity maps

    multislice

"""

import math
import numpy as np

import ismrmrd

from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts

import pulseq_helper as ph

def gre_refscan(seq, prot=None, system=Opts(), params=None):

    # decrease slew rate a bit
    save_slew = system.max_slew
    system.max_slew = 130 * system.gamma

    # default parameters
    params_def = {"fov":210e-3, "res":3e-3, "flip_angle":12, "rf_dur":1e-3, "tbp": 2, "slices":1, "slice_res":2e-3, "readout_bw": 600, "dist_fac":0 , "half":None}
    for key in params:
        params_def[key] = params[key]
    params = params_def
    
    # RF
    rf, gz, gz_reph, rf_del = make_sinc_pulse(flip_angle=params["flip_angle"] * math.pi / 180, duration=params["rf_dur"], slice_thickness=params["slice_res"],
                                apodization=0.5, time_bw_product=params["tbp"], system=system, return_gz=True, return_delay=True)
    # Calculate readout gradient and ADC parameters
    delta_k = 1 / params["fov"]
    Nx = Ny = int(params["fov"]/params["res"]+0.5)
    #Ny = Ny//2
    samples = 2*Nx
    gx_flat_time_us = int(1e6/params["readout_bw"]) # readout_bw is in Hz/Px
    dwelltime_us = gx_flat_time_us / samples
    gx_flat_time = round(1e-6*dwelltime_us*samples, 5)

    # Gradients
    gx = make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=gx_flat_time, system=system)
    amp_gx_pre, ftop_gx_pre, ramp_gx_pre= ph.trap_from_area(-gx.area / 2, system)
    gx_pre = make_trapezoid(channel='x', system=system, amplitude=amp_gx_pre, duration=2*ramp_gx_pre+ftop_gx_pre, rise_time=ramp_gx_pre)
    phase_areas = (np.arange(Ny) - Ny / 2) * delta_k

    # Gradient spoiler
    amp_gx_spoil, ftop_gx_spoil, ramp_gx_spoil= ph.trap_from_area(4 / params["res"], system, slewrate=120)
    gx_spoil = make_trapezoid(channel='x', system=system, amplitude=amp_gx_spoil, duration=2*ramp_gx_spoil+ftop_gx_spoil, rise_time=ramp_gx_spoil)
    amp_gz_spoil, ftop_gz_spoil, ramp_gz_spoil= ph.trap_from_area(4 / params["slice_res"], system, slewrate=120)
    gz_spoil = make_trapezoid(channel='z', system=system, amplitude=amp_gz_spoil, duration=2*ramp_gz_spoil+ftop_gz_spoil, rise_time=ramp_gz_spoil)

    # take minimum TE rounded up to .1 ms
    min_TE = np.ceil((gz.fall_time + gz.flat_time / 2 + calc_duration(gx_pre,gz_reph) + calc_duration(gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
    TE = ph.round_up_to_raster(min_TE, decimals=4)
    delay_TE = round(TE-min_TE,5)

    # take minimum TR rounded up to .1 ms
    min_TR = calc_duration(gx_pre,gz_reph) + calc_duration(gz) + calc_duration(gx) + delay_TE + calc_duration(gx_spoil, gz_spoil)
    TR = ph.round_up_to_raster(min_TR, decimals=4)
    delay_TR = round(TR - min_TR,5)
    print("TR=",TR)

    # ADC with 2x oversampling
    adc = make_adc(num_samples=samples, dwell=1e-6*dwelltime_us, delay=gx.rise_time, system=system)
    
    # RF spoiling
    rf_spoiling_inc = 117
    rf_phase = 0
    rf_inc = 0

    # build sequence
    prepscans = 5 # number of dummy preparation scans

    if params["slices"]%2 == 1:
        slc = 0
    else:
        slc = 1
    for s in range(params["slices"]):
        if s==int(params["slices"]/2+0.5):
            if params["half"]:
                break
            if params["slices"]%2 == 1:
                slc = 1
            else:
                slc = 0
        rf.freq_offset = gz.amplitude * params["slice_res"] * (1+params["dist_fac"]) * (slc - (params["slices"] - 1) / 2)

        # prepscans
        for d in range(prepscans):
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            seq.add_block(rf, gz, rf_del)
            gy_pre = make_trapezoid(channel='y', area=phase_areas[Ny//2], duration=calc_duration(gx_pre), system=system)
            seq.add_block(gx_pre, gy_pre, gz_reph)
            seq.add_block(make_delay(delay_TE))
            seq.add_block(gx)
            gy_pre.amplitude = -gy_pre.amplitude
            seq.add_block(gx_spoil, gy_pre, gz_spoil)
            seq.add_block(make_delay(delay_TR))

        # imaging scans
        for i in range(Ny):
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            seq.add_block(rf, gz, rf_del)
            gy_pre = make_trapezoid(channel='y', area=phase_areas[i], duration=calc_duration(gx_pre), system=system)
            seq.add_block(gx_pre, gy_pre, gz_reph)
            seq.add_block(make_delay(delay_TE))
            seq.add_block(gx, adc)
            gy_pre.amplitude = -gy_pre.amplitude
            seq.add_block(gx_spoil, gy_pre, gz_spoil)
            seq.add_block(make_delay(delay_TR))

            if prot is not None:
                acq = ismrmrd.Acquisition()
                acq.idx.kspace_encode_step_1 = i
                acq.idx.kspace_encode_step_2 = 0 # only 2D atm
                acq.idx.slice = slc
                acq.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
                if i == Ny-1:
                    acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                prot.append_acquisition(acq)
                
        slc += 2 # acquire every 2nd slice, afterwards fill slices inbetween

    delay_end = make_delay(d=5)
    seq.add_block(delay_end)
    system.max_slew = save_slew
    print("nach refscan:",seq.duration()[0])