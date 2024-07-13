""" This function adds a 3D GRE reference scan to a sequence for calculation of sensitivity maps
    
    Linear two phase encoding order

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

def gre_3DrefscanLinear(seq, prot=None, system=Opts(), params=None):

    # decrease slew rate a bit
    save_slew = system.max_slew
    system.max_slew = 130 * system.gamma
    
    #default parameters
    params_def = {"fov":210e-3, "flip_angle":12, "rf_dur":1e-3, "tbp": 2, "readout_bw": 1000, "ref_lines": 30, "TE": None}
    for key in params:
        params_def[key] = params[key]
    params = params_def

    # RF
    rf, gz, gz_reph, rf_del = make_sinc_pulse(flip_angle=params["flip_angle"] * math.pi / 180, duration=params["rf_dur"], slice_thickness=params["fov"],
                                apodization=0.5, time_bw_product=params["tbp"], system=system, return_gz=True, return_delay=True)

    # Calculate readout gradient and ADC parameters
    delta_k = 1 / params["fov"]
    Ny = Nz = params["ref_lines"]

    # even matrix size (+matrix size should not be bigger than imaging matrix)
    if Nz % 2 != 0: 
        Nz -= 1 
    if Ny % 2 != 0: 
        Ny -= 1 
    Nx = Ny
    samples = 2*Nx # 2x oversampling
    gx_flat_time_us = int(1e6/params["readout_bw"]) # readout_bw is in Hz/Px
    dwelltime_us = gx_flat_time_us / samples
    gx_flat_time = round(1e-6*dwelltime_us*samples, 5)

    # Gradients
    
    # readout
    gx = make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=gx_flat_time, system=system)
    amp_gx_pre, ftop_gx_pre, ramp_gx_pre= ph.trap_from_area(-gx.area / 2, system)
    gx_pre = make_trapezoid(channel='x', system=system, amplitude=amp_gx_pre, duration=2*ramp_gx_pre+ftop_gx_pre, rise_time=ramp_gx_pre)
    gx_mid = make_trapezoid(channel='x', area=-gx.area, system=system)

    # linear two phase encoding order
    gy_area    = (np.arange(Ny) - Ny / 2) * delta_k #[1/m]
    gz_area    = (np.arange(Nz) - Nz / 2) * delta_k #[1/m]
    amp_gz_pre, ftop_gz_pre, ramp_gz_pre= ph.trap_from_area(gz_area[0] + gz_reph.area, system)
    gz_pre0 = make_trapezoid(channel='z', system=system, amplitude=amp_gz_pre, duration=2*ramp_gz_pre+ftop_gz_pre, rise_time=ramp_gz_pre)

    # Gradient spoiler
    spoiler_area = 4 / (params["fov"] / Nz)
    dur = np.zeros(np.size(gz_area))   
    for i in range(np.size(gz_area)):
        area = -gz_area[i] + spoiler_area
        amp_gz_spoil, ftop_gz_spoil, ramp_gz_spoil= ph.trap_from_area(area, system, slewrate=120)
        gz_spoil = make_trapezoid(channel='z', system=system, amplitude=amp_gz_spoil, duration=2*ramp_gz_spoil+ftop_gz_spoil, rise_time=ramp_gz_spoil)
        dur[i] = calc_duration(gz_spoil)
    max_dur = dur.max()

    # take minimum TE rounded up to .1 ms or take TE from parameters
    min_TE = np.ceil((gz.fall_time + gz.flat_time / 2 + calc_duration(gz_pre0,gx_pre) + calc_duration(gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
    
    if params["TE"] is None:
        TE = ph.round_up_to_raster(min_TE, decimals=4)
        delay_TE = round(TE-min_TE,5)
        n_TE = 1
    else:
        params["TE"] = sorted(params["TE"])
        n_TE = len(params["TE"])
        d_TE = np.asarray(params["TE"][1:]) - np.asarray(params["TE"][:-1])
        delay_TE = [params["TE"][0] - min_TE]
        for d_te in d_TE:
            delay_TE.append(d_te - calc_duration(gx) - calc_duration(gx_mid))
    if np.min(delay_TE) < 0:
        raise ValueError(f"TE is too small by {1e3*abs(min(delay_TE))} ms. Increase readout bandwidth.")

    # ADC with 2x oversampling
    adc = make_adc(num_samples=samples, dwell=1e-6*dwelltime_us, delay=gx.rise_time, system=system)
    
    # RF spoiling
    rf_spoiling_inc = 117
    rf_phase = 0
    rf_inc = 0

    # build sequence
    prepscans = 5 # number of dummy preparation scans

    # prepscans
    for d in range(prepscans):
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        seq.add_block(rf, gz, rf_del)
        gz_pre = make_trapezoid(channel='z', area=gz_area[int(Nz//2)] + gz_reph.area, duration=calc_duration(gz_pre0), system=system)
        gy_pre = make_trapezoid(channel='y', area=gy_area[int(Ny//2)], duration=calc_duration(gz_pre0), system=system)
        seq.add_block(gx_pre, gy_pre, gz_pre)
        seq.add_block(make_delay(delay_TE[0]))
        seq.add_block(gx)
        for k in range(n_TE-1):
            seq.add_block(gx_mid)
            seq.add_block(make_delay(delay_TE[k+1]))
            seq.add_block(gx)
        gz_rew = make_trapezoid(channel='z', area=-gz_area[int(Nz//2)] + spoiler_area, duration=max_dur, system=system)
        gy_rew = make_trapezoid(channel='y', area=-gy_area[int(Ny//2)], duration=max_dur, system=system)
        seq.add_block(gy_rew,gz_rew)

    # imaging scans
    for j in range(np.size(gz_area)):
        gz_pre = make_trapezoid(channel='z', area=gz_area[j] + gz_reph.area, duration=calc_duration(gz_pre0), system=system) 
        for i in range(np.size(gy_area)):
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

            seq.add_block(rf, gz, rf_del)
            gy_pre = make_trapezoid(channel='y', area=gy_area[i], duration=calc_duration(gz_pre0), system=system)
            seq.add_block(gx_pre, gy_pre, gz_pre)
            seq.add_block(make_delay(delay_TE[0]))
            seq.add_block(gx, adc)
            for k in range(n_TE-1):
                seq.add_block(gx_mid)
                seq.add_block(make_delay(delay_TE[k+1]))
                seq.add_block(gx, adc)
            gz_rew = make_trapezoid(channel='z', area=-gz_area[j] + spoiler_area, duration=max_dur, system=system)
            gy_rew = make_trapezoid(channel='y', area=-gy_area[i], duration=max_dur, system=system)
            seq.add_block(gy_rew,gz_rew)

            if prot is not None:
                for k in range(n_TE):
                    acq = ismrmrd.Acquisition()
                    acq.idx.kspace_encode_step_1 = i
                    acq.idx.kspace_encode_step_2 = j
                    acq.idx.slice = 0 # 3D
                    acq.idx.contrast = k
                    acq.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
                    if j == np.size(gz_area)-1 and i == np.size(gy_area)-1 and k == n_TE-1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                    prot.append_acquisition(acq)
    
    if prot is not None and n_TE>1:
        prot.append_array("echo_times", np.asarray(params["TE"]))

    delay_end = make_delay(d=5)
    seq.add_block(delay_end)
    system.max_slew = save_slew
    print("Seq duration after refscan:",seq.duration()[0])
    