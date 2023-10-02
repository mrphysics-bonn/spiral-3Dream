#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multislice Version for Dream (mit STEAM Block Pulsen) with acceleration
"""

import math
import numpy as np
import datetime
from pathlib import Path  # create directories
import os
#import shutil

import ismrmrd

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.make_arbitrary_grad import make_arbitrary_grad

import pulseq_helper as ph
from prot import create_hdr

#import matplotlib.pyplot as plt

# def gre_refscan(seq, prot=None, system=Opts(), params=None):

# %% Parameters
"""
PyPulseq units (SI): 
time:       [s] (not [ms] as in documentation)
spatial:    [m]
gradients:  [Hz/m] (gamma*T/m)
grad area:  [1/m]
flip angle: [rad]
"""

# General
seq_dest = "sim"    # save parameter: 'None': save nothing, 'scanner': save seq & prot in seq/prot folder & in exchange folder -> muss noch unten definiert werden 'sim': save seq in sim folder
seq_name = 'dream_cart_multi_acc'    # sequence/protocol filename

# STEAM preparation
# first rf pulse:
flip_angle_ste = 60 # flip angle of first excitation pulse of the STEAM prep-sequence [°]
rf_dur_ste = 0.4 # RF duration [ms]

# Sequence - Contrast and Geometry
fov = 200       # field of view [mm] NB: fov bei STE und FID gleich (old:230)
TR = 10       # repetition time [ms]
TE_ste = 4        # echo time [ms] NB: T2* Anteil ist vollkommen rephasiert
TE_fid = 6     #[ms]
res = 5      # in plane resolution [mm]
slice_res = 5       # slice thickness [mm] (old:2)
slices = 40         # number of slices (old:31)

# times of Steam prep:
time_scheme = 1 # 1: TS = TE_ste, 2: TS = TE_ste + TE_fid , 3: TS = TE_fid - TE_vste; hier immer: STE/VSTE first
if time_scheme == 1:        # zeit zwischen den ersten beiden pulsen in der STEAM prep-seq [ms]
    TS = TE_ste
elif time_scheme == 2:
    TS = TE_ste + TE_fid
elif time_scheme == 3:
    TS = TE_fid - TE_ste

TD = 13 # zeit zwischen den ersten puls der steam prep sequence und dem anregungspuls der gre sequenz [ms]
TM = TD - TS # mixing time: zeit zwischen zweiten und dritten puls klein gegenüber T1-Relaxationszeit [ms]

# RF - GRE
flip_angle = 6        # flip angle of excitation pulse [°] NB: kleinerer flipwinkel gewählt
rf_dur = 2         # RF duration [ms]
tbp_exc = 4         # time bandwidth product excitation pulse

prepscans = 0        # number of dummy preparation scans (bei STEAM notwendig?)

# ADC
readout_bw = 800 # readout bandwidth [Hz], readout_bw is in Hz/Px (bw/Nx)

# Acceleration
R = 2 # acceleration factor

# Gradients
max_slew = 160       # maximum slewrate [T/m/s] (system limit)
max_grad = 35 # maximum gradient amplitude [mT/m] (system limit) - used also for diffusion gradients

# %%

# Set System limits
rf_dead_time = 100e-6  # lead time before rf can be applied
rf_ringdown_time = 30e-6 # coil hold time (20e-6) + frequency reset time (10e-6)
system = Opts(max_grad=max_grad, grad_unit='mT/m', max_slew=max_slew,slew_unit='T/m/s', rf_dead_time=rf_dead_time, rf_ringdown_time=rf_ringdown_time)

# convert parameters to Pulseq units
rf_dur_ste *= 1e-3

fov *= 1e-3 #[m]
res *= 1e-3 #[m]
TR *= 1e-3  # [s]
TE_ste *= 1e-3  # [s]
TE_fid *= 1e-3  # [s]
slice_res *= 1e-3  # [m]

TS *= 1e-3  # [s]
TD *= 1e-3  # [s]
TM *= 1e-3  # [s]

rf_dur *= 1e-3  # [s]

# standard rotation matrix for Pulseq sequence from logical to patient coordinate system
rotmat = -1*np.eye(3)

# %% RF

rf, gz, gz_reph, rf_del = make_sinc_pulse(flip_angle=flip_angle * math.pi / 180, duration=rf_dur, slice_thickness=slice_res,
                                          apodization=0.5, time_bw_product=tbp_exc, system=system, return_gz=True, return_delay=True)     # rf_delay entspricht der Dauer des schichtselektionsgradienten

# %% Calculate readout gradient and ADC parameters

delta_k = 1 / fov
Nx = Ny = int(fov/res+0.5) # +0.5, damit int() die 155 und nicht 154 nimmt
samples = 2*Nx

gm_ste_flat_time_us = int(1e6/readout_bw) # readout_bw is in Hz/Px      1/bw_pixel = Tacq
gm_ste_dwelltime_us = gm_ste_flat_time_us / samples  # delta t = Tacq/nsamples
gm_ste_flat_time = round(1e-6*gm_ste_dwelltime_us*samples, 5)

gm_fid_flat_time_us = int(1e6/readout_bw) # readout_bw is in Hz/Px      1/bw_pixel = Tacq
gm_fid_dwelltime_us = gm_fid_flat_time_us / samples  # delta t = Tacq/nsamples
gm_fid_flat_time = round(1e-6*gm_fid_dwelltime_us*samples, 5)

delta_TE = TE_fid - TE_ste #[s]
x = delta_TE - gm_ste_flat_time/2 - gm_fid_flat_time/2 #[s]

# %% Gradients

# Gm_ste, Gm_mitte und Gm_fid "normal" und einzelnd bestimmen:
gm_ste = make_trapezoid(channel='x', flat_area=Nx*delta_k, flat_time=gm_ste_flat_time, system=system) # readout gradient for ste
gm_mitte = make_trapezoid(channel='x', amplitude=gm_ste.amplitude, delay=0, flat_time=x, system=system) # readout gradient between ste and fid
gm_fid = make_trapezoid(channel='x', flat_area=Nx*delta_k, flat_time=gm_fid_flat_time, system=system) # readout gradient for fid

# jeweils eine Rampe von Gm_ste und Gm_fid und beide Rampen von Gm_mitte abcutten:
gm_ste_values = ph.waveform_from_seqblock(gm_ste)
ramp_wert_index = 0
for element in range(np.size(gm_ste_values)):
    if (gm_ste_values[element] > gm_ste_values[element+1]):
        ramp_wert_index = element
        break
gm_ste_cut_values = gm_ste_values[:ramp_wert_index+1]

gm_fid_values = ph.waveform_from_seqblock(gm_fid)
ramp_wert_index = 0
for element in range(np.size(gm_fid_values)):
    if (gm_fid_values[element] == gm_fid_values[element+1]):
        ramp_wert_index = element
        break
gm_fid_cut_values = gm_fid_values[ramp_wert_index:]

gm_mitte_values = ph.waveform_from_seqblock(gm_mitte)
ramp_wertUP_index = 0
for element in range(np.size(gm_mitte_values)):
    if (gm_mitte_values[element] == gm_mitte_values[element+1]):
        ramp_wertUP_index = element
        break
ramp_wertDOWN_index = 0
for element in range(np.size(gm_mitte_values)):
    if (gm_mitte_values[element] > gm_mitte_values[element+1]):
        ramp_wertDOWN_index = element
        break
gm_mitte_cut_values = gm_mitte_values[ramp_wertUP_index:ramp_wertDOWN_index+1]

# gm_ste_cut und gm_mitte_cut zu einem Gradienten Gm_1 zusammenfügen (erster Teil von Gm):
gm_1_values = np.append(gm_ste_cut_values,gm_mitte_cut_values)
gm_1 = make_arbitrary_grad(channel='x', waveform=gm_1_values, system=system)

# gm_fid_cut als Gradient Gm_2 (zweiter Teil von Gm):  
gm_2 = make_arbitrary_grad(channel='x', waveform=gm_fid_cut_values, system=system)

# Prephaser Gm1 erstellen:
A2 = gm_ste.area/2 #[1/m]
A1 = delta_TE * gm_ste.amplitude #[1/m]

gm1_area = -A1 - A2 #[1/m]
amp_gm1, ftop_gm1, ramp_gm1 = ph.trap_from_area(gm1_area, system)
gm1 = make_trapezoid(channel='x', system=system, amplitude=amp_gm1, duration=2*ramp_gm1+ftop_gm1, rise_time=ramp_gm1) # prephaser Gm1

# Phasenkodierung:
phase_areas = (np.arange(Ny) - Ny / 2) * delta_k #[1/m]
if R != 0:
    phase_areas = phase_areas[::R]

# %% reduce slew rate of spoilers to avoid stimulation

gx_spoil = make_trapezoid(channel='x', area=2 * Nx *delta_k, system=system, max_slew=120*system.gamma)
gz_spoil = make_trapezoid(channel='z', area=4 / slice_res,system=system, max_slew=120*system.gamma)

# %% ADC

adc_ste = make_adc(num_samples=samples, dwell=1e-6*gm_ste_dwelltime_us, delay=gm_ste.rise_time, system=system)
adc_fid = make_adc(num_samples=samples, dwell=1e-6*gm_fid_dwelltime_us, system=system)

# %% RF spoiling

rf_spoiling_inc = 117
rf_phase = 0
rf_inc = 0

#%% STEAM preparation

# first rf pulse (without gz rephaser)
rf_ste, rf_ste_delay = make_block_pulse(flip_angle=flip_angle_ste*math.pi/180, duration=rf_dur_ste, return_delay=True,system=system, use='excitation')

# dephaser Gm2 in Steam prep sequence
if (time_scheme == 1 or time_scheme == 2):        # zeit zwischen den ersten beiden pulsen in der STEAM prep-seq [ms]
    gm2_area = -A1 #[1/m]
elif time_scheme == 3:
    gm2_area = A1 #[1/m]
amp_gm2, ftop_gm2, ramp_gm2 = ph.trap_from_area(gm2_area, system)
gm2 = make_trapezoid(channel='x', system=system, amplitude=amp_gm2, duration=2*ramp_gm2+ftop_gm2, rise_time=ramp_gm2) # dephaser Gm2

# spoiler (after second rf pulse)
spoiler_area = gz.flat_area - gz.area/2 # 2x moment under excitation pulse
gx_spoil_ste = make_trapezoid(channel='x', area=spoiler_area, system=system, max_slew=120*system.gamma)
gy_spoil_ste = make_trapezoid(channel='y', area=spoiler_area, system=system, max_slew=120*system.gamma)
gz_spoil_ste = make_trapezoid(channel='z', area=spoiler_area, system=system, max_slew=120*system.gamma)


# Delays
x_ste = calc_duration(rf_ste) - rf_ste.dead_time
a_ste = rf.dead_time + x_ste/2
b_ste = rf_ste_delay.delay - a_ste

# take minimum TS rounded up to .1 ms
min_TS = b_ste + calc_duration(gm2) + a_ste
# wenn min_TS>TS wird eine Fehlermeldung ausgegeben
if min_TS > TS:
    min_TS = ph.round_up_to_raster(min_TS, decimals=5)*1e3
    raise ValueError('TS zu klein gewählt, muss größer als {} ms sein'.format(min_TS))
delay_TS = TS - min_TS
ts_delay = make_delay(d=delay_TS)

# take minimum TM rounded up to .1 ms
min_TM = b_ste + calc_duration(gz_spoil_ste, gx_spoil_ste, gy_spoil_ste) + gz.rise_time + gz.flat_time/2
# wenn min_TM>TM wird eine Fehlermeldung ausgegeben
if min_TM > TM:
    min_TM = ph.round_up_to_raster(min_TM, decimals=5)*1e3
    raise ValueError('TM zu klein gewählt, muss größer als {} ms sein'.format(min_TM)) 
delay_TM = TM - min_TM
tm_delay = make_delay(d=delay_TM)

# %% delay calculation (der gre sequenz --> TE ist von TS abhängig)

# take minimum TE rounded up to .1 ms
min_TE_ste = gz.fall_time + gz.flat_time / 2 + calc_duration(gm1) + calc_duration(gm_ste) / 2    # gibt die größere ganze Zahl aus
# wenn min_TE_ste>TE_ste wird eine Fehlermeldung ausgegeben
if min_TE_ste > TE_ste:
    min_TE_ste = ph.round_up_to_raster(min_TE_ste, decimals=5)*1e3
    raise ValueError('TE_ste zu klein gewählt, muss größer als {} ms sein'.format(min_TE_ste)) 
delay_TE_ste = round(TE_ste - min_TE_ste,5)
te_ste_delay = make_delay(d=delay_TE_ste)

# take minimum TR rounded up to .1 ms
min_TR = calc_duration(gm1,gz_reph) + calc_duration(gz) + calc_duration(gm_1,adc_ste) + calc_duration(gm_2,adc_fid) + delay_TE_ste + calc_duration(gx_spoil, gz_spoil)
# wenn min_TR>TR wird eine Fehlermeldung ausgegeben
if min_TR > TR:
    min_TR = ph.round_up_to_raster(min_TR, decimals=5)*1e3
    raise ValueError('TR zu klein gewählt, muss größer als {} ms sein'.format(min_TR)) 
delay_TR = TR - min_TR
tr_delay = make_delay(d=delay_TR)

# %% create protocol & write header (xml header)

date = datetime.date.today().strftime('%Y%m%d')
filename = date + '_' + seq_name    # seq_name oben definiert

if seq_dest == 'scanner':

    # create new directory if needed
    Path("../Protocols/"+date).mkdir(parents=True, exist_ok=True)
    filepath = "../Protocols/"+date +"/{}.h5".format(filename)       # protocol file name

    # set up protocol file and create header
    if os.path.exists(filepath):
        raise ValueError("Protocol name already exists. Choose different name")
    # ismrmrd file mit vordefinierten filepath erstellen
    prot = ismrmrd.Dataset(filepath)
    hdr = ismrmrd.xsd.ismrmrdHeader()   # xml-header erstellen

    params_hdr = {"trajtype": "cartesian", "fov": fov*1e3, "res": res*1e3, "slices": slices, "slice_res": slice_res*1e3, "nintl": 1, "avg": 1,"nsegments": 1, "dwelltime": 0, "ncontrast":2}     # Parameter für den header in ein dictionary legen
    # leeren xml header mit den parametern füllen
    create_hdr(hdr, params_hdr)
    # gefüllten xml-header in das ismrmrd protocol file einfügen
    prot.write_xml_header(hdr.toxml('utf-8'))
else:
    prot = None

# %% build sequence

seq = Sequence()

# for the DEFINITIONS keywords in the pypulseq-file:
seq.set_definition("Name", filename) # protocol name is saved in Siemens header for FIRE reco
seq.set_definition("FOV", [fov, fov, slice_res]) # for FOV positioning
#seq.set_definition("Slice_Thickness", "%f" % (slice_res*slices))?      
             
if slices % 2 == 1:
    slc = 0
else:
    slc = 1
for s in range(slices):
    if s == int(slices/2+0.5):
        if slices % 2 == 1:
            slc = 1
        else:
            slc = 0
    rf.freq_offset = gz.amplitude * slice_res * (slc - (slices - 1) / 2)     # frequenz-offset des rf pulses

    # prepscans (for steady state)
    for d in range(prepscans):
        """
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        seq.add_block(rf, gz, rf_del)
        gy_pre = make_trapezoid(channel='y', area=phase_areas[0], duration=1.4e-3, system=system)
        seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(te_delay)
        seq.add_block(gx)
        gy_pre.amplitude = -gy_pre.amplitude
        seq.add_block(tr_delay, gx_spoil, gy_pre, gz_spoil)
        """
    
    # STEAM preparation
    seq.add_block(rf_ste,rf_ste_delay)
    seq.add_block(gm2)
    seq.add_block(ts_delay)
    seq.add_block(rf_ste,rf_ste_delay)
    seq.add_block(gz_spoil_ste, gx_spoil_ste, gy_spoil_ste)
    seq.add_block(tm_delay)
    
    
    # imaging scans
    for i in range(np.size(phase_areas)):
        #rf spoiling
        # rf.phase_offset = rf_phase / 180 * np.pi
        # adc_ste.phase_offset = rf_phase / 180 * np.pi
        # adc_fid.phase_offset = rf_phase / 180 * np.pi
        # rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        # rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        # phasenkodierung
        seq.add_block(rf, gz, rf_del)
        gy_pre = make_trapezoid(channel='y', area=phase_areas[i], duration=calc_duration(gm1), system=system) # hat dieselbe dauer wie gx_pre       
        seq.add_block(gm1, gy_pre, gz_reph)
        seq.add_block(te_ste_delay)
        seq.add_block(gm_1, adc_ste)
        seq.add_block(gm_2, adc_fid)
        gy_pre.amplitude = -gy_pre.amplitude
        seq.add_block(gx_spoil, gy_pre, gz_spoil)
        seq.add_block(tr_delay)

        # acquisition data header pro phasenkodierschritt erstellen
        for n in range(2):
            if prot is not None:
                acq = ismrmrd.Acquisition()     # leeren acquisitionsheader erstellen
                acq.idx.kspace_encode_step_1 = i    # kspace counter
                acq.idx.kspace_encode_step_2 = 0 # only 2D atm -> zählt die k-raum schritte in kz bei der 3D-Bildgebung
                acq.idx.slice = slc   # angabe der schichtnummer
                acq.idx.contrast = n
                acq.phase_dir[:] = rotmat[:, 0]
                acq.read_dir[:] = rotmat[:, 1]
                acq.slice_dir[:] = rotmat[:, 2]
                # wenn wir uns im letzten phasenkodierschritt einer schicht befinden... (range() geht von 0 bis Ny-1)
                if i == np.size(phase_areas)-1:
                    acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE) # angabe im akquisitionsheader, dass wir uns im letzten phasenkodierschritt der jeweiligen schicht befinden
                prot.append_acquisition(acq) # akquisitionsheader in das ismrmrd protocol file einfügen
        

    slc += 2  # acquire every 2nd slice, afterwards fill slices inbetween

if seq_dest=='scanner':
    prot.close()

# %% save sequence and write sequence & protocol to scanner exchange folder

if seq_dest == "sim":   # save to simulation folder, dafür muss die Virtual Box aktiviert sein
    seq.write('/mnt/pulseq/external.seq')

elif seq_dest=='scanner':
    prot.close()
    Path("../Pulseq_sequences/"+date).mkdir(parents=True, exist_ok=True) # create new directory if needed
    seq.write("../Pulseq_sequences/"+date+"/{}.seq".format(filename)) # save seq
    #seq.write("/mnt/mrdata/pulseq_exchange/{}.seq".format(filename)) # seq to scanner

    # prot to scanner
    #shutil.copyfile("../Protocols/"+date+"/{}.h5".format(filename), "/mnt/mrdata/pulseq_exchange/{}.h5".format(filename))
