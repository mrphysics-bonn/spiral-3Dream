#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3DREAM cartesian version (parameters from threedream by P.Ehses) for all three timing schemes (but with Ulam Spiral two phase encoding order)
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
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.make_arbitrary_grad import make_arbitrary_grad

import pulseq_helper as ph
from prot import create_hdr
import phase_enc_helper as pe

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
seq_name = '3DREAM_cart_US_ts3'    # sequence/protocol filename

# STEAM preparation
# first rf pulse:
flip_angle_ste = 60 # flip angle of first excitation pulse of the STEAM prep-sequence [°]
rf_dur_ste = 0.4 # RF duration [ms]

# Sequence - Contrast and Geometry
fov = 200       # field of view [mm] NB: fov bei STE und FID gleich, fov is in all three dimensions the same
TR = 3.41       # repetition time [ms]
TE_ste = 1.17        # echo time [ms] NB: T2* Anteil ist vollkommen rephasiert
TE_fid = 2.25     #[ms]
res = 5      # in plane resolution [mm]
slice_res = 5       # slice thickness [mm]
Nz = 40         # number of slices

# times of Steam prep:
time_scheme = 3 # 1: TS = TE_ste, 2: TS = TE_ste + TE_fid , 3: TS = TE_fid - TE_vste; hier immer: STE/VSTE first
if time_scheme == 1:        # zeit zwischen den ersten beiden pulsen in der STEAM prep-seq [ms]
    TS = TE_ste
elif time_scheme == 2:
    TS = TE_ste + TE_fid
elif time_scheme == 3:
    TS = TE_fid - TE_ste

TM = 9.94 # mixing time: zeit zwischen zweiten und dritten puls klein gegenüber T1-Relaxationszeit [ms]
TD = TM + TS # zeit zwischen den ersten puls der steam prep sequence und dem anregungspuls der gre sequenz [ms]

# RF - GRE
flip_angle = 6       # flip angle of excitation pulse [°] NB: kleinerer flipwinkel gewählt
rf_dur = 0.2         # RF duration [ms]

# ADC
readout_bw = 1000 # readout bandwidth [Hz], readout_bw is in Hz/Px (bw/Nx)

# Preperation scans
prepscans = 2        # number of dummy preparation scans (bei STEAM notwendig?)

# Gradients
max_slew = 160       # maximum slewrate [T/m/s] (system limit)
max_grad = 35 # maximum gradient amplitude [mT/m] (system limit) - used also for diffusion gradients

# T1 estimate for global filter approach
t1 = 784e-3   # [s] - approximately Gufi Phantom at 7T, brain = 2s

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

rf,rf_delay= make_block_pulse(flip_angle=flip_angle*math.pi/180, duration=rf_dur,return_delay=True,use='excitation', system=system)

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
# using cartesian ulam spiral two phase encoding order to measure the contrast at the beginning (kspace center first)
gz_area_comp, gy_area_comp, gz_area,gy_area = pe.UlamSpiral(fov, Nz, Ny, comp=True) #[1/m]

# %% reduce slew rate of spoilers to avoid stimulation

gx_spoil_area = 6.6526e-06*system.gamma #[1/m]
amp_gx_spoil, ftop_gx_spoil, ramp_gx_spoil = ph.trap_from_area(gx_spoil_area, system)
gx_spoil = make_trapezoid(channel='x', system=system, amplitude=amp_gx_spoil, duration=2*ramp_gx_spoil+ftop_gx_spoil, rise_time=ramp_gx_spoil)

gy_spoil = make_trapezoid(channel='y', area=1 /(res), system=system, max_slew=120*system.gamma)
gz_spoil = make_trapezoid(channel='z', area=1 /(res),system=system, max_slew=120*system.gamma)

# %% ADC

adc_ste = make_adc(num_samples=samples, dwell=1e-6*gm_ste_dwelltime_us, delay=gm_ste.rise_time, system=system)
adc_fid = make_adc(num_samples=samples, dwell=1e-6*gm_fid_dwelltime_us, system=system)

# %% delay calculation (der gre sequenz --> TE ist von TS abhängig)

x_img = calc_duration(rf) - rf.dead_time
a_img = rf.dead_time + x_img/2
b_img = rf_delay.delay - a_img

# take minimum TE rounded up to .1 ms
min_TE_ste = b_img + calc_duration(gm1) + calc_duration(gm_ste) / 2    # gibt die größere ganze Zahl aus
# wenn min_TE_ste>TE_ste wird eine Fehlermeldung ausgegeben
if min_TE_ste > TE_ste:
    min_TE_ste = ph.round_up_to_raster(min_TE_ste, decimals=5)*1e3
    raise ValueError('TE_ste zu klein gewählt, muss größer als {} ms sein'.format(min_TE_ste)) 
delay_TE_ste = TE_ste - min_TE_ste
te_ste_delay = make_delay(d=delay_TE_ste)

# take minimum TR rounded up to .1 ms
min_TR = calc_duration(rf_delay) + calc_duration(gm1) + calc_duration(gm_1,adc_ste) + calc_duration(gm_2,adc_fid) + delay_TE_ste + calc_duration(gx_spoil)
# wenn min_TR>TR wird eine Fehlermeldung ausgegeben
if min_TR > TR:
    min_TR = ph.round_up_to_raster(min_TR, decimals=5)*1e3
    raise ValueError('TR zu klein gewählt, muss größer als {} ms sein'.format(min_TR)) 
delay_TR = TR - min_TR
tr_delay = make_delay(d=delay_TR)

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
spoiler_area = 30e-06*system.gamma # Area aus 'threedream'
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
min_TM = b_ste + calc_duration(gz_spoil_ste, gx_spoil_ste, gy_spoil_ste) + a_img + prepscans*TR
# wenn min_TM>TM wird eine Fehlermeldung ausgegeben
if min_TM > TM:
    min_TM = ph.round_up_to_raster(min_TM, decimals=5)*1e3
    raise ValueError('TM zu klein gewählt, muss größer als {} ms sein'.format(min_TM)) 
delay_TM = TM - min_TM
tm_delay = make_delay(d=delay_TM)

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

    params_hdr = {"trajtype": "cartesian", "fov": fov*1e3, "res": res*1e3, "slices": 1, "slice_res": slice_res, "npartitions": Nz, "ncontrast": 2}    # Parameter für den header in ein dictionary legen
    # leeren xml header mit den parametern füllen
    create_hdr(hdr, params_hdr)
    # gefüllten xml-header in das ismrmrd protocol file einfügen
    prot.write_xml_header(hdr.toxml('utf-8'))
    
    dream = np.array([0,TR,flip_angle_ste,flip_angle,prepscans,t1])
    prot.append_array('dream', dream)
else:
    prot = None

# %% build sequence

seq = Sequence()

# for the DEFINITIONS keywords in the pypulseq-file:
seq.set_definition("Name", filename) # protocol name is saved in Siemens header for FIRE reco
seq.set_definition("FOV", [fov, fov, fov]) # for FOV positioning
    
# STEAM preparation
seq.add_block(rf_ste,rf_ste_delay)
seq.add_block(gm2)
seq.add_block(ts_delay)
seq.add_block(rf_ste,rf_ste_delay)
seq.add_block(gz_spoil_ste, gx_spoil_ste, gy_spoil_ste)
seq.add_block(tm_delay)

# prepscans (for steady state)
for d in range(prepscans):
    rf.phase_offset = rf_phase / 180 * np.pi
    adc_ste.phase_offset = rf_phase / 180 * np.pi
    adc_fid.phase_offset = rf_phase / 180 * np.pi
    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

    seq.add_block(rf, rf_delay)
    gz_pre = make_trapezoid(channel='z', area=gz_area[0], duration=calc_duration(gm1), system=system)
    gy_pre = make_trapezoid(channel='y', area=gy_area[0], duration=calc_duration(gm1), system=system) # hat dieselbe dauer wie gx_pre       
    seq.add_block(gm1, gz_pre, gy_pre)
    seq.add_block(te_ste_delay)
    seq.add_block(gm_1)
    seq.add_block(gm_2)
    seq.add_block(gx_spoil, gy_spoil, gz_spoil)
    seq.add_block(tr_delay)


# imaging scans  
for i in range(np.size(gz_area)):
    #rf spoiling
    rf.phase_offset = rf_phase / 180 * np.pi
    adc_ste.phase_offset = rf_phase / 180 * np.pi
    adc_fid.phase_offset = rf_phase / 180 * np.pi
    rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
    rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
    
    # phase encoding and readout
    seq.add_block(rf, rf_delay)
    gz_pre = make_trapezoid(channel='z', area=gz_area[i], duration=calc_duration(gm1), system=system)
    gy_pre = make_trapezoid(channel='y', area=gy_area[i], duration=calc_duration(gm1), system=system) # hat dieselbe dauer wie gx_pre       
    seq.add_block(gm1, gz_pre, gy_pre)
    seq.add_block(te_ste_delay)
    seq.add_block(gm_1, adc_ste)
    seq.add_block(gm_2, adc_fid)
    seq.add_block(gx_spoil, gy_spoil, gz_spoil)
    seq.add_block(tr_delay)
    
    # acquisition data header pro phasenkodierschritt erstellen
    for n in range(2):
        if prot is not None:
            acq = ismrmrd.Acquisition()     # leeren acquisitionsheader erstellen
            acq.idx.kspace_encode_step_1 = int(Ny/2+gy_area_comp[i])     # kspace counter (k-raum schritte in ky)
            acq.idx.kspace_encode_step_2 = int(Nz/2+gz_area_comp[i]) # only 2D atm -> zählt die k-raum schritte in kz bei der 3D-Bildgebung
            acq.idx.slice = 0
            acq.idx.contrast = n # zur Unterscheidung vom STE und FID signal
            acq.phase_dir[:] = rotmat[:, 0]
            acq.read_dir[:] = rotmat[:, 1]
            acq.slice_dir[:] = rotmat[:, 2]
            # wenn wir uns im letzten phasenkodierschritt einer schicht befinden... (range() geht von 0 bis Ny-1)
            if i == np.size(gz_area)-1:
                acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE) # angabe im akquisitionsheader, dass wir uns im letzten phasenkodierschritt der jeweiligen schicht befinden
            prot.append_acquisition(acq) # akquisitionsheader in das ismrmrd protocol file einfügen

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
