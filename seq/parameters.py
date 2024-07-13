"""
4 mm protocol with B0 correction Rxy=3, Rz=2, intl=4
"""

# General
seq_dest        = "sim"     # save parameter: 'None': save nothing, 'scanner': save seq & prot in seq/prot folder & in exchange folder 'sim': save seq in sim folder
seq_name        = '3Dream_4mmiso_Rxy3_Rz2_4intl'      # sequence/protocol filename
meas_date       = None
plot            = False         # plot sequence (refscans & noisescans will be deactivated)

# Sequence - Contrast and Geometry
fov             = 200           # field of view [mm]
TR              = 6.24          # repetition time [ms]
TE_ste          = 0.78          # STE/ STE* echo time [ms]
TE_fid          = 3.65            # FID echo time [ms]
res             = 4             # in plane resolution [mm]                   
Nz              = 52            # number of partitions
averages        = 1             # number of averages

refscan         = 1             # 0: no refscan, 1: Cartesian spiral two phase encoding order, 2: linear phase encoding order
ref_lines       = 40            # number of reference lines (Nx,Ny & Nz)
b0_corr         = True          # Activate B0 correction (acquires 2-echo reference scan)
prepscans       = 0             # number of preparation/dummy scans before spiral sequence
noisescans      = 16            # number of noise scans

# STEAM prep
flip_angle_ste  = 50            # flip angle of the excitation pulses of the STEAM prep-sequence [°]
rf_dur_ste      = 0.4           # RF duration [ms]
time_scheme     = 3             # 1: TS = TE_ste, 2: TS = TE_ste + TE_fid , 3: TS = TE_fid - TE_ste; for all: STE/ STE* first
TM              = 1.56          # mixing time: time span between the 2nd alpha and 1st beta pulse -> small compared to T1 relaxation time [ms]

# RF - Imaging
flip_angle      = 5             # flip angle of excitation pulse [°]
rf_dur          = 0.2           # RF duration [ms]
tbp_exc         = 6             # time bandwidth product excitation pulse
rf_spoiling     = False         # RF spoiling

fatsat          = True          # Fat saturation pulse
fatsat_dur      = 2.1           # duration of fatsat pulse [ms]

# ADC
os_factor       = 2             # oversampling factor (automatic 2x os from Siemens is not applied)

# Gradients
max_slew        = 200           # maximum slewrate [T/m/s] (system limit)
spiral_slew     = 198           # maximum slew rate of spiral gradients - for pre Emph: set lower than max_slew
max_grad        = 55            # maximum gradient amplitude [mT/m] (system limit) - used also for diffusion gradients
max_grad_sp     = 42            # maximum gradient amplitude of spiral gradients - for pre_emph: set lower than max_grad

Nintl           = 12             # spiral interleaves
redfac          = 3             # in-plane acceleration factor
spiral_os       = 1             # spiral oversampling in center
Rz              = 2             # acceleration factor in kz direction
caipi_shift     = 2            # CAIPI-shift

# T1 estimation for global filter approach
t1              = 2             # expected T1 value [s] (780ms GUFI, 2s grey matter)


"""
3 mm protocol Rxy=3, Rz=2, intl=4
"""

# General
seq_dest        = "sim"     # save parameter: 'None': save nothing, 'scanner': save seq & prot in seq/prot folder & in exchange folder 'sim': save seq in sim folder
seq_name        = '3Dream_2p5mmiso_Rxy3_Rz2_4intl'      # sequence/protocol filename
meas_date       = "20220428"
plot            = False         # plot sequence (refscans & noisescans will be deactivated)

# Sequence - Contrast and Geometry
fov             = 200           # field of view [mm]
TR              = 7.65          # repetition time [ms]
TE_ste          = 0.86          # STE/ STE* echo time [ms]
TE_fid          = 4.35            # FID echo time [ms]
res             = 3             # in plane resolution [mm]                   
Nz              = 68            # number of partitions
averages        = 1             # number of averages

refscan         = 1             # 0: no refscan, 1: Cartesian spiral two phase encoding order, 2: linear phase encoding order
tbp_refscan     = 23            # time bandwidth product refscan pulse
res_refscan     = 5             # refscan resolution (isotropic) [mm]
dur_refscan     = 2             # refscan pulse duration [ms]
prepscans       = 0             # number of preparation/dummy scans before spiral sequence
noisescans      = 16            # number of noise scans

# STEAM prep
flip_angle_ste  = 50            # flip angle of the excitation pulses of the STEAM prep-sequence [°]
rf_dur_ste      = 0.4           # RF duration [ms]
time_scheme     = 3             # 1: TS = TE_ste, 2: TS = TE_ste + TE_fid , 3: TS = TE_fid - TE_ste; for all: STE/ STE* first
TM              = 1.72          # mixing time: time span between the 2nd alpha and 1st beta pulse -> small compared to T1 relaxation time [ms]

# RF - Imaging
flip_angle      = 5             # flip angle of excitation pulse [°]
rf_dur          = 0.2           # RF duration [ms]
tbp_exc         = 6             # time bandwidth product excitation pulse
rf_spoiling     = False         # RF spoiling

fatsat          = True          # Fat saturation pulse
fatsat_dur      = 2.1           # duration of fatsat pulse [ms]

# ADC
os_factor       = 2             # oversampling factor (automatic 2x os from Siemens is not applied)

# Gradients
max_slew        = 200           # maximum slewrate [T/m/s] (system limit)
spiral_slew     = 198           # maximum slew rate of spiral gradients - for pre Emph: set lower than max_slew
max_grad        = 55            # maximum gradient amplitude [mT/m] (system limit) - used also for diffusion gradients
max_grad_sp     = 42            # maximum gradient amplitude of spiral gradients - for pre_emph: set lower than max_grad

Nintl           = 12             # spiral interleaves
redfac          = 3             # in-plane acceleration factor
spiral_os       = 1             # spiral oversampling in center
Rz              = 2             # acceleration factor in kz direction
caipi_shift     = 2            # CAIPI-shift

# T1 estimation for global filter approach
t1              = 2             # expected T1 value [s] (780ms GUFI, 2s grey matter)
