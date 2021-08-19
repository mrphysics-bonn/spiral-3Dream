#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions for specific phase encoding orders
"""
import numpy as np

def UlamSpiral(fov,Nz,Ny,comp=False):
    """ Calculates the components and areas of the gradients for the Ulam Spiral two phase encoding order
    Parameters
    ----------
    - fov : FOV in [m]
    - Nz : matrix size in z direction
    - Ny : matrix size in y direction
    - comp: determines whether the components are returned or not; default=False
    
    Returns
    -------
    - gz_area_comp: steps in z phase encoding direction (without delta_k)
    - gy_area_comp: steps in y phase encoding direction (without delta_k)
    - gz_area: Areas in z phase encoding direction
    - gy_area: Areas in y phase encoding direction

    """
    delta_k = 1/fov
    
    # first encoding step
    gz_area_comp = np.zeros(1)
    for i in range(1,Nz+1):
        step1 = 1
        step0 = 0
        rep = i
        if (i%2 != 0) and (i!=Nz):
            step1 *= -1
        elif (i==Nz):
            rep -= 1
        repetitions1 = np.repeat(step1,rep)
        gz_area_comp = np.append(gz_area_comp,repetitions1)
        if i!=Nz:
            repetitions0 = np.repeat(step0,rep)
            gz_area_comp = np.append(gz_area_comp,repetitions0)
    
    gz_area_comp = gz_area_comp.cumsum()
    gz_area = gz_area_comp * delta_k
    
    
    # second encoding step
    gy_area_comp = np.zeros(1)
    for i in range(1,Ny):
        step0 = 0
        step1 = 1
        rep = i
        if (i%2 != 0):
            step1 *= -1
        repetitions0 = np.repeat(step0,rep)
        gy_area_comp = np.append(gy_area_comp,repetitions0)
        repetitions1 = np.repeat(step1,rep)
        gy_area_comp = np.append(gy_area_comp,repetitions1)
    zeros = np.zeros([np.size(gz_area_comp)-np.size(gy_area_comp)])
    gy_area_comp = np.concatenate((gy_area_comp,zeros))
    
    gy_area_comp = gy_area_comp.cumsum()
    gy_area = gy_area_comp * delta_k
    
    if comp:
        return gz_area_comp, gy_area_comp, gz_area, gy_area
    else:
        return gz_area, gy_area



def CentricOrder(N,fov,comp=False):
    """ Calculates the components and areas of the gradients for the centric phase encoding order

    :param N: Number of phase encoding steps
    :param fov: FoV in the specific direction [m]
    :param comp: if True, returns the components; default:False
    :returns: phase_enc_steps (if comp=True), phase_areas
    """
    phase_enc_steps = np.zeros(N)
    n = -1
    for i in range (1,N,2):
        phase_enc_steps[i] = n
        n -= 1
    n = 1
    for i in range (2,N,2):
        phase_enc_steps[i] = n
        n += 1
    
    delta_k = 1/fov #[1/m]
    phase_areas = phase_enc_steps * delta_k
    
    if comp:
        return phase_enc_steps, phase_areas
    else:
        return phase_areas









