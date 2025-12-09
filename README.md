# QM_MoP

This repository contains code to calculate the Method of Planes (MoP) stress in the Atomic Simulation Environment (ASE) using the MACE machine learning potential. Please see the linked preprint for details of why this work is interesting 
http://arxiv.org/abs/2509.16257


The Method of Planes (MoP) is a pressure measurement used in molecular dynamics (MD) from B. D. Todd, Denis J. Evans, and Peter J. Daivis: “Pressure tensor for inhomogeneous fluids”, Phys. Rev. E 52, 1627 (1995) [see e.g. LAMMPS documentation](https://docs.lammps.org/compute_stress_mop.html). This simply ends up being force divided by area but has the useful property that it is the only pressure which can be linked to the control volume balance equations, making it valid arbitarily far from equilibirum. The attached code and associated manuscript show this is valid for MACE, and more generally machine learning potentials which use graph networks, extending the work of [Langer et al (2023)](https://arxiv.org/abs/2303.14434) to local forms of pressure.

The validaity of this approach is demonstrated by showing the MoP pressure respects static equilibirum near a liquid-solid interface (a test which the virial or IK1 form fails),

<img src="https://github.com/edwardsmith999/QM_MoP/blob/main/figures/stress_in_channel.png" width="400">

It also shows conservation in an arbitary control volume between two planes (a test for validity that can be used in any system)m 
for momentum, which validates the measured stress is exactly the one that is changing the momentum in a system

<img src="https://github.com/edwardsmith999/QM_MoP/blob/main/figures/CV_stress_trace.png" width="400">

as well as energy, which validates the form of heat flux is exactly the one responsible for changing the energy in a volume

<img src="https://github.com/edwardsmith999/QM_MoP/blob/main/figures/CV_energy_trace.png" width="400">

