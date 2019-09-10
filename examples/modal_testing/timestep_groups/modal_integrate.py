#!/usr/bin/env python
from __future__ import print_function
from modal_comps import *
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.api import NonlinearBlockGS, LinearBlockGS



nsteps = 10
nmodes = 2
root_name='sphere_body1'
z0 = [1.0,2.0]
m = [1.0,1.0]
c = [0.0,0.0]
k = [1.0,1.5]
dt = 1.0
amp = 1.0
omega = 0.1

time = np.linspace(0.0,dt*nsteps,nsteps+1)

prob = Problem()
model = prob.model

indeps = IndepVarComp()
indeps.add_output('z0',z0)
indeps.add_output('m',m)
indeps.add_output('c',c)
indeps.add_output('k',k)
indeps.add_output('amp',amp)
indeps.add_output('freq',omega)
indeps.add_output('time',time)
model.add_subsystem('Indeps',indeps)

for step in range(1,nsteps+1):
    step_group = Group()
    step_group.add_subsystem('modal_forces',ModalForces(nmodes=nmodes,root_name=root_name))
    step_group.add_subsystem('modal_solver',ModalStep(nmodes=nmodes,dt=dt))
    step_group.add_subsystem('modal_disps' ,ModalDisplacements(nmodes=nmodes,root_name=root_name))
    step_group.add_subsystem('forcer',HarmonicForcer(root_name=root_name))
    step_group.nonlinear_solver = NonlinearBlockGS()
    step_group.linear_solver = LinearBlockGS()
    model.add_subsystem('Step'+str(step),step_group)

    # coupling loop
    model.connect('Step'+str(step)+'.modal_forces.mf','Step'+str(step)+'.modal_solver.f')
    model.connect('Step'+str(step)+'.modal_solver.zn','Step'+str(step)+'.modal_disps.md')
    model.connect('Step'+str(step)+'.modal_disps.dx','Step'+str(step)+'.forcer.dx')
    model.connect('Step'+str(step)+'.forcer.f','Step'+str(step)+'.modal_forces.f')

    # modal properties
    model.connect('Indeps.m','Step'+str(step)+'.modal_solver.m')
    model.connect('Indeps.c','Step'+str(step)+'.modal_solver.c')
    model.connect('Indeps.k','Step'+str(step)+'.modal_solver.k')

    # forcer
    model.connect('Indeps.time','Step'+str(step)+'.forcer.time',src_indices=step)
    model.connect('Indeps.amp','Step'+str(step)+'.forcer.amp')
    model.connect('Indeps.freq','Step'+str(step)+'.forcer.freq')

    # backplanes of data for bdf derivative
    for backplane in range(1,5):
        back_step = step - backplane
        if back_step > 0:
            model.connect('Step'+str(back_step)+'.modal_solver.zn','Step'+str(step)+'.modal_solver.znm'+str(backplane))
        else:
            model.connect('Indeps.z0','Step'+str(step)+'.modal_solver.znm'+str(backplane))

prob.setup()
prob.run_model()
for step in range(1,nsteps+1):
    print('State',step,prob['Step'+str(step)+'.modal_solver.zn'][0])
print('Final state',prob['Step'+str(nsteps)+'.modal_solver.zn'][0])
