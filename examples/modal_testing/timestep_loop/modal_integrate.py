#!/usr/bin/env python
from __future__ import print_function
from modal_integrator import *
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
model.add_subsystem('indeps',indeps)


integrator = ModalIntegrator(nmodes=nmodes,nsteps=nsteps,dt=dt,root_name=root_name)
model.add_subsystem('integrator',integrator)

model.connect('indeps.z0','integrator.z0')
model.connect('indeps.m','integrator.m')
model.connect('indeps.c','integrator.c')
model.connect('indeps.k','integrator.k')
model.connect('indeps.amp','integrator.amp')
model.connect('indeps.freq','integrator.freq')

prob.setup()
prob.run_model()
for step in range(1,nsteps+1):
    print('State',step,integrator.z[step,0])
print('Final state',integrator.z[-1,0])
