import numpy as np
import openmdao.api as om

from modal_comps import ModalDisplacements, ModalForces, ModalStep, HarmonicForcer


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

prob = om.Problem()
model = prob.model

indeps = om.IndepVarComp()
indeps.add_output('z0',z0)
indeps.add_output('m',m)
indeps.add_output('c',c)
indeps.add_output('k',k)
indeps.add_output('amp',amp)
indeps.add_output('freq',omega)
indeps.add_output('time',time)
model.add_subsystem('indeps',indeps)

for step in range(1,nsteps+1):
    step_group = om.Group()
    step_group_name = f'Step{step}'

    step_group.add_subsystem('modal_forces',ModalForces(nmodes=nmodes,root_name=root_name))
    step_group.add_subsystem('modal_solver',ModalStep(nmodes=nmodes,dt=dt))
    step_group.add_subsystem('modal_disps' ,ModalDisplacements(nmodes=nmodes,root_name=root_name))
    step_group.add_subsystem('forcer',HarmonicForcer(root_name=root_name))
    step_group.nonlinear_solver = om.NonlinearBlockGS()
    step_group.linear_solver = om.LinearBlockGS()
    model.add_subsystem(step_group_name, step_group)

    # coupling loop
    model.connect(f'{step_group_name}.modal_forces.mf',f'{step_group_name}.modal_solver.f')
    model.connect(f'{step_group_name}.modal_solver.zn',f'{step_group_name}.modal_disps.md')
    model.connect(f'{step_group_name}.modal_disps.dx',f'{step_group_name}.forcer.dx')
    model.connect(f'{step_group_name}.forcer.f',f'{step_group_name}.modal_forces.f')

    # modal properties
    model.connect('indeps.m',f'{step_group_name}.modal_solver.m')
    model.connect('indeps.c',f'{step_group_name}.modal_solver.c')
    model.connect('indeps.k',f'{step_group_name}.modal_solver.k')

    # forcer
    model.connect('indeps.time',f'{step_group_name}.forcer.time',src_indices=step)
    model.connect('indeps.amp',f'{step_group_name}.forcer.amp')
    model.connect('indeps.freq',f'{step_group_name}.forcer.freq')

    # backplanes of data for bdf derivative
    for backplane in range(1,5):
        back_step = step - backplane
        if back_step > 0:
            model.connect(f'Step{back_step}.modal_solver.zn',f'Step{step}.modal_solver.znm{backplane}')
        else:
            model.connect('indeps.z0',f'Step{step}.modal_solver.znm{backplane}')

prob.setup()
prob.run_model()
for step in range(1,nsteps+1):
    print('State',step, prob[f'Step{step}.modal_solver.zn'][0])
print('Final state',prob[f'Step{nsteps}.modal_solver.zn'][0])

derivs = prob.compute_totals(of=[f'Step{nsteps}.modal_solver.zn'],wrt=['indeps.k'])
print('Derivs',derivs)

prob.check_totals(of=[f'Step{nsteps}.modal_solver.zn'],wrt=['indeps.k','indeps.m','indeps.z0', 'indeps.amp'])
