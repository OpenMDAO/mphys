import numpy as np
from mpi4py import MPI
import openmdao.api as om

from mphys.integrated_forces import IntegratedSurfaceForces
from iris_wrapper import Iris
from sfe.sfe_parfait import SFE
from sfe.sfe_openmdao import SfeSolverOpenMdao, SfeForcesOpenMdao


comm = MPI.COMM_WORLD
iris = Iris(comm)

meshfile = 'crm_invis_tet.b8.ugrid'
boundary_tag_list = [3]
sfe_solver = SFE.from_meshfile(meshfile,iris)

num_nodes = sfe_solver.mesh.resident_node_count()

x,y,z = sfe_solver.get_boundary_node_coordinates(boundary_tag_list,owned_only=True)
num_surface_nodes = x.size
coords = np.concatenate((x.reshape((-1,1)),y.reshape((-1,1)),z.reshape((-1,1))),axis=1).flatten()

coords_sizes = comm.gather(coords.size)
if comm.Get_rank()==0:
    full_coords = np.zeros(sum(coords_sizes))
else:
    full_coords = np.zeros(0)
comm.Gatherv(coords,(full_coords,coords_sizes),0)
full_coords = comm.bcast(full_coords)

prob = om.Problem()
model = prob.model

ivc = model.add_subsystem('ivc',om.IndepVarComp())
ivc.add_output('mach',0.2)
ivc.add_output('aoa',1.0, units='deg')
ivc.add_output('reynolds',0.0)
ivc.add_output('u_g',np.zeros(comm.allreduce(num_nodes*3)))
ivc.add_output('x_aero',full_coords)
ivc.add_output('q_inf',1.0)

sfe_comp = model.add_subsystem('sfe',SfeSolverOpenMdao(sfe_solver = sfe_solver))
forces_comp = model.add_subsystem('forces',SfeForcesOpenMdao(sfe_solver = sfe_solver,
                                                             boundary_tag_list = boundary_tag_list))
inte_forces = model.add_subsystem('integrated_forces',IntegratedSurfaceForces(number_of_surface_nodes=num_surface_nodes))

model.connect('ivc.mach',['sfe.mach','forces.mach'])
model.connect('ivc.aoa',['sfe.aoa','integrated_forces.aoa'])
model.connect('ivc.reynolds_number',['sfe.reynolds_number','forces.reynolds_number'])
model.connect('ivc.u_g',['sfe.u_g','forces.u_g'])
model.connect('ivc.q_inf',['forces.q_inf','integrated_forces.q_inf'])
model.connect('ivc.x_aero',['integrated_forces.x_aero'])
model.connect('sfe.q',['forces.q'])
model.connect('forces.f_a',['integrated_forces.f_a'])

prob.setup(mode='rev')
#om.n2(prob, show_browser=False, outfile='n2_sfe.html')
prob.run_model()
prob.check_partials(compact_print=True)
#prob.check_totals(of=['integrated_forces.F_Z'], wrt=['ivc.mach','ivc.aoa'])
#print('TOTALS',prob.compute_totals(of=['integrated_forces.F_Z'], wrt=['ivc.q_inf','ivc.aoa','ivc.mach']))
