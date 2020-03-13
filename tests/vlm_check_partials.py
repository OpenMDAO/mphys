import numpy as np
from mpi4py import MPI
from openmdao.api import Problem, Group, IndepVarComp, ExecComp
from omfsi.vlm_component import *

comm = MPI.COMM_WORLD

# VLM options

aero_options = {
    'mesh_file':'debug_VLM.dat',
    'mach':0.85,
    'alpha':1*np.pi/180.,
    'q_inf':25000.,
    'vel':178.,
    'mu':3.5E-5,
}

# VLM mesh read

def read_VLM_mesh(mesh):
    f=open(mesh, "r")
    contents = f.read().split()

    a = [i for i in contents if 'NODES' in i][0]
    N_nodes = int(a[a.find("=")+1:a.find(",")])
    a = [i for i in contents if 'ELEMENTS' in i][0]
    N_elements = int(a[a.find("=")+1:a.find(",")])

    a = np.array(contents[16:16+N_nodes*3],'float')
    X = a[0:N_nodes*3:3]
    Y = a[1:N_nodes*3:3]
    Z = a[2:N_nodes*3:3]
    a = np.array(contents[16+N_nodes*3:None],'int')
    quad = np.reshape(a,[N_elements,4])

    xa = np.c_[X,Y,Z].flatten(order='C')

    f.close()

    return N_nodes, N_elements, xa, quad

aero_options['N_nodes'], aero_options['N_elements'], aero_options['x_a0'], aero_options['quad'] = read_VLM_mesh(aero_options['mesh_file'])

## assembler

assembler = VlmAssembler(aero_options,comm)

## openmdao setup

connection_srcs = {}
prob = Problem()
model = prob.model

indeps = IndepVarComp()
indeps.add_output('alpha',aero_options['alpha'])
indeps.add_output('x_a',aero_options['x_a0'])
model.add_subsystem('indeps',indeps)
connection_srcs['alpha'] = 'indeps.alpha'
connection_srcs['x_a'] = 'indeps.x_a'

assembler.add_model_components(model,connection_srcs)

scenario = model.add_subsystem('cruise1',Group())
fsi_group = scenario.add_subsystem('fsi_group',Group())

assembler.add_scenario_components(model,scenario,connection_srcs)

assembler.add_fsi_components(model,scenario,fsi_group,connection_srcs)
assembler.connect_inputs(model,scenario,fsi_group,connection_srcs)

prob.setup(force_alloc_complex=True)
prob.run_model()
prob.check_partials(method='cs',compact_print=True)






