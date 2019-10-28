#rst Imports
from __future__ import print_function, division
import numpy
#from baseclasses import *
from mpi4py import MPI
from tacs import elements, constitutive

from omfsi.fsi_assembler import *
from omfsi.vlm_component import *
from omfsi.tacs_component import *
from omfsi.meld_xfer_component import *

from openmdao.api import Problem, ScipyOptimizeDriver
from openmdao.api import ExplicitComponent, ExecComp, IndepVarComp, Group
from openmdao.api import NonlinearRunOnce, LinearRunOnce
from openmdao.api import NonlinearBlockGS, LinearBlockGS
from openmdao.api import view_model

comm = MPI.COMM_WORLD

# VLM options

aero_options = {
    'mesh_file':'wing_VLM.dat',
    'mach':0.85,
    'alpha':2*np.pi/180.,
    'q_inf':100.,
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

# VLM assembler

aero_assembler = VlmAssembler(aero_options,comm)

# TACS setup

def add_elements(mesh):
    rho = 2780.0            # density, kg/m^3
    E = 73.1e9              # elastic modulus, Pa
    nu = 0.33               # poisson's ratio
    kcorr = 5.0 / 6.0       # shear correction factor
    ys = 324.0e6            # yield stress, Pa
    thickness= 0.020
    min_thickness = 0.002
    max_thickness = 0.05

    num_components = mesh.getNumComponents()
    for i in range(num_components):
        descript = mesh.getElementDescript(i)
        stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, thickness, i,
                                     min_thickness, max_thickness)
        element = None
        if descript in ["CQUAD", "CQUADR", "CQUAD4"]:
            element = elements.MITCShell(2,stiff,component_num=i)
        mesh.setElement(i, element)

    ndof = 6
    ndv = num_components

    return ndof, ndv

def get_funcs(tacs):
    ks_weight = 50.0
    return [ functions.KSFailure(tacs,ks_weight), functions.StructuralMass(tacs)]

tacs_setup = {'add_elements': add_elements,
              'mesh_file'   : 'wingbox_Y_Z_flip.bdf',
              'get_funcs'   : get_funcs}

# TACS assembler

struct_assembler = TacsOmfsiAssembler(tacs_setup)

# MELD setup

meld_options = {'isym': 2,
                'n': 200,
                'beta': 0.5}

# MELD assembler

xfer_assembler = MeldAssembler(meld_options,struct_assembler,aero_assembler)

# FSI assembler

assembler = FsiAssembler(struct_assembler,aero_assembler,xfer_assembler)

# OpenMDAO setup

prob = Problem()
model = prob.model

model.nonlinear_solver = NonlinearRunOnce()
model.linear_solver = LinearRunOnce()

# add the components and groups to the model

indeps = IndepVarComp()
indeps.add_output('dv_struct',np.array(810*[0.01]))
indeps.add_output('alpha',aero_options['alpha'])
model.add_subsystem('dv',indeps)

assembler.connection_srcs['dv_struct'] = 'dv.dv_struct'
assembler.connection_srcs['alpha'] = 'dv.alpha'

assembler.add_model_components(model)

scenario = model.add_subsystem('cruise1',Group())
scenario.nonlinear_solver = NonlinearRunOnce()
scenario.linear_solver = LinearRunOnce()

fsi_group = assembler.add_fsi_subsystem(model,scenario)
fsi_group.nonlinear_solver = NonlinearBlockGS(maxiter=100)
fsi_group.linear_solver = LinearBlockGS(maxiter=100)

# run OpenMDAO

prob.setup()
#view_model(prob)
prob.run_model()

