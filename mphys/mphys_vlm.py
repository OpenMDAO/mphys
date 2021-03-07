import numpy as np

import openmdao.api as om
from mphys import Builder

from vlm_solver import VLM_solver, VLM_forces

class VlmMesh(om.IndepVarComp):
    def initialize(self):
        self.options.declare('x_aero0')
    def setup(self):
        self.add_output('x_aero0', val=self.options['x_aero0'], tags=['mphys_coordinates'])

class VlmMeshAeroOnly(om.IndepVarComp):
    def initialize(self):
        self.options.declare('x_aero0')
    def setup(self):
        self.add_output('x_aero', val=self.options['x_aero0'], tags=['mphys_coordinates'])

class VlmGroup(om.Group):

    def initialize(self):
        self.options.declare('connectivity')
        self.options.declare('number_of_nodes')
        self.options.declare('compute_tractions', default=False)

    def setup(self):
        self.add_subsystem('solver', VLM_solver(connectivity=self.options['connectivity']),
                           promotes_inputs=['aoa','mach','x_aero'])

        self.add_subsystem('forces', VLM_forces(connectivity=self.options['connectivity'],
                                                number_of_nodes = self.options['number_of_nodes'],
                                                compute_tractions=self.options['compute_tractions']),
                           promotes_inputs=['mach','q_inf','vel','mu', 'x_aero'],
                           promotes_outputs=['f_aero','C_L','C_D'])

        self.connect('solver.Cp', 'forces.Cp')

class DummyVlmSolver(object):
    """
    RLT specific data storage and methods
    """
    def __init__(self, x_aero0, conn):
        self.x_aero0 = x_aero0
        self.conn = conn

    def getSurfaceCoordinates(self, group):
        return self.x_aero0

    def getSurfaceConnectivity(self, group):
        fortran_offset = -1
        conn = self.conn.copy() + fortran_offset
        faceSizes = 4*np.ones(len(conn), 'intc')
        return conn.astype('intc'), faceSizes

class VlmBuilder(Builder):
    def __init__(self, meshfile, compute_tractions=False):
        self.meshfile = meshfile
        self.compute_tractions = compute_tractions

    def initialize(self, comm):
        self._read_mesh(self.meshfile)
        self.solver = DummyVlmSolver(self.x_aero0, self.connectivity)

    def get_mesh_coordinate_subsystem(self):
        return VlmMesh(x_aero0=self.x_aero0)

    def get_coupling_group_subsystem(self):
        number_of_nodes = self.x_aero0.size // 3
        return VlmGroup(connectivity=self.connectivity,
                        number_of_nodes=number_of_nodes,
                        compute_tractions=self.compute_tractions)

    def get_scenario_subsystems(self):
        return None, None

    def get_number_of_nodes(self):
        return self.x_aero0.size // 3

    def get_ndof(self):
        return self.x_aero0.size // 3

    def _read_mesh(self, meshfile):
        with  open(meshfile,'r') as f:
            contents = f.read().split()

        a = [i for i in contents if 'NODES' in i][0]
        num_nodes = int(a[a.find("=")+1:a.find(",")])
        a = [i for i in contents if 'ELEMENTS' in i][0]
        num_elements = int(a[a.find("=")+1:a.find(",")])

        a = np.array(contents[16:16+num_nodes*3],'float')
        x = a[0:num_nodes*3:3]
        y = a[1:num_nodes*3:3]
        z = a[2:num_nodes*3:3]
        a = np.array(contents[16+num_nodes*3:None],'int')

        self.connectivity = np.reshape(a,[num_elements,4])
        self.x_aero0 = np.c_[x,y,z].flatten(order='C')

class VlmBuilderAeroOnly(VlmBuilder):
    def get_mesh_coordinate_subsystem(self):
        return VlmMeshAeroOnly(x_aero0=self.x_aero0)
