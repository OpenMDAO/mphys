# complex step partial derivative check of MELD transfer components
# must compile funtofem in complex mode
import numpy as np
from mpi4py import MPI

import openmdao.api as om
from mphys.mphys_meld import MeldDispXfer, MeldLoadXfer

from funtofem import TransferScheme

class FakeStructMesh(om.ExplicitComponent):
    def initialize(self):
        self.nodes = np.random.rand(12)

    def setup(self):
        self.add_output('x_s',shape=self.nodes.size)

    def compute(self,inputs,outputs):
        outputs['x_s'] = self.nodes

class FakeStructDisps(om.ExplicitComponent):
    def initialize(self):
        self.nodes = np.random.rand(12)
        self.nodes = np.arange(12)

    def setup(self):
        self.add_output('u_s',shape=self.nodes.size)

    def compute(self,inputs,outputs):
        outputs['u_s'] = self.nodes

class FakeAeroLoads(om.ExplicitComponent):
    def initialize(self):
        self.nodes = np.random.rand(12)

    def setup(self):
        self.add_output('f_a',shape=self.nodes.size)

    def compute(self,inputs,outputs):
        outputs['f_a'] = self.nodes

class FakeAeroMesh(om.ExplicitComponent):
    def initialize(self):
        self.nodes = np.random.rand(12)

    def setup(self):
        self.add_output('x_a',shape=self.nodes.size)

    def compute(self,inputs,outputs):
        outputs['x_a'] = self.nodes

comm = MPI.COMM_WORLD
isym = 1
n = 20
beta = 0.5
meld = TransferScheme.pyMELD(comm, comm,0, comm,0, isym,n,beta)

struct_ndof = 3
struct_nnodes = 4
aero_nnodes = 4

prob = om.Problem()
prob.model.add_subsystem('aero_mesh',FakeAeroMesh())
prob.model.add_subsystem('struct_mesh',FakeStructMesh())
prob.model.add_subsystem('struct_disps',FakeStructDisps())
prob.model.add_subsystem('aero_loads',FakeAeroLoads())
disp = prob.model.add_subsystem('disp_xfer',MeldDispXfer(xfer_object=meld,struct_ndof=struct_ndof,struct_nnodes=struct_nnodes,aero_nnodes=aero_nnodes,check_partials=True))
load = prob.model.add_subsystem('load_xfer',MeldLoadXfer(xfer_object=meld,struct_ndof=struct_ndof,struct_nnodes=struct_nnodes,aero_nnodes=aero_nnodes,check_partials=True))

prob.model.add_subsystem('aero_mesh2',FakeAeroMesh())
prob.model.add_subsystem('struct_mesh2',FakeStructMesh())
prob.model.add_subsystem('struct_disps2',FakeStructDisps())
prob.model.add_subsystem('aero_loads2',FakeAeroLoads())
disp = prob.model.add_subsystem('disp_xfer2',MeldDispXfer(xfer_object=meld,struct_ndof=struct_ndof,struct_nnodes=struct_nnodes,aero_nnodes=aero_nnodes,check_partials=True))
load = prob.model.add_subsystem('load_xfer2',MeldLoadXfer(xfer_object=meld,struct_ndof=struct_ndof,struct_nnodes=struct_nnodes,aero_nnodes=aero_nnodes,check_partials=True))

prob.model.connect('aero_mesh.x_a',['disp_xfer.x_a0','load_xfer.x_a0'])
prob.model.connect('struct_mesh.x_s',['disp_xfer.x_s0','load_xfer.x_s0'])
prob.model.connect('struct_disps.u_s',['disp_xfer.u_s','load_xfer.u_s'])
prob.model.connect('aero_loads.f_a',['load_xfer.f_a'])

prob.model.connect('aero_mesh2.x_a',['disp_xfer2.x_a0','load_xfer2.x_a0'])
prob.model.connect('struct_mesh2.x_s',['disp_xfer2.x_s0','load_xfer2.x_s0'])
prob.model.connect('struct_disps2.u_s',['disp_xfer2.u_s','load_xfer2.u_s'])
prob.model.connect('aero_loads2.f_a',['load_xfer2.f_a'])


prob.setup(force_alloc_complex=True)
prob.run_model()
prob.check_partials(method='cs',compact_print=True)
