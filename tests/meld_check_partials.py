# complex step partial derivative check of MELD transfer components
# must compile funtofem in complex mode
import numpy as np
from mpi4py import MPI

import openmdao.api as om
from mphys.mphys_meld import MeldDispXfer, MeldLoadXfer

from funtofem import TransferScheme

class FakeStructMesh(om.ExplicitComponent):
    def initialize(self):
        self.nodes = np.random.rand(15)

    def setup(self):
        self.add_output('x_s',shape=self.nodes.size)

    def compute(self,inputs,outputs):
        outputs['x_s'] = self.nodes

class FakeStructDisps(om.ExplicitComponent):
    def initialize(self):
        self.nodes = np.random.rand(15)

    def setup(self):
        self.add_output('u_s',shape=self.nodes.size)

    def compute(self,inputs,outputs):
        outputs['u_s'] = self.nodes

class FakeAeroMesh(om.ExplicitComponent):
    def initialize(self):
        self.nodes = np.random.rand(15)

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
struct_nnodes = 5
aero_nnodes = 5

prob = om.Problem()
prob.model.add_subsystem('aero_mesh',FakeAeroMesh())
prob.model.add_subsystem('struct_mesh',FakeStructMesh())
prob.model.add_subsystem('struct_disps',FakeStructDisps())
disp = prob.model.add_subsystem('disp_xfer',MeldDispXfer(xfer_object=meld,struct_ndof=struct_ndof,struct_nnodes=struct_nnodes,aero_nnodes=aero_nnodes))
load = prob.model.add_subsystem('load_xfer',MeldLoadXfer(xfer_object=meld,struct_ndof=struct_ndof,struct_nnodes=struct_nnodes,aero_nnodes=aero_nnodes))

disp.check_partials = True
load.check_partials = True

prob.model.connect('aero_mesh.x_a',['disp_xfer.x_a0','load_xfer.x_a0'])
prob.model.connect('struct_mesh.x_s',['disp_xfer.x_s0','load_xfer.x_s0'])
prob.model.connect('struct_disps.u_s',['disp_xfer.u_s','load_xfer.u_s'])


prob.setup(force_alloc_complex=True)
prob.run_model()
prob.check_partials(method='cs',compact_print=True)
