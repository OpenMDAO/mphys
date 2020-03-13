# complex step partial derivative check of MELD transfer components
# must compile funtofem in complex mode
import numpy as np

from openmdao.api import Problem, ExplicitComponent
from omfsi.meld_xfer_component import MeldDisplacementTransfer, MeldLoadTransfer

from funtofem import TransferScheme

class FakeStructMesh(ExplicitComponent):
    def initialize(self):
        self.nodes = np.random.rand(15)

    def setup(self):
        self.add_output('x_s',shape=self.nodes.size)

    def compute(self,inputs,outputs):
        outputs['x_s'] = self.nodes

class FakeStructDisps(ExplicitComponent):
    def initialize(self):
        self.nodes = np.random.rand(15)

    def setup(self):
        self.add_output('u_s',shape=self.nodes.size)

    def compute(self,inputs,outputs):
        outputs['u_s'] = self.nodes

class FakeAeroMesh(ExplicitComponent):
    def initialize(self):
        self.nodes = np.random.rand(15)

    def setup(self):
        self.add_output('x_a',shape=self.nodes.size)

    def compute(self,inputs,outputs):
        outputs['x_a'] = self.nodes

class DummyAssembler(object):
    def __init__(self):
        self.meld = None
    def add_components(self,prob):
        disp = prob.model.add_subsystem('disp_xfer',MeldDisplacementTransfer(setup_function = self.xfer_setup))
        load = prob.model.add_subsystem('load_xfer',MeldLoadTransfer(setup_function = self.xfer_setup))

        disp.check_partials = True
        load.check_partials = True

    def xfer_setup(self,comm):
        if self.meld is None:
            isym = 1
            n = 20
            beta = 0.5
            self.meld = TransferScheme.pyMELD(comm, comm,0, comm,0, isym,n,beta)

        struct_ndof = 3
        struct_nnodes = 4
        aero_nnodes = 4
        return self.meld, struct_ndof, struct_nnodes, aero_nnodes

prob = Problem()
prob.model.add_subsystem('aero_mesh',FakeAeroMesh())
prob.model.add_subsystem('struct_mesh',FakeStructMesh())
prob.model.add_subsystem('struct_disps',FakeStructDisps())

dummy = DummyAssembler()
dummy.add_components(prob)
prob.model.connect('aero_mesh.x_a',['disp_xfer.x_a0','load_xfer.x_a0'])
prob.model.connect('struct_mesh.x_s',['disp_xfer.x_s0','load_xfer.x_s0'])
prob.model.connect('struct_disps.u_s',['disp_xfer.u_s','load_xfer.u_s'])


prob.setup(force_alloc_complex=True)
prob.run_model()
prob.check_partials(method='cs',compact_print=True)