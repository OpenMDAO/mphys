import numpy as np
import openmdao.api as om
from funtofem import TransferScheme
from builder_class import Builder

""" builder and components to wrap meld thermal to transfert temperature and
heat transfer rate between the convective and conductive analysis."""


class MELDThermal_temp_xfer(om.ExplicitComponent):
    """
    Component to perform displacement transfer using MELD
    """
    def initialize(self):
        self.options.declare('xfer_object')
        self.options.declare('cond_ndof')
        self.options.declare('cond_nnodes')

        self.options.declare('conv_nnodes')
        self.options.declare('check_partials')
        self.options.declare('mapping')

        self.meldThermal = None
        self.initialized_meld = False

        self.cond_ndof = None
        self.cond_nnodes = None
        self.conv_nnodes = None
        self.check_partials = False

    def setup(self):
        self.meldThermal = self.options['xfer_object']

        self.cond_ndof   = self.options['cond_ndof']
        self.cond_nnodes = self.options['cond_nnodes']
        self.conv_nnodes   = self.options['conv_nnodes']
        self.check_partials= self.options['check_partials']
        conv_nnodes = self.conv_nnodes

        # inputs
        self.add_input('x_struct0', distributed=True, shape_by_conn=True, desc='initial structural node coordinates')
        self.add_input('x_aero0', distributed=True, shape_by_conn=True, desc='initial aerodynamic surface node coordinates')
        self.add_input('T_conduct', distributed=True, shape_by_conn=True, desc='conductive node displacements')

        # outputs
        print('T_convect', conv_nnodes)

        self.add_output('T_convect', shape = conv_nnodes,
                                     distributed=True,
                                     val=np.ones(conv_nnodes)*301,
                                     desc='conv surface temperatures')


    def compute(self, inputs, outputs):

        x_s0 = np.array(inputs['x_struct0'],dtype=TransferScheme.dtype)
        x_a0 = np.array(inputs['x_aero0'],dtype=TransferScheme.dtype)
        mapping = self.options['mapping']

        # x_surface =  np.zeros((len(mapping), 3))

        # for i in range(len(mapping)):
        #     idx = mapping[i]*3
        #     x_surface[i] = x_s0[idx:idx+3]


        self.meldThermal.setStructNodes(x_s0)
        self.meldThermal.setAeroNodes(x_a0)

        # heat_xfer_cond0 = np.array(inputs['heat_xfer_cond0'],dtype=TransferScheme.dtype)
        # heat_xfer_conv0 = np.array(inputs['heat_xfer_conv0'],dtype=TransferScheme.dtype)
        temp_conv  = np.array(outputs['T_convect'],dtype=TransferScheme.dtype)

        temp_cond  = np.array(inputs['T_conduct'],dtype=TransferScheme.dtype)
        # for i in range(3):
        #     temp_cond[i::3] = inputs['T_conduct'][i::self.cond_ndof]


        if not self.initialized_meld:
            self.meldThermal.initialize()
            self.initialized_meld = True

        self.meldThermal.transferTemp(temp_cond,temp_conv)

        outputs['T_convect'] = temp_conv

class MELDThermal_heat_xfer_rate_xfer(om.ExplicitComponent):
    """
    Component to perform load transfers using MELD
    """
    def initialize(self):
        self.options.declare('xfer_object')
        self.options.declare('cond_ndof')
        self.options.declare('cond_nnodes')

        self.options.declare('conv_nnodes')
        self.options.declare('check_partials')
        self.options.declare('mapping')

        self.meldThermal = None
        self.initialized_meld = False

        self.cond_ndof = None
        self.cond_nnodes = None
        self.conv_nnodes = None
        self.check_partials = False

    def setup(self):
        # get the transfer scheme object
        self.meldThermal = self.options['xfer_object']

        self.cond_ndof   = self.options['cond_ndof']
        self.cond_nnodes = self.options['cond_nnodes']
        self.conv_nnodes   = self.options['conv_nnodes']
        self.check_partials= self.options['check_partials']

        # inputs
        self.add_input('x_struct0', distributed=True, shape_by_conn=True, desc='initial structural node coordinates')
        self.add_input('x_aero0', distributed=True, shape_by_conn=True, desc='initial aerodynamic surface node coordinates')
        self.add_input('q_convect', distributed=True, shape_by_conn=True, desc='initial conv heat transfer rate')

        print('q_conduct', self.cond_nnodes)

        # outputs
        self.add_output('q_conduct', distributed=True, shape = self.cond_nnodes, desc='heat transfer rate on the conduction mesh at the interface')


    def compute(self, inputs, outputs):

        heat_xfer_conv =  np.array(inputs['q_convect'],dtype=TransferScheme.dtype)
        heat_xfer_cond = np.zeros(self.cond_nnodes,dtype=TransferScheme.dtype)

        # if self.check_partials:
        #     x_s0 = np.array(inputs['x_struct0'],dtype=TransferScheme.dtype)
        #     x_a0 = np.array(inputs['x_aero0'],dtype=TransferScheme.dtype)
        #     self.meldThermal.setStructNodes(x_s0)
        #     self.meldThermal.setAeroNodes(x_a0)

        #     #TODO meld needs a set state rather requiring transferDisps to update the internal state

        #     temp_conv = np.zeros(inputs['q_convect'].size,dtype=TransferScheme.dtype)
        #     temp_cond  = np.zeros(self.cond_surface_nnodes,dtype=TransferScheme.dtype)
        #     for i in range(3):
        #         temp_cond[i::3] = inputs['T_conduct'][i::self.cond_ndof]


        #     self.meldThermal.transferTemp(temp_cond,temp_conv)

        self.meldThermal.transferFlux(heat_xfer_conv,heat_xfer_cond)
        outputs['q_conduct'] = heat_xfer_cond

class MELDThermal_builder(Builder):

    def __init__(self, options, conv_builder, cond_builder, check_partials=False):
        super(MELDThermal_builder, self).__init__(options)
        self.check_partials = check_partials
        # TODO we can move the conv and cond builder to init_xfer_object call so that user does not need to worry about this
        self.conv_builder = conv_builder
        self.cond_builder = cond_builder

    # api level method for all builders
    def init_xfer_object(self, comm):
        # create the transfer
        self.xfer_object = TransferScheme.pyMELDThermal(comm,
                                                 comm, 0,
                                                 comm, 0,
                                                 self.options['isym'],
                                                 self.options['n'],
                                                 self.options['beta'])

        # TODO also do the necessary calls to the cond and conv builders to fully initialize MELD
        # for now, just save the counts
        self.cond_ndof = self.cond_builder.get_ndof()
        tacs = self.cond_builder.get_solver()
        get_surface = self.cond_builder.options['get_surface']

        surface_nodes, mapping = get_surface(tacs)
        # get mapping of flow edge
        self.mapping = mapping
        self.cond_nnodes = len(mapping)

        self.conv_nnodes = self.conv_builder.get_nnodes(groupName='allIsothermalWalls')

    # api level method for all builders
    def get_xfer_object(self):
        return self.xfer_object

    # api level method for all builders
    def get_element(self):

        temp_xfer = MELDThermal_temp_xfer(
            xfer_object=self.xfer_object,
            cond_ndof=self.cond_ndof,
            cond_nnodes=self.cond_nnodes,
            conv_nnodes=self.conv_nnodes,
            check_partials=self.check_partials
        )



        heat_xfer_xfer = MELDThermal_heat_xfer_rate_xfer(
            xfer_object=self.xfer_object,
            cond_ndof=self.cond_ndof,
            cond_nnodes=self.cond_nnodes,
            conv_nnodes=self.conv_nnodes,
            check_partials=self.check_partials
        )

        return temp_xfer, heat_xfer_xfer


    def build_object(self, comm):
        self.init_xfer_object(comm)

    def get_object(self):
        return self.xfer_object()

    def get_component(self):


        temp_xfer = MELDThermal_temp_xfer(
            xfer_object=self.xfer_object,
            cond_ndof=self.cond_ndof,
            cond_nnodes=self.cond_nnodes,
            conv_nnodes=self.conv_nnodes,
            check_partials=self.check_partials,
            mapping = self.mapping

        )

        yield '_temps', temp_xfer

        heat_xfer_xfer = MELDThermal_heat_xfer_rate_xfer(
            xfer_object=self.xfer_object,
            cond_ndof=self.cond_ndof,
            cond_nnodes=self.cond_nnodes,
            conv_nnodes=self.conv_nnodes,
            check_partials=self.check_partials,
            mapping = self.mapping
        )

        yield  '_heat_rate', heat_xfer_xfer
