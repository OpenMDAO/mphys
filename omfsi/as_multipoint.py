import openmdao.api as om

# get the mesh components. Ideally, these classes will come from whatever
# solver we pick and will not be specific to adflow and tacs
from omfsi.scenario_group import omfsi_scenario
from omfsi.tacs_component_configure import TacsMesh
from omfsi.adflow_component_configure import AdflowMesh

from baseclasses import AeroProblem
from adflow import ADFLOW
from idwarp import USMesh

from tacs import TACS

from funtofem import TransferScheme



class AS_Multipoint(om.Group):

    def initialize(self):

        # define the inputs we need
        self.options.declare('aero_options', allow_none=False)
        self.options.declare('struct_options', allow_none=False)
        self.options.declare('transfer_options', allow_none=False)
        self.options.declare('n_scenario', default=1)

    def setup(self):

        # create solvers
        # this puts in the aero solver in self.aero_solver etc
        self.create_solvers()

        # add an ivc to this level for DVs that are shared between the two scenarios
        dv = om.IndepVarComp()
        # add the foo output here bec. we may not have any DVs
        # (even though we most likely will),
        # and w/o any output for the ivc, om will complain
        dv.add_output('foo', val=1)
        self.add_subsystem('dv', dv)

        # add the meshes
        self.add_subsystem('struct_mesh', TacsMesh(struct_solver=self.struct_solver))
        self.add_subsystem('aero_mesh', AdflowMesh(aero_solver=self.aero_solver))

        # create the cruise cases
        n_scenario = self.options['n_scenario']
        for i in range(n_scenario):
            self.add_subsystem('cruise%d'%i, omfsi_scenario(
                aero_solver=self.aero_solver,
                struct_solver=self.struct_solver,
                struct_objects=self.struct_objects,
                xfer_object=self.xfer_object
            ))

        # set solvers
        self.nonlinear_solver = om.NonlinearRunOnce()
        self.linear_solver    = om.LinearRunOnce()

    def configure(self):

        # connect the initial mesh coordinates.
        # with the configure-based approach, we do not need to have
        # separate components just to carry the initial mesh coordinates,
        # but we can directly pass them to all of the components here.
        # at this stage, everything is allocated and every group/component
        # below this level is set up.

        # loop over scenarios and connect them all
        n_scenario = self.options['n_scenario']
        for i in range(n_scenario):
            target_x_s0 = [
                'cruise%d.fsi_group.disp_xfer.x_s0'%i,
                'cruise%d.fsi_group.load_xfer.x_s0'%i,
                'cruise%d.fsi_group.struct.x_s0'%i,
                'cruise%d.struct_funcs.x_s0'%i,
                'cruise%d.struct_mass.x_s0'%i
            ]
            self.connect('struct_mesh.x_s0_mesh', target_x_s0)

            target_x_a0 = [
                'cruise%d.fsi_group.disp_xfer.x_a0'%i,
                'cruise%d.fsi_group.geo_disp.x_a0'%i,
                'cruise%d.fsi_group.load_xfer.x_a0'%i
            ]
            self.connect('aero_mesh.x_a0_mesh', target_x_a0)

    def create_solvers(self):
        """ This method is called from the setup method of this group.
        Here, we have a comm. Defined on the group. This method will
        create the python objects required from each scenario we will
        create under this group.

        Right now, this code has a lot of solver-specific calls. Ideally
        the solvers (or their om components) can provide a class or a function
        as input that takes in a dict of options, and a comm. Then, we can use
        these input class/functions to create the solver objects themselves.
        """

        # just use the comm of the group; however, we can also do splits
        comm = self.comm

        # create the aero solver
        #self.aero_solver = ADFLOW(options=self.options['aero_options'], comm=self.comm)
        self.aero_solver.get_solver(<get_local_scenario_comm>)


        # create the TACS object and the other solver objects it needs
        self.struct_solver, self.struct_objects = self.create_tacs()

        # create the transfer
        transfer_options = self.options['transfer_options']
        self.xfer_object = TransferScheme.pyMELD(self.comm,
                                                 self.comm, 0,
                                                 self.comm, 0,
                                                 transfer_options['isym'],
                                                 transfer_options['n'],
                                                 transfer_options['beta'])

    def create_tacs(self):
        """ This method creates the TACS object. This is all tacs specific code,
        instead, maybe we can use pyTACS to provide the common initialization API
        for the TACS solver
        """

        solver_dict={}

        mesh = TACS.MeshLoader(self.comm)
        mesh.scanBDFFile(self.options['struct_options']['mesh_file'])

        ndof, ndv = self.options['struct_options']['add_elements'](mesh)
        self.n_dv_struct = ndv

        tacs = mesh.createTACS(ndof)

        nnodes = int(tacs.createNodeVec().getArray().size / 3)

        mat = tacs.createFEMat()
        pc = TACS.Pc(mat)

        subspace = 100
        restarts = 2
        gmres = TACS.KSM(mat, pc, subspace, restarts)

        solver_dict['ndv']    = ndv
        solver_dict['ndof']   = ndof
        solver_dict['nnodes'] = nnodes
        solver_dict['get_funcs'] = self.options['struct_options']['get_funcs']

        # put the rest of the stuff in a tuple
        solver_objects = [mat, pc, gmres, solver_dict]

        return tacs, solver_objects

    def set_aero_problems(self, ap_list):
        # set the aero problems for each of the scenarios
        n_scenario = self.options['n_scenario']
        for i in range(n_scenario):
            ap = ap_list[i]
            scenario = getattr(self, 'cruise%d'%i)
            scenario.set_ap(ap)

    def add_dv_struct(self, name, val, **kwargs):
        self.dv.add_output(name, val)
        # connect to struct DVs at each scenario
        n_scenario = self.options['n_scenario']
        for i in range(n_scenario):
            target_dv_sturct = [
                'cruise%d.fsi_group.struct.dv_struct'%i,
                'cruise%d.struct_funcs.dv_struct'%i,
                'cruise%d.struct_mass.dv_struct'%i
            ]
            self.connect('dv.dv_struct', target_dv_sturct)

        self.add_design_var('dv.dv_struct', val, **kwargs)

