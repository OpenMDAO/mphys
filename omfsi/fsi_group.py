import openmdao.api as om

class fsi_group(om.Group):

    # def initialize(self):

    #     # define the inputs we need
    #     self.options.declare('aero_options', allow_none=False)
    #     self.options.declare('struct_options', allow_none=False)
    #     self.options.declare('transfer_options', allow_none=False)
    #     self.options.declare('n_scenario', default=1)

    def setup(self):

        # # add an ivc to this level for DVs that are shared between the two scenarios
        # dvs = om.IndepVarComp()
        # dvs.add_output{'foo', val=1}
        # self.add_subsystem('dvs', dvs)

        # # add the meshes
        # self.add_subsystem('struct_mesh', TacsMesh())
        # self.add_subsystem('aero_mesh', AdflowMesh())

        # # create the cruise cases
        # n_scenario = self.options['n_scenario']
        # for i in range(n_scenario):
        #     self.add_subsystem('cruise%d'%i, omfsi_scenario())

        # set solvers
        self.nonlinear_solver=om.NonlinearRunOnce()
        self.linear_solver = om.LinearRunOnce()

    # def configure(self):

        # now we need to connect all these components...

