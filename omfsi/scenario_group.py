import openmdao.api as om

class omfsi_scenario(om.Group):

    def initialize(self):

        # define the inputs we need
        self.options.declare('aero_solver', allow_none=False)
        self.options.declare('struct_solver', allow_none=False)
        self.options.declare('struct_objects', allow_none=False)
        self.options.declare('xfer_object', allow_none=False)

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

