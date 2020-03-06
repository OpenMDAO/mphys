import openmdao.api as om
from omfsi.fsi_group import fsi_group
from omfsi.tacs_component_configure import TacsFunctions, TacsMass
from omfsi.adflow_component_configure import AdflowFunctions
from adflow.python.om_utils import get_dvs_and_cons

class omfsi_scenario(om.Group):

    def initialize(self):

        # define the inputs we need
        self.options.declare('aero_solver', allow_none=False)
        self.options.declare('struct_solver', allow_none=False)
        self.options.declare('struct_objects', allow_none=False)
        self.options.declare('xfer_object', allow_none=False)

    def setup(self):

        # create the dv component to store the dvs for this scenario
        dv = om.IndepVarComp()
        # add the foo output here bec. we may not have any DVs
        # (even though we most likely will),
        # and w/o any output for the ivc, om will complain
        dv.add_output('foo', val=1)
        self.add_subsystem('dv', dv)

        # get all the initialized objects for computations
        self.aero_solver = self.options['aero_solver']
        self.struct_solver = self.options['struct_solver']
        self.struct_objects = self.options['struct_objects']
        self.xfer_object = self.options['xfer_object']

        # create the components and groups
        self.add_subsystem('fsi_group', fsi_group(
            aero_solver=self.aero_solver,
            struct_solver=self.struct_solver,
            struct_objects=self.struct_objects,
            xfer_object=self.xfer_object
        ))
        self.add_subsystem('struct_funcs', TacsFunctions(
            struct_solver=self.struct_solver
        ))
        self.add_subsystem('struct_mass', TacsMass(
            struct_solver=self.struct_solver
        ))
        self.add_subsystem('aero_funcs', AdflowFunctions(
            aero_solver=self.aero_solver
        ))

        # set solvers
        self.nonlinear_solver=om.NonlinearRunOnce()
        self.linear_solver = om.LinearRunOnce()

    def configure(self):

        # add any io that has a size dependency
        ndv = self.fsi_group.struct.get_ndv()
        get_funcs = self.fsi_group.struct.get_funcs()
        self.struct_funcs.add_io(ndv, get_funcs)
        self.struct_mass.add_io(ndv)

        # do the connections
        self.connect('fsi_group.struct.u_s', 'struct_funcs.u_s')
        self.connect('fsi_group.aero.deformer.x_g', 'aero_funcs.x_g')
        self.connect('fsi_group.aero.solver.q', 'aero_funcs.q')

    def set_ap(self, ap):
        # this function sets the aero problem in all relevant spots
        # and adds the DVs of the aero problem

        # call the set_ap function on every component that uses the ap
        self.fsi_group.aero.solver.set_ap(ap)
        self.fsi_group.aero.force.set_ap(ap)
        self.aero_funcs.set_ap(ap)

        # connect the AP DVs to all components that use them
        self.ap_vars,_ = get_dvs_and_cons(ap=ap)
        for (args, kwargs) in self.ap_vars:
            name = args[0]
            size = args[1]

            self.dv.add_output(name, val=kwargs['value'])

            ap_vars_target = [
                'fsi_group.aero.solver.%s'%name,
                'fsi_group.aero.force.%s'%name,
                'aero_funcs.%s'%name
            ]

            self.connect('dv.%s'%name, ap_vars_target)