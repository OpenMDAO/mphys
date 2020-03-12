import openmdao.api as om
from omfsi.geo_disp import Geo_Disp

class AS_Scenario(om.Group):

    def initialize(self):

        # define the inputs we need
        self.options.declare('aero_builder', allow_none=False)
        self.options.declare('struct_builder', allow_none=False)
        self.options.declare('xfer_builder', allow_none=False)

    def setup(self):

        # set the builders
        self.aero_builder = self.options['aero_builder']
        self.struct_builder = self.options['struct_builder']
        self.xfer_builder = self.options['xfer_builder']

        # get the elements from each builder
        aero = self.aero_builder.get_element()
        struct = self.struct_builder.get_element()
        disp_xfer, load_xfer = self.xfer_builder.get_element()
        # component that only adds a geoemtric displacement to the aero surface. might remove
        geo_disp = Geo_Disp()

        # add the subgroups
        self.add_subsystem('disp_xfer', disp_xfer)
        self.add_subsystem('geo_disp', geo_disp)
        self.add_subsystem('aero', aero)
        self.add_subsystem('load_xfer', load_xfer)
        self.add_subsystem('struct', struct)

        # set solvers
        self.nonlinear_solver=om.NonlinearBlockGS(maxiter=100)
        self.nonlinear_solver.options['iprint']=2
        self.linear_solver = om.LinearBlockGS(maxiter=100)

    def configure(self):
        return
        # # add any io that has a size dependency
        # ndv = self.fsi_group.struct.get_ndv()
        # get_funcs = self.fsi_group.struct.get_funcs()
        # self.struct_funcs.add_io(ndv, get_funcs)
        # self.struct_mass.add_io(ndv)

        # # do the connections
        # self.connect('fsi_group.struct.u_s', 'struct_funcs.u_s')
        # self.connect('fsi_group.aero.deformer.x_g', 'aero_funcs.x_g')
        # self.connect('fsi_group.aero.solver.q', 'aero_funcs.q')

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