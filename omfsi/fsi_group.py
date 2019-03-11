import numpy as np
from openmdao.api import ExplicitComponent, Group, NonlinearBlockGS, LinearBlockGS

class FsiSolver(Group):
    """
    FSI coupling loop

    Variables:
        x_a0 = aerodynamic surface with geom changes
        u_a  = aerodynamic surface displacements
        x_a  = deformed aerodynamic surface
        f_a  = aerodynamic forces
        x_g  = aerodynamic volume mesh
          q  = aerodynamic state vector

        x_s0 = structural mesh with geom changes
        u_s  = structural mesh displacements
        f_s  = structural forces
    """
    def initialize(self):
        self.options.declare('aero'     , default=None,desc='aerodynamic solver group')
        self.options.declare('struct'   , default=None,desc='structural solver group')
        self.options.declare('disp_xfer', default=None,desc='displacement transfer component')
        self.options.declare('load_xfer', default=None,desc='load transfer component')
        self.options.declare('struct_nprocs',default=1,desc='number of procs for structual analysis')
        self.options.declare('get_vector_size',default=None,desc='callback function to get local size of aero surface vector')

    def setup(self):
        # geo_disp does x_a0 + u_a
        get_vector_size = self.options['get_vector_size']

        self.add_subsystem('disp_xfer',self.options['disp_xfer'])
        self.add_subsystem('geo_disps',GeoDisp(get_vector_size=get_vector_size))
        self.add_subsystem('aero',  self.options['aero'])
        self.add_subsystem('load_xfer',self.options['load_xfer'])
        self.add_subsystem('struct',self.options['struct'])

        self.nonlinear_solver = NonlinearBlockGS(maxiter=50)
        #self.nonlinear_solver = NonlinearBlockGS(debug_print=True)
        self.linear_solver = LinearBlockGS(maxiter=50)

class GeoDisp(ExplicitComponent):
    """
    This components is a component that adds the aerodynamic
    displacements to the geometry-changed aerodynamic surface
    """
    def initialize(self):
        self.options.declare('get_vector_size',default=None,desc='callback function to get the size of the aero surface vectors')
        self.options['distributed'] = True

    def setup(self):
        local_size = self.options['get_vector_size']()
        n_list = self.comm.allgather(local_size)
        irank  = self.comm.rank

        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input('x_a0',shape=local_size,src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface with geom changes')
        self.add_input('u_a', shape=local_size,src_indices=np.arange(n1,n2,dtype=int),desc='aerodynamic surface displacements')

        self.add_output('x_a',shape=local_size,desc='deformed aerodynamic surface')

    def compute(self,inputs,outputs):
        outputs['x_a'] = inputs['x_a0'] + inputs['u_a']

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode == 'fwd':
            if 'x_a' in d_outputs:
                if 'x_a0' in d_inputs:
                    d_outputs['x_a'] += d_inputs['x_a0']
                if 'u_a' in d_inputs:
                    d_outputs['x_a'] += d_inputs['u_a']
        if mode == 'rev':
            if 'x_a' in d_outputs:
                if 'x_a0' in d_inputs:
                    d_inputs['x_a0'] += d_outputs['x_a']
                if 'u_a' in d_inputs:
                    d_inputs['u_a']  += d_outputs['x_a']
