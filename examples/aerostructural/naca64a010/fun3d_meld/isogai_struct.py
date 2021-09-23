import numpy as np
from numpy.__config__ import show
import openmdao.api as om
import scipy.linalg as linalg
from mphys import Builder
from mphys.mphys_modal_solver import ModalGroup
from mpi4py import MPI

class IsogaiStructMesh(om.IndepVarComp):
    def setup(self):
        self.nnodes = 30 if self.comm.Get_rank()==0 else 0

        coords = np.zeros(self.nnodes)

        if self.comm.Get_rank() == 0:
            coords = np.zeros((self.nnodes,3))
            # x
            coords[0:10,0] = np.linspace(0,1,num=10)
            coords[10:20,0] = np.linspace(0,1,num=10)
            coords[20:30,0] = np.linspace(0,1,num=10)

            # y
            coords[0:10,1] = 0.0
            coords[10:20,1] = 0.5
            coords[20:30,1] = 1.0
            coords = coords.flatten()

        self.add_output('x_struct0',coords, distributed=True, tags = ['mphys_coordinates'])

class IsogaiEigenComp(om.ExplicitComponent):
    def initialize(self):
        self.nmodes = 2

        self.b = 0.5
        self.rho = 1.225

        self.unbal = 1.8   # static unbalance
        self.ra2 = 3.48    # r_a^2 square of radius of gyration
        self.mu = 60.0

    def setup(self):
        self.nnodes = 30 if self.comm.Get_rank()==0 else 0

        self.add_input('pitch_frequency', tags=['mphys_input'])
        self.add_input('plunge_frequency', tags=['mphys_input'])
        self.add_input('x_struct0', shape_by_conn=True, distributed=True, tags=['mphys_coordinates'])

        self.add_output('mode_shapes_struct', shape = [self.nnodes*3, self.nmodes], distributed=True, tags=['mphys_coupling'])

        self.add_output('modal_stiffness', shape=[self.nmodes], tags=['mphys_coupling'])
        self.add_output('modal_mass', shape=[self.nmodes], tags=['mphys_coupling'])
        self.add_output('modal_damping', shape=[self.nmodes], tags=['mphys_coupling'])

    def compute(self, inputs, outputs):

        mass, stiff, vec0, vec1 = self._structural_eigen_decomposition(inputs['plunge_frequency'], inputs['pitch_frequency'] )
        outputs['modal_mass'] = mass
        outputs['modal_stiffness'] = stiff
        outputs['modal_damping'][:] = 0.0

        if self.comm.Get_rank() == 0:
            x = inputs['x_struct0'][::3]
            y = inputs['x_struct0'][1::3]
            z = inputs['x_struct0'][2::3]

            outputs['mode_shapes_struct'][:,0] = self._compute_mode_shapes(x, y, z, vec0)
            outputs['mode_shapes_struct'][:,1] = self._compute_mode_shapes(x, y, z, vec1)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if 'plunge_frequency' in d_inputs:
            perturb = 1e-6
            mass, stiff, vec0, vec1 = self._structural_eigen_decomposition(inputs['plunge_frequency'], inputs['pitch_frequency'] )
            pmass, pstiff, pvec0, pvec1 = self._structural_eigen_decomposition(inputs['plunge_frequency']+perturb, inputs['pitch_frequency'] )
            dmass = (pmass - mass) / perturb
            dstiff = (pstiff - stiff) / perturb

            if 'modal_mass' in d_outputs:
                if mode == 'fwd':
                    d_outputs['modal_mass'] += dmass * d_inputs['plunge_frequency']
                if mode == 'rev':
                    d_inputs['plunge_frequency'] += np.sum(dmass * d_outputs['modal_mass'])
            if 'modal_stiffness' in d_outputs:
                if mode == 'fwd':
                    d_outputs['modal_stiffness'] += dstiff * d_inputs['plunge_frequency']
                if mode == 'rev':
                    d_inputs['plunge_frequency'] += np.sum(dstiff * d_outputs['modal_stiffness'])

            dvec0 = (pvec0 - vec0) / perturb
            dvec1 = (pvec1 - vec1) / perturb
            if self.comm.Get_rank() == 0:
                x = inputs['x_struct0'][::3]
                y = inputs['x_struct0'][1::3]
                z = inputs['x_struct0'][2::3]
                dmodeshapes = np.zeros((3*self.nnodes,self.nmodes))
                dmodeshapes[:,0] = self._compute_mode_shapes(x, y, z, dvec0)
                dmodeshapes[:,1] = self._compute_mode_shapes(x, y, z, dvec1)

                if 'mode_shapes_struct' in d_outputs:
                    if mode == 'fwd':
                        d_outputs['mode_shapes_struct'] += dmodeshapes * d_inputs['plunge_frequency']
                    if mode == 'rev':
                        d_inputs['plunge_frequency'] += np.sum(dmodeshapes * d_outputs['mode_shapes_struct'])

        if 'pitch_frequency' in d_inputs:
            perturb = 1e-6
            mass, stiff, vec0, vec1 = self._structural_eigen_decomposition(inputs['plunge_frequency'], inputs['pitch_frequency'] )
            pmass, pstiff, pvec0, pvec1 = self._structural_eigen_decomposition(inputs['plunge_frequency'], inputs['pitch_frequency']+perturb )
            dmass = (pmass - mass) / perturb
            dstiff = (pstiff - stiff) / perturb

            if 'modal_mass' in d_outputs:
                if mode == 'fwd':
                    d_outputs['modal_mass'] += dmass * d_inputs['pitch_frequency']
                if mode == 'rev':
                    d_inputs['pitch_frequency'] += np.sum(dmass * d_outputs['modal_mass'])
            if 'modal_stiffness' in d_outputs:
                if mode == 'fwd':
                    d_outputs['modal_stiffness'] += dstiff * d_inputs['pitch_frequency']
                if mode == 'rev':
                    d_inputs['pitch_frequency'] += np.sum(dstiff * d_outputs['modal_stiffness'])

            dvec0 = (pvec0 - vec0) / perturb
            dvec1 = (pvec1 - vec1) / perturb
            if self.comm.Get_rank() == 0:
                x = inputs['x_struct0'][::3]
                y = inputs['x_struct0'][1::3]
                z = inputs['x_struct0'][2::3]
                dmodeshapes = np.zeros((3*self.nnodes,self.nmodes))
                dmodeshapes[:,0] = self._compute_mode_shapes(x, y, z, dvec0)
                dmodeshapes[:,1] = self._compute_mode_shapes(x, y, z, dvec1)

                if 'mode_shapes_struct' in d_outputs:
                    if mode == 'fwd':
                        d_outputs['mode_shapes_struct'] += dmodeshapes * d_inputs['pitch_frequency']
                    if mode == 'rev':
                        d_inputs['pitch_frequency'] += np.sum(dmodeshapes * d_outputs['mode_shapes_struct'])

        if 'x_struct0' in d_inputs:
            if 'mode_shapes_struct' in d_outputs:
                mass, stiff, vec0, vec1 = self._structural_eigen_decomposition(inputs['plunge_frequency'], inputs['pitch_frequency'] )
                if mode == 'fwd':
                    d_outputs['mode_shapes_struct'][2::3,0] -= vec0[1] * d_inputs['x_struct0'][::3]
                    d_outputs['mode_shapes_struct'][2::3,1] -= vec1[1] * d_inputs['x_struct0'][::3]
                if mode == 'rev':
                    d_inputs['x_struct0'][::3] -= vec0[1] * d_outputs['mode_shapes_struct'][2::3,0]
                    d_inputs['x_struct0'][::3] -= vec1[1] * d_outputs['mode_shapes_struct'][2::3,1]







    def _compute_mode_shapes(self,x,y,z,vec):
        a = -2.0      # nondimensional elastic axis location measured from the midchord

        # mode 1 - plunge
        xmd = np.zeros_like(x)
        ymd = np.zeros_like(y)
        zmd = -vec[0]

        # mode 2 - linearized pitch about elastic axis
        zmd += vec[1] * ( 0.5 + a * self.b - x)
        return np.c_[xmd, ymd, zmd].flatten()

    def _structural_eigen_decomposition(self, omega_h, omega_alpha):
        # Isogai properties

        # Dimensional mass like terms (see kiviaho matrix pencil note 2019)
        m = self.mu * self.rho * np.pi * self.b**2.0
        Ia = self.ra2 * m * self.b**2.0
        Sa = self.unbal * m * self.b

        # Mass and Stiffness matrix
        M = np.zeros((2,2))
        M[0,0] = m
        M[0,1] = Sa
        M[1,0] = Sa
        M[1,1] = Ia

        K = np.zeros((2,2))
        K[0,0] = m * omega_h**2.0
        K[1,1] = Ia * omega_alpha**2.0

        lam, vec = linalg.eig(K,M)
        ind = np.argsort(lam)
        vec = vec[:,ind]
        for i in range(2):
            val = vec[np.argmax(np.abs(vec[:,i])),i]
            vec[:,i] = vec[:,i] * val / np.abs(val)

        # scale the eigenvectors to get unit modal mass
        mass  = np.zeros(2)
        stiff = np.zeros(2)
        for i in range(2):
            mass[i] = M @ vec[:,i] @ vec[:,i]
            vec[:,i] /= np.sqrt(mass[i])
        for i in range(2):
            mass[i] = M @ vec[:,i] @ vec[:,i]
            stiff[i] = K @ vec[:,i] @ vec[:,i]

        for i in range(2):
            #print('Structural Decomposition: mode=', i, 'freq=', np.sqrt(lam[i]), 'shape=',vec[:,i], 'mass=', mass[i], 'stiff=',stiff[i], 'freq2=',np.sqrt(stiff[i]))
            print('Structural Decomposition: mode=', i, 'stiff=',stiff[i])

        return mass, stiff, vec[:,0], vec[:,1]

# Static loads solver in the modal space
class ModalSolver(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nmodes',default=1)
    def setup(self):
        #self.set_check_partial_options(wrt='*',directional=True)

        nmodes = self.options['nmodes']

        self.add_input('modal_stiffness', shape_by_conn=True, desc = 'modal stiffness', tags=['mphys_coupling'])
        self.add_input('mf', shape_by_conn=True, desc = 'modal force')

        self.add_output('z', shape=nmodes,
                             val=np.ones(nmodes),
                             desc = 'modal displacement')

    def compute(self,inputs,outputs):
        k = inputs['modal_stiffness']
        outputs['z'] = inputs['mf'] / k

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        k = inputs['modal_stiffness']
        if mode == 'fwd':
            if 'z' in d_outputs:

                if 'mf' in d_inputs:
                    d_outputs['z'] += d_inputs['mf'] / k

                if 'modal_stiffness' in d_inputs:
                    d_outputs['z'] += - inputs['mf'] / (k**2.0) * d_inputs['modal_stiffness']

        if mode == 'rev':
            if 'z' in d_outputs:

                if 'mf' in d_inputs:
                    d_inputs['mf'] += d_outputs['z'] / k

                if 'modal_stiffness' in d_inputs:
                    d_inputs['modal_stiffness'] += - inputs['mf'] / (k**2.0) * d_outputs['z']


# Reduces force vector to the modal space
class ModalForces(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('load_size')
        self.options.declare('nmodes')

    def setup(self):
        #self.set_check_partial_options(wrt='*',directional=True)

        self.load_size = self.options['load_size']
        self.nmodes = self.options['nmodes']

        self.add_input('mode_shapes_struct', shape_by_conn=True, distributed=True, desc='structural mode shapes')
        self.add_input('f_struct',shape_by_conn=True, distributed=True, tags=['mphys_coupling'], desc = 'nodal force')

        self.add_output('mf',np.zeros(self.nmodes), desc = 'modal force')

    def compute(self,inputs,outputs):
        outputs['mf'][:] = 0.0
        if self.load_size > 0:
            for imode in range(self.nmodes):
                outputs['mf'][imode] = np.sum(inputs['mode_shapes_struct'][:,imode] * inputs['f_struct'][:])
        outputs['mf'] = self.comm.allreduce(outputs['mf'], op=MPI.SUM)

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode=='fwd':
            if 'mf' in d_outputs:
                if self.load_size > 0:

                    if 'f_struct' in d_inputs:
                        for imode in range(self.options['nmodes']):
                            d_outputs['mf'][imode] += np.sum(inputs['mode_shapes_struct'][:,imode] * d_inputs['f_struct'][:])

                    if 'mode_shapes_struct' in d_inputs:
                            for imode in range(self.options['nmodes']):
                                d_outputs['mf'][imode] += np.sum(d_inputs['mode_shapes_struct'][:,imode] * inputs['f_struct'][:])

                d_outputs['mf'] = self.comm.bcast(d_outputs['mf'])

        if mode=='rev':
            if 'mf' in d_outputs:
                if self.load_size > 0:

                    if 'f_struct' in d_inputs:
                        for imode in range(self.options['nmodes']):
                            d_inputs['f_struct'][:] += inputs['mode_shapes_struct'][:,imode] * d_outputs['mf'][imode]

                    if 'mode_shapes_struct' in d_inputs:
                        for imode in range(self.options['nmodes']):
                            d_inputs['mode_shapes_struct'][:,imode] += inputs['f_struct'][:] * d_outputs['mf'][imode]


# Projects deformation vector back to the full-order space
class ModalDisplacements(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('nmodes')
        self.options.declare('mode_size')

    def setup(self):
        #self.set_check_partial_options(wrt='*',directional=True)

        self.nmodes    = self.options['nmodes']
        self.mode_size = self.options['mode_size']

        self.add_input('mode_shapes_struct', shape_by_conn=True, distributed=True, desc='structural mode shapes', tags=['mphys_coupling'])
        self.add_input('z', shape_by_conn=True, desc='modal displacement')

        self.add_output('u_struct', np.zeros(self.mode_size), distributed=True, desc = 'nodal displacement', tags=['mphys_coupling'])

    def compute(self,inputs,outputs):
        if self.mode_size > 0:
            outputs['u_struct'] = np.zeros(self.mode_size)
            for imode in range(self.nmodes):
                outputs['u_struct'] += inputs['mode_shapes_struct'][:,imode] * inputs['z'][imode]

    def compute_jacvec_product(self,inputs,d_inputs,d_outputs,mode):
        if mode=='fwd':
            if 'u_struct' in d_outputs:
                if self.mode_size > 0:

                    if 'z' in d_inputs:
                        for imode in range(self.options['nmodes']):
                            d_outputs['u_struct'][:] += inputs['mode_shapes_struct'][:,imode] * d_inputs['z'][imode]

                    if 'mode_shapes_struct' in d_inputs:
                        for imode in range(self.options['nmodes']):
                            d_outputs['u_struct'][:] += d_inputs['mode_shapes_struct'][:,imode] * inputs['z'][imode]

                d_outputs['u_struct'] = self.comm.bcast(d_outputs['u_struct'])

        if mode=='rev':
            if 'u_struct' in d_outputs:

                if 'z' in d_inputs:
                    if self.mode_size > 0:
                        for imode in range(self.options['nmodes']):
                            d_inputs['z'][imode] += np.sum(inputs['mode_shapes_struct'][:,imode] * d_outputs['u_struct'][:])
                    d_inputs['z'] = self.comm.allreduce(d_inputs['z'], op=MPI.SUM)

                if 'mode_shapes_struct' in d_inputs:
                    if self.mode_size > 0:
                        for imode in range(self.options['nmodes']):
                            d_inputs['mode_shapes_struct'][:,imode] += inputs['z'][imode] * d_outputs['u_struct'][:]

class ModalSolverGroup(om.Group):
    def setup(self):
        nmodes = 2
        nnodes = 30

        if self.comm.Get_rank() == 0:
            load_size = nnodes * 3
        else:
            load_size = 0
        mode_size = load_size

        self.add_subsystem('modal_forces', ModalForces(
            load_size=load_size,
            nmodes=nmodes),
            promotes_inputs=['mode_shapes_struct', 'f_struct']
        )

        self.add_subsystem('modal_solver', ModalSolver(
            nmodes=nmodes),
            promotes_inputs=['modal_stiffness']
        )

        self.add_subsystem('modal_disps', ModalDisplacements(
            nmodes=nmodes,
            mode_size=mode_size),
            promotes_inputs=['mode_shapes_struct'],
            promotes_outputs=['u_struct']
        )

        self.connect('modal_forces.mf', 'modal_solver.mf')
        self.connect('modal_solver.z', 'modal_disps.z')

class IsogaiStructBuilder(Builder):
    def __init__(self):
        self.nmodes = 2
        self.number_of_nodes = 30
        self.ndof = 3
        self.check_partials = False

    def initialize(self, comm):
        self.comm = comm

    def get_number_of_nodes(self):
        return self.number_of_nodes if self.comm.Get_rank() == 0 else 0

    def get_ndof(self):
        return self.ndof

    def get_mesh_coordinate_subsystem(self):
        return IsogaiStructMesh()

    def get_pre_coupling_subsystem(self):
        return IsogaiEigenComp()

    def get_coupling_group_subsystem(self):
        return ModalSolverGroup()

if __name__ == '__main__':
    np.random.seed(0)
    nnodes = 30 if MPI.COMM_WORLD.Get_rank() == 0 else 0
    prob = om.Problem()
    prob.model.add_subsystem('mesh',IsogaiStructMesh(), promotes=['*'])
    ivc = prob.model.add_subsystem('ivc',om.IndepVarComp(),promotes=['*'])
    ivc.add_output('pitch_frequency', 100.0)
    ivc.add_output('plunge_frequency', 100.0)
    ivc.add_output('f_struct', np.random.rand(3*nnodes), distributed=True)
    prob.model.add_subsystem('iso',IsogaiEigenComp(),promotes=['*'])
    prob.model.add_subsystem('coupling', ModalSolverGroup(),promotes=['*'])

    #prob.setup()
    prob.setup(mode='rev')
    om.n2(prob, show_browser=False)
    prob.run_model()
    #prob['pitch_frequency'] += 1e-6
    #prob.run_model()
    prob.check_partials(compact_print=True,step = 1e-6)
    #prob.check_totals(of=['modal_solver.z'],wrt=['plunge_frequency'])
