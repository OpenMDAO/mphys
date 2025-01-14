# A demonstration of basic functions of the Python interface for TACS: loading a
# mesh, creating elements, evaluating functions, solution, and output
from __future__ import print_function

# Import necessary libraries
import numpy as np
from mpi4py import MPI
from pyOpt import Optimization, pySLSQP
from tacs import TACS, constitutive, elements, functions


class TacsOpt(object):
    def __init__(self):

        self.x0 = 0.0031
        self.var_scale = 1.0
        self.c1 = 1.0/100000.0
        self.c2 = 1.0 / self.var_scale

        # Load structural mesh from BDF file
        tacs_comm = MPI.COMM_WORLD
        struct_mesh = TACS.MeshLoader(tacs_comm)
        struct_mesh.scanBDFFile("CRM_box_2nd.bdf")

        # Set constitutive properties
        rho = 2500.0  # density, kg/m^3
        E = 70.0e9 # elastic modulus, Pa
        nu = 0.3 # poisson's ratio
        kcorr = 5.0 / 6.0 # shear correction factor
        ys = 350e6  # yield stress, Pa
        t= 0.013
        min_thickness = 0.00
        max_thickness = 1.00

        # Loop over components, creating stiffness and element object for each
        num_components = struct_mesh.getNumComponents()
        for i in range(num_components):
            descriptor = struct_mesh.getElementDescript(i)
            stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, t, i,
                                         min_thickness, max_thickness)
            element = None
            if descriptor in ["CQUAD", "CQUADR", "CQUAD4"]:
                element = elements.MITCShell(2, stiff, component_num=i)
            struct_mesh.setElement(i, element)

        # Create tacs assembler object from mesh loader
        self.tacs = struct_mesh.createTACS(6)
        tacs = self.tacs

        # Create the KS Function
        ksWeight = 100.0
        self.funcs = [functions.KSFailure(tacs, ksWeight),functions.StructuralMass(tacs)]

        # Get the design variable values
        x = np.zeros(num_components, TACS.dtype)
        tacs.getDesignVars(x)

        # Create the forces
        self.forces = tacs.createVec()
        force_array = self.forces.getArray()
        force_array[2::6] += 100.0 # uniform load in z direction
        tacs.applyBCs(self.forces)

        self.res = tacs.createVec()
        self.ans = tacs.createVec()
        self.adjoint = tacs.createVec()
        self.mat = tacs.createFEMat()

        self.pc = TACS.Pc(self.mat)
        subspace = 100
        restarts = 2
        self.gmres = TACS.KSM(self.mat, self.pc, subspace, restarts)

    def solve(self,x):
        tacs  = self.tacs
        funcs = self.funcs
        res   = self.res
        mat   = self.mat
        ans   = self.ans

        print('x',x[:5]*self.var_scale)
        # Set up and solve the analysis problem
        tacs.setDesignVars(x*self.var_scale)


        # Assemble the Jacobian and factor
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        tacs.zeroVariables()
        tacs.assembleJacobian(alpha, beta, gamma, res, mat)
        self.pc.factor()

        # Solve the linear system
        self.gmres.solve(self.forces, ans)
        tacs.setVariables(ans)

        # Evaluate the functions
        fvals = tacs.evalFunctions(funcs)
        obj = np.zeros(1)
        con = np.zeros(1)

        obj[:] = self.c1 * fvals[1]
        con[:] = self.c2 * (1.5*fvals[0] - 1.0)

        fail = 0

        # Output for visualization
        flag = (TACS.ToFH5.NODES |
                TACS.ToFH5.DISPLACEMENTS |
                TACS.ToFH5.STRAINS |
                TACS.ToFH5.EXTRAS)
        f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
        f5.writeToFile('ucrm.f5')
        print('obj',obj)
        print('con',con)

        return obj, con, fail

    def gradient(self,x,obj,con):
        tacs  = self.tacs
        funcs = self.funcs
        res   = self.res
        mat   = self.mat
        ans   = self.ans
        adjoint = self.adjoint

        # Solve for the adjoint variables
        tacs.evalSVSens(funcs[0], res)
        res_array = res.getArray()
        res_array *= -1.0
        self.gmres.solve(res, adjoint)

        # Compute the total derivative w.r.t. material design variables
        fdvSens = np.zeros(x.shape, TACS.dtype)
        product = np.zeros(x.shape, TACS.dtype)
        tacs.evalDVSens(funcs[0], fdvSens)
        tacs.evalAdjointResProduct(adjoint, product)
        fdvSens = fdvSens + product

        A = np.zeros((1,240))
        A[:,:] = self.c2 * 1.5 * fdvSens * self.var_scale

        tacs.evalDVSens(funcs[1], fdvSens)
        g = np.zeros((1,240))
        g[:,:] = self.c1 * fdvSens * self.var_scale

        fail = 0

        return g, A, fail

dp = TacsOpt()

opt_prob = Optimization('crm_togw',dp.solve)

opt_prob.addObj('mass')
opt_prob.addCon('ksfailure',type='i')

for i in range(240):
    opt_prob.addVar('thickness '+str(i),type='c',value=dp.x0/dp.var_scale,
                                                 lower=0.001/dp.var_scale,
                                                 upper=0.075/dp.var_scale)

opt = pySLSQP.SLSQP(pll_type='POA')
opt.setOption('MAXIT',999)
opt(opt_prob,sens_type=dp.gradient,disp_opts=True)
