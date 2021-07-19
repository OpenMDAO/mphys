import numpy as np
import sys, time
from mpi4py import MPI
from openmdao.api import Group, ImplicitComponent, ExplicitComponent, AnalysisError
from dafoam import PYDAFOAM
from idwarp import USMesh
from mphys.builder import Builder
import petsc4py
from petsc4py import PETSc

petsc4py.init(sys.argv)

np.set_printoptions(precision=16, suppress=True)


class DAFoamBuilder(Builder):
    """
    DAFoam builder called from runScript.py
    """

    def __init__(
        self,
        options,  # DAFoam options
        mesh_options=None,  # IDWarp options
        scenario="aerodynamic",  # scenario type to configure the groups
    ):

        # options dictionary for DAFoam
        self.options = options

        # mesh warping option
        if mesh_options is None:
            raise AnalysisError("mesh_options not found!")
        else:
            self.mesh_options = mesh_options

        # check scenario
        if scenario != "aerodynamic":
            raise AnalysisError("scenario not valid! Option: aerodynamic")

    # api level method for all builders
    def initialize(self, comm):
        # initialize the PYDAFOAM class, defined in pyDAFoam.py
        self.DASolver = PYDAFOAM(options=self.options, comm=comm)
        # always set the mesh
        mesh = USMesh(options=self.mesh_options, comm=comm)
        self.DASolver.setMesh(mesh)
        # add the design surface family group
        self.DASolver.addFamilyGroup(
            self.DASolver.getOption("designSurfaceFamily"), self.DASolver.getOption("designSurfaces")
        )
        self.DASolver.printFamilyList()

    def get_solver(self):
        # this method is only used by the RLT transfer scheme
        return self.DASolver

    # api level method for all builders
    def get_coupling_group_subsystem(self):
        dafoam_group = DAFoamGroup(solver=self.DASolver)
        return dafoam_group

    def get_mesh_coordinate_subsystem(self):

        # just return the component that outputs the surface mesh.
        return DAFoamMesh(solver=self.DASolver)

    def get_pre_coupling_subsystem(self):
        # we warp as a pre-processing step
        return DAFoamWarper(solver=self.DASolver)

    def get_post_coupling_subsystem(self):
        return DAFoamFunctions(solver=self.DASolver)

    # TODO the get_nnodes is deprecated. will remove
    def get_nnodes(self, groupName=None):
        return int(self.DASolver.getSurfaceCoordinates(groupName=groupName).size / 3)

    def get_number_of_nodes(self, groupName=None):
        return int(self.DASolver.getSurfaceCoordinates(groupName=groupName).size / 3)


class DAFoamGroup(Group):
    """
    DAFoam solver group
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]

        # add the solver implicit component
        self.add_subsystem(
            "solver",
            DAFoamSolver(solver=self.DASolver),
            promotes_inputs=["dafoam_vol_coords","dafoam_aoa"],
            promotes_outputs=["dafoam_states"],
        )

    # connect the input and output for the solver, called from runScript.py
    def mphys_set_dvs_and_cons(self):

        # promote the DVs
        DVNames, _ = self.DASolver.getDVsCons()

        for DVName in DVNames:
            self.promotes("solver", inputs=[DVName])

    # set the DVGeo for DASolver, called from runScript.py
    def mphys_set_dvgeo(self, DVGeo):
        self.DASolver.setDVGeo(DVGeo)


class DAFoamSolver(ImplicitComponent):
    """
    OpenMDAO component that wraps the DAFoam flow and adjoint solvers
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):
        self.DASolver = self.options["solver"]

        # always run coloring
        self.DASolver.runColoring()

        # determine which function to compute the adjoint
        self.evalFuncs = []
        self.DASolver.setEvalFuncs(self.evalFuncs)

        # setup input and output for the solver
        local_state_size = self.DASolver.getNLocalAdjointStates()
        self.add_input("dafoam_vol_coords", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_input("dafoam_aoa", distributed=True, shape=(1), tags=["mphys_coupling"])
        self.add_output("dafoam_states", distributed=True, shape=local_state_size, tags=["mphys_coupling"])

    # calculate the residual
    def apply_nonlinear(self, inputs, outputs, residuals):
        DASolver = self.DASolver
        DASolver.setStates(outputs["dafoam_states"])

        # get flow residuals from DASolver
        residuals["dafoam_states"] = DASolver.getResiduals()

    # solve the flow
    def solve_nonlinear(self, inputs, outputs):
        DASolver = self.DASolver
        xDVs = DASolver.DVGeo.getValues()
        Info("\n")
        Info("+--------------------------------------------------------------------------+")
        Info("|                  Evaluating Objective Functions %03d                      |" % DASolver.nSolvePrimals)
        Info("+--------------------------------------------------------------------------+")
        Info("Design Variables: ")
        Info(xDVs)

        # set the runStatus, this is useful when the actuator term is activated
        DASolver.setOption("runStatus", "solvePrimal")
        DASolver.updateDAOption()

        # solve the flow with the current design variable
        DASolver()

        # get the objective functions
        funcs = {}
        DASolver.evalFunctions(funcs, evalFuncs=self.evalFuncs)
        Info("Objective Functions: ")
        Info(funcs)

        # assign the computed flow states to outputs
        outputs["dafoam_states"] = DASolver.getStates()

    def linearize(self, inputs, outputs, residuals):
        # NOTE: we do not do any computation in this function, just print some information

        DASolver = self.DASolver

        Info("\n")
        Info("+--------------------------------------------------------------------------+")
        Info("|              Evaluating Objective Function Sensitivities %03d             |" % DASolver.nSolveAdjoints)
        Info("+--------------------------------------------------------------------------+")

        # move the solution folder to 0.000000x
        DASolver.renameSolution(DASolver.nSolveAdjoints)

        Info("Running adjoint Solver %03d" % DASolver.nSolveAdjoints)

        # set the runStatus, this is useful when the actuator term is activated
        DASolver.setOption("runStatus", "solveAdjoint")
        DASolver.updateDAOption()

        DASolver.nSolveAdjoints += 1

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # compute the matrix vector products for states and volume mesh coordinates
        # i.e., dRdWT*psi, dRdXvT*psi

        # we do not support forward mode
        if mode == "fwd":
            raise AnalysisError("fwd mode not implemented!")

        DASolver = self.DASolver

        # assign the states in outputs to the OpenFOAM flow fields
        DASolver.setStates(outputs["dafoam_states"])

        if "dafoam_states" in d_residuals:

            # get the reverse mode AD seed from d_residuals
            resBar = d_residuals["dafoam_states"]
            # convert the seed array to Petsc vector
            resBarVec = DASolver.array2Vec(resBar)

            # this computes [dRdW]^T*Psi using reverse mode AD
            if "dafoam_states" in d_outputs:
                prodVec = DASolver.wVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdRdWTPsiAD(DASolver.xvVec, DASolver.wVec, resBarVec, prodVec)
                wBar = DASolver.vec2Array(prodVec)
                d_outputs["dafoam_states"] += wBar

            # this computes [dRdXv]^T*Psi using reverse mode AD
            if "dafoam_vol_coords" in d_inputs:
                prodVec = DASolver.xvVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdRdXvTPsiAD(DASolver.xvVec, DASolver.wVec, resBarVec, prodVec)
                xVBar = DASolver.vec2Array(prodVec)
                d_inputs["dafoam_vol_coords"] += xVBar

            if "dafoam_aoa" in d_inputs:
                prodVec = DASolver.xvVec.duplicate()
                prodVec.zeroEntries()
                DASolver.solverAD.calcdRdAOATPsiAD(DASolver.xvVec, DASolver.wVec, resBarVec, "alpha".encode(), prodVec)
                xVBar = DASolver.vec2Array(prodVec)
                d_inputs["dafoam_aoa"] += xVBar

        # NOTE: we only support states, vol_coords partials, and angle of attack. 
        # Other variables such as angle of attack, is not implemented yet!

    def solve_linear(self, d_outputs, d_residuals, mode):
        # solve the adjoint equation [dRdW]^T * Psi = dFdW

        # we do not support forward mode
        if mode == "fwd":
            raise AnalysisError("fwd mode not implemented!")

        DASolver = self.DASolver

        # we only support JacobianFree because we need to use AD
        if DASolver.getOption("adjJacobianOption") != "JacobianFree":
            raise AnalysisError("adjJacobianOption not valid! Only support JacobianFree!")

        # initialize the dRdWT matrix-free matrix in DASolver
        DASolver.solverAD.initializedRdWTMatrixFree(DASolver.xvVec, DASolver.wVec)

        # compute the preconditioiner matrix for the adjoint linear equation solution
        # NOTE: we compute this only once and will reuse it during optimization
        if DASolver.dRdWTPC is None:
            DASolver.dRdWTPC = PETSc.Mat().create(PETSc.COMM_WORLD)
            DASolver.solver.calcdRdWT(DASolver.xvVec, DASolver.wVec, 1, DASolver.dRdWTPC)

        # create the Petsc KSP object
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        DASolver.solverAD.createMLRKSPMatrixFree(DASolver.dRdWTPC, ksp)

        # solution vector
        psi = DASolver.wVec.duplicate()
        psi.zeroEntries()
        # right hand side array from d_outputs
        dFdWArray = d_outputs["dafoam_states"]
        # convert the array to vector
        dFdW = DASolver.array2Vec(dFdWArray)
        # actually solving the adjoint linear equation using Petsc
        DASolver.solverAD.solveLinearEqn(ksp, dFdW, psi)
        # convert the solution vector to array and assign it to d_residuals
        d_residuals["dafoam_states"] = DASolver.vec2Array(psi)

        return True, 0, 0


class DAFoamMesh(ExplicitComponent):
    """
    Component to get the partitioned initial surface mesh coordinates
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]

        # design surface coordinates
        self.x_a0 = self.DASolver.getSurfaceCoordinates(self.DASolver.designFamilyGroup).flatten(order="C")

        # add output
        coord_size = self.x_a0.size
        self.add_output(
            "x_aero0",
            distributed=True,
            shape=coord_size,
            desc="initial aerodynamic surface node coordinates",
            tags=["mphys_coordinates"],
        )

    def mphys_add_coordinate_input(self):
        self.add_input(
            "x_aero0_points", distributed=True, shape_by_conn=True, desc="aerodynamic surface with geom changes"
        )

        # return the promoted name and coordinates
        return "x_aero0_points", self.x_a0

    def mphys_get_surface_mesh(self):
        return self.x_a0

    def mphys_get_triangulated_surface(self, groupName=None):
        # this is a list of lists of 3 points
        # p0, v1, v2

        return self.DASolver.getTriangulatedMeshSurface()

    def compute(self, inputs, outputs):
        # just assign the surface mesh coordinates
        if "x_aero0_points" in inputs:
            outputs["x_aero0"] = inputs["x_aero0_points"]
        else:
            outputs["x_aero0"] = self.x_a0

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # we do not support forward mode AD
        if mode == "fwd":
            raise AnalysisError("fwd mode not implemented!")

        # just assign the matrix-vector product
        if "x_aero0_points" in d_inputs:
            d_inputs["x_aero0_points"] += d_outputs["x_aero0"]


class DAFoamFunctions(ExplicitComponent):
    """
    DAFoam objective and constraint functions component
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

        # a list that contains all function names, e.g., CD, CL
        self.funcs = None

        self.nProcs = MPI.COMM_WORLD.size

    def setup(self):

        self.DASolver = self.options["solver"]

        self.solution_counter = 0

        self.add_input("dafoam_vol_coords", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_input("dafoam_aoa", distributed=True, shape=(1), tags=["mphys_coupling"])
        self.add_input("dafoam_states", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

    # connect the input and output for the function, called from runScript.py
    def mphys_set_dvs_and_cons(self):

        DVNames, DVSizes = self.DASolver.getDVsCons()

        # parameter inputs
        for idxI, DVName in enumerate(DVNames):
            DVSize = DVSizes[idxI]
            self.add_input(DVName, distributed=False, shape=DVSize, units=None, tags=["mphys_input"])

    # add the function names to this component, called from runScript.py
    def mphys_add_funcs(self, funcs):

        self.funcs = funcs

        # loop over the functions here and create the output
        for f_name in funcs:
            self.add_output(f_name, distributed=False, shape=1, units=None, tags=["mphys_result"])

    def _set_states(self, inputs):
        self.DASolver.setStates(inputs["dafoam_states"])

    # get the objective function from DASolver
    def compute(self, inputs, outputs):
        DASolver = self.DASolver

        funcs = {}

        if self.funcs is not None:
            DASolver.evalFunctions(funcs, evalFuncs=self.funcs)
            for f_name in self.funcs:
                if f_name in funcs:
                    outputs[f_name] = funcs[f_name]

    # compute the partial derivatives of functions
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        DASolver = self.DASolver

        # we do not support forward mode AD
        if mode == "fwd":
            raise AnalysisError("fwd not implemented!")

        funcsBar = {}

        # assign value to funcsBar. NOTE: we only assign seed if d_outputs has
        # non-zero values!
        if self.funcs is None:
            raise AnalysisError("functions not set! Forgot to call mphys_add_funcs?")
        else:
            for func_name in self.funcs:
                if func_name in d_outputs and d_outputs[func_name] != 0.0:
                    funcsBar[func_name] = d_outputs[func_name][0]

        # funcsBar should have only one seed for which we need to compute partials
        # Info(funcsBar)

        # get the name of the functions we need to compute partials for
        objFuncName = list(funcsBar.keys())[0]

        # compute dFdW
        if "dafoam_states" in d_inputs:
            dFdW = DASolver.wVec.duplicate()
            dFdW.zeroEntries()
            DASolver.solverAD.calcdFdWAD(DASolver.xvVec, DASolver.wVec, objFuncName.encode(), dFdW)
            # *************************************************************************************
            # NOTE: here we need to divide dFdW by the total number of CPU cores because in DAFoam
            # the dFdW is already MPI.Reduce from all processors, however, it seems that OM requires
            # dFdW that belongs to each proc. So we need to divide dFdW by self.nProcs and then
            # assign it to wBar for OM
            wBar = DASolver.vec2Array(dFdW) / self.nProcs
            d_inputs["dafoam_states"] += wBar

        # compute dFdXv
        if "dafoam_vol_coords" in d_inputs:
            dFdXv = DASolver.xvVec.duplicate()
            dFdXv.zeroEntries()
            DASolver.solverAD.calcdFdXvAD(DASolver.xvVec, DASolver.wVec, objFuncName.encode(), "dummy".encode(), dFdXv)
            # *************************************************************************************
            # NOTE: here we need to divide dFdXv by the total number of CPU cores because in DAFoam
            # the dFdXv is already MPI.Reduce from all processors, however, it seems that OM requires
            # dFdXv that belongs to each proc. So we need to divide dFdXv by self.nProcs and then
            # assign it to xVBar for OM
            xVBar = DASolver.vec2Array(dFdXv) / self.nProcs
            d_inputs["dafoam_vol_coords"] += xVBar

        # compute dFdAOA
        if "dafoam_aoa" in d_inputs:
            dFdAOA = DASolver.xvVec.duplicate()
            dFdAOA.zeroEntries()
            DASolver.solverAD.calcdFdAOAAD(DASolver.xvVec, DASolver.wVec, objFuncName.encode(), "alpha".encode(), dFdAOA)
            # TODO: check the following:
            # *************************************************************************************
            # NOTE: here we need to divide dFdAOA by the total number of CPU cores because in DAFoam
            # the dFdAOA is already MPI.Reduce from all processors, however, it seems that OM requires
            # dFdAOA that belongs to each proc. So we need to divide dFdAOA by self.nProcs and then
            # assign it to xVBar for OM
            xVBar = DASolver.vec2Array(dFdAOA) / self.nProcs
            d_inputs["dafoam_aoa"] += xVBar

        # NOTE: we only support states, vol_coords partials, and angle of attack. 
        # Other variables such as angle of attack, is not implemented yet!


class DAFoamWarper(ExplicitComponent):
    """
    OpenMDAO component that wraps the warping.
    """

    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]
        DASolver = self.DASolver

        # state inputs and outputs
        local_volume_coord_size = DASolver.mesh.getSolverGrid().size

        self.add_input("x_aero", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_output("dafoam_vol_coords", distributed=True, shape=local_volume_coord_size, tags=["mphys_coupling"])

    def compute(self, inputs, outputs):
        # given the new surface mesh coordinates, compute the new volume mesh coordinates
        # the mesh warping will be called in getSolverGrid()
        DASolver = self.DASolver

        x_a = inputs["x_aero"].reshape((-1, 3))
        DASolver.setSurfaceCoordinates(x_a)
        outputs["dafoam_vol_coords"] = DASolver.mesh.getSolverGrid()

    # compute the mesh warping products in IDWarp
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # we do not support forward mode AD
        if mode == "fwd":
            raise AnalysisError("fwd not implemented!")

        # compute dXv/dXs such that we can propagate the partials (e.g., dF/dXv) to Xs
        # then the partial will be further propagated to XFFD in pyGeo
        if "dafoam_vol_coords" in d_outputs:
            if "x_aero" in d_inputs:
                dxV = d_outputs["dafoam_vol_coords"]
                self.DASolver.mesh.warpDeriv(dxV)
                dxS = self.DASolver.mesh.getdXs()
                dxS = self.DASolver.mapVector(dxS, self.DASolver.meshFamilyGroup, self.DASolver.designFamilyGroup)
                d_inputs["x_aero"] += dxS.flatten()


class Info(object):
    """
    Print information and flush to screen for parallel cases
    """

    def __init__(self, message):
        if MPI.COMM_WORLD.rank == 0:
            print(message, flush=True)
        MPI.COMM_WORLD.Barrier()
