import numpy as np
import sys
from mpi4py import MPI
from openmdao.api import Group, ImplicitComponent, ExplicitComponent, AnalysisError
from dafoam import PYDAFOAM
from idwarp import USMesh
from mphys.builder import Builder
import petsc4py
from petsc4py import PETSc

petsc4py.init(sys.argv)


class DAFoamBuilder(Builder):
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
        self.DASolver = PYDAFOAM(options=self.options, comm=comm)
        mesh = USMesh(options=self.mesh_options, comm=comm)
        self.DASolver.setMesh(mesh)
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
    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.DASolver = self.options["solver"]

        self.add_subsystem(
            "solver",
            DAFoamSolver(solver=self.DASolver),
            promotes_inputs=["dafoam_vol_coords"],
            promotes_outputs=["dafoam_states"],
        )

    def mphys_set_dvs_and_cons(self):

        # promote the DVs
        DVNames, _ = self.DASolver.getDVsCons()

        for DVName in DVNames:
            self.promotes("solver", inputs=[DVName])

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

        self.DASolver.runColoring()

        local_state_size = self.DASolver.getNLocalAdjointStates()

        self.add_input("dafoam_vol_coords", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_output("dafoam_states", distributed=True, shape=local_state_size, tags=["mphys_coupling"])

    def setDVGeo(self, DVGeo):
        self.DASolver.setDVGeo(DVGeo)

    def apply_nonlinear(self, inputs, outputs, residuals):
        DASolver = self.DASolver
        DASolver.setStates(outputs["dafoam_states"])

        # flow residuals
        residuals["dafoam_states"] = DASolver.getResiduals()

    def solve_nonlinear(self, inputs, outputs):
        DASolver = self.DASolver
        DASolver()

        outputs["dafoam_states"] = DASolver.getStates()

    def linearize(self, inputs, outputs, residuals):
        pass

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):

        DASolver = self.DASolver

        DASolver.setStates(outputs["dafoam_states"])

        if mode == "fwd":

            raise AnalysisError("fwd mode not implemented!")

        elif mode == "rev":
            if "dafoam_states" in d_residuals:

                resBar = d_residuals["dafoam_states"]
                resBarVec = DASolver.array2Vec(resBar)

                if "dafoam_states" in d_outputs:
                    prodVec = DASolver.wVec.duplicate()
                    prodVec.zeroEntries()
                    DASolver.solverAD.calcdRdWTPsiAD(DASolver.xvVec, DASolver.wVec, resBarVec, prodVec)
                    wBar = DASolver.vec2Array(prodVec)

                    d_outputs["dafoam_states"] += wBar

                if "dafoam_vol_coords" in d_inputs:
                    prodVec = DASolver.xvVec.duplicate()
                    prodVec.zeroEntries()
                    DASolver.solverAD.calcdRdXvTPsiAD(DASolver.xvVec, DASolver.wVec, resBarVec, prodVec)
                    xVBar = DASolver.vec2Array(prodVec)

                    d_inputs["dafoam_vol_coords"] += xVBar

                xDVs = DASolver.DVGeo.getValues()
                for dvName in xDVs:
                    dvDict = DASolver.getOption("designVar")[dvName]
                    dvType = dvDict["designVarType"]
                    if dvType == "AOA":
                        prodVec = DASolver.wVec.duplicate()
                        prodVec.zeroEntries()
                        DASolver.solverAD.calcdRdAOATPsiAD(
                            DASolver.xvVec, DASolver.wVec, resBarVec, dvName.encode(), prodVec
                        )
                        xDVBar = DASolver.vec2Array(prodVec)
                        d_inputs[dvName] += xDVBar.flatten()
                    elif dvType == "FFD":
                        # we have already handle that in xVBar
                        pass
                    else:
                        raise AnalysisError("dvType not implemented!")

    def solve_linear(self, d_outputs, d_residuals, mode):
        DASolver = self.DASolver

        if DASolver.getOption("adjJacobianOption") != "JacobianFree":
            raise AnalysisError("adjJacobianOption not valid! Only support JacobianFree!")

        Info("Solving linear in mphys_dafoam")

        if mode == "fwd":
            raise AnalysisError("fwd mode not implemented!")
        elif mode == "rev":
            DASolver.solverAD.initializedRdWTMatrixFree(DASolver.xvVec, DASolver.wVec)
            if DASolver.dRdWTPC is None:
                DASolver.dRdWTPC = PETSc.Mat().create(PETSc.COMM_WORLD)
                DASolver.solver.calcdRdWT(DASolver.xvVec, DASolver.wVec, 1, DASolver.dRdWTPC)

            ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
            DASolver.solverAD.createMLRKSPMatrixFree(DASolver.dRdWTPC, ksp)

            psi = DASolver.wVec.duplicate()
            psi.zeroEntries()

            dFdWArray = d_outputs["dafoam_states"]
            dFdW = DASolver.array2Vec(dFdWArray)

            DASolver.solverAD.solveLinearEqn(ksp, dFdW, psi)

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

        self.x_a0 = self.DASolver.getSurfaceCoordinates(self.DASolver.designFamilyGroup).flatten(order="C")

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
        if "x_aero0_points" in inputs:
            outputs["x_aero0"] = inputs["x_aero0_points"]
        else:
            outputs["x_aero0"] = self.x_a0

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            if "x_aero0_points" in d_inputs:
                d_outputs["x_aero0"] += d_inputs["x_aero0_points"]
        elif mode == "rev":
            if "x_aero0_points" in d_inputs:
                d_inputs["x_aero0_points"] += d_outputs["x_aero0"]


class DAFoamFunctions(ExplicitComponent):
    def initialize(self):
        self.options.declare("solver", recordable=False)

        self.extra_funcs = None

    def setup(self):

        self.DASolver = self.options["solver"]

        self.solution_counter = 0

        self.add_input("dafoam_vol_coords", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_input("dafoam_states", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

    def mphys_set_dvs_and_cons(self):

        DVNames, DVSizes = self.DASolver.getDVsCons()

        # parameter inputs
        for idxI, DVName in enumerate(DVNames):
            DVSize = DVSizes[idxI]
            self.add_input(DVName, distributed=False, shape=DVSize, units=None, tags=["mphys_input"])

    def mphys_add_funcs(self, funcs):

        self.extra_funcs = funcs

        # loop over the functions here and create the output
        for f_name in funcs:
            self.add_output(f_name, distributed=False, shape=1, units=None, tags=["mphys_result"])

    def _set_states(self, inputs):
        self.DASolver.setStates(inputs["dafoam_states"])

    def compute(self, inputs, outputs):
        DASolver = self.DASolver

        funcs = {}

        if self.extra_funcs is not None:
            DASolver.evalFunctions(funcs, evalFuncs=self.extra_funcs)
            for f_name in self.extra_funcs:
                if f_name in funcs:
                    outputs[f_name] = funcs[f_name]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        DASolver = self.DASolver

        if mode == "fwd":

            raise AnalysisError("fwd not implemented!")

        elif mode == "rev":
            funcsBar = {}

            # also do the same for extra functions
            if self.extra_funcs is not None:
                for func_name in self.extra_funcs:
                    if func_name in d_outputs and d_outputs[func_name] != 0.0:
                        funcsBar[func_name] = d_outputs[func_name][0]

            Info(funcsBar)

            objFuncName = list(funcsBar.keys())[0]

            if "dafoam_states" in d_inputs:
                dFdW = DASolver.wVec.duplicate()
                dFdW.zeroEntries()
                DASolver.solverAD.calcdFdWAD(DASolver.xvVec, DASolver.wVec, objFuncName.encode(), dFdW)
                wBar = DASolver.vec2Array(dFdW)
                d_inputs["dafoam_states"] += wBar
            if "dafoam_vol_coords" in d_inputs:
                dFdXv = DASolver.xvVec.duplicate()
                dFdXv.zeroEntries()
                DASolver.solverAD.calcdFdXvAD(DASolver.xvVec, DASolver.wVec, objFuncName.encode(), "dummy".encode(), dFdXv)
                xVBar = DASolver.vec2Array(dFdXv)
                d_inputs["dafoam_vol_coords"] += xVBar

            xDVs = DASolver.DVGeo.getValues()
            for dvName in xDVs:
                dvDict = DASolver.getOption("designVar")[dvName]
                dvType = dvDict["designVarType"]
                if dvType == "AOA":
                    dFdAOA = DASolver.wVec.duplicate()
                    dFdAOA.zeroEntries()
                    DASolver.calcdFdAOAAnalytical(dvName.encode(), dFdAOA)
                    xDVBar = DASolver.vec2Array(dFdAOA)
                    d_inputs[dvName] += xDVBar.flatten()
                elif dvType == "FFD":
                    # we have already handle that in xVBar
                    pass
                else:
                    raise AnalysisError("dvType not implemented!")


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

        DASolver = self.DASolver

        x_a = inputs["x_aero"].reshape((-1, 3))
        DASolver.setSurfaceCoordinates(x_a)
        outputs["dafoam_vol_coords"] = DASolver.mesh.getSolverGrid()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        if mode == "fwd":
            if "dafoam_vol_coords" in d_outputs:
                if "x_aero" in d_inputs:
                    dxS = d_inputs["x_aero"]
                    dxV = self.DASolver.mesh.warpDerivFwd(dxS)
                    d_outputs["dafoam_vol_coords"] += dxV

        elif mode == "rev":
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
