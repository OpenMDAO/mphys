import numpy as np
from mpi4py import MPI
from openmdao.api import Group, ImplicitComponent, ExplicitComponent, AnalysisError
from dafoam import PYDAFOAM
from idwarp import USMesh
from mphys.builder import Builder


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
        self.solver = PYDAFOAM(options=self.options, comm=comm)
        mesh = USMesh(options=self.mesh_options, comm=comm)
        self.solver.setMesh(mesh)
        self.solver.addFamilyGroup(self.solver.getOption("designSurfaceFamily"), self.solver.getOption("designSurfaces"))
        self.solver.printFamilyList()

    def get_solver(self):
        # this method is only used by the RLT transfer scheme
        return self.solver

    # api level method for all builders
    def get_coupling_group_subsystem(self):
        dafoam_group = DAFoamGroup(solver=self.solver)
        return dafoam_group

    def get_mesh_coordinate_subsystem(self):

        # TODO modify this so that we can move the volume mesh warping to the top level
        # we need this to do mesh warping only once for all serial points.
        # for now, volume warping is duplicated on all scenarios, which is not efficient

        # use_warper = not self.warp_in_solver
        # # if we do warper in the mesh element, we will do a group thing
        # if use_warper:
        #     return ADflowMeshGroup(aero_solver=self.solver)
        # else:

        # just return the component that outputs the surface mesh.
        return DAFoamMesh(aero_solver=self.solver)

    def get_pre_coupling_subsystem(self):
        # we warp as a pre-processing step
        return DAFoamWarper(aero_solver=self.solver)

    def get_post_coupling_subsystem(self):
        return DAFoamFunctions(aero_solver=self.solver)

    # TODO the get_nnodes is deprecated. will remove
    def get_nnodes(self, groupName=None):
        return int(self.solver.getSurfaceCoordinates(groupName=groupName).size / 3)

    def get_number_of_nodes(self, groupName=None):
        return int(self.solver.getSurfaceCoordinates(groupName=groupName).size / 3)


class DAFoamGroup(Group):
    def initialize(self):
        self.options.declare("solver", recordable=False)

    def setup(self):

        self.aero_solver = self.options["solver"]

        self.add_subsystem(
            "solver",
            DAFoamSolver(aero_solver=self.aero_solver),
            promotes_inputs=["dafoam_vol_coords"],
            promotes_outputs=["dafoam_states"],
        )

    def mphys_set_dvs_and_cons(self):

        # promote the DVs
        DVNames, _ = self.aero_solver.getDVsCons()

        for DVName in DVNames:
            self.promotes("solver", inputs=[DVName])
    
    def mphys_set_dvgeo(self, DVGeo):
        self.aero_solver.setDVGeo(DVGeo)


class DAFoamSolver(ImplicitComponent):
    """
    OpenMDAO component that wraps the DAFoam flow and adjoint solvers
    """

    def initialize(self):
        self.options.declare("aero_solver", recordable=False)

    def setup(self):
        self.solver = self.options["aero_solver"]

        local_state_size = self.solver.getNLocalAdjointStates()

        self.add_input("dafoam_vol_coords", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_output("dafoam_states", distributed=True, shape=local_state_size, tags=["mphys_coupling"])
    
    def setDVGeo(self, DVGeo):
        self.solver.setDVGeo(DVGeo)

    def apply_nonlinear(self, inputs, outputs, residuals):
        pass

    def solve_nonlinear(self, inputs, outputs):
        solver = self.solver
        solver()

        outputs["dafoam_states"] = solver.getStates()

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        pass

    def solve_linear(self, d_outputs, d_residuals, mode):
        pass


class DAFoamMesh(ExplicitComponent):
    """
    Component to get the partitioned initial surface mesh coordinates

    """

    def initialize(self):
        self.options.declare("aero_solver", recordable=False)

    def setup(self):

        self.aero_solver = self.options["aero_solver"]

        # no argument is given to getSurfaceCoordinates so it will use allWallFamily
        self.x_a0 = self.aero_solver.getSurfaceCoordinates().flatten(order="C")

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

        return self.aero_solver.getTriangulatedMeshSurface()

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
        self.options.declare("aero_solver", recordable=False)

        self.extra_funcs = None

    def setup(self):

        self.solver = self.options["aero_solver"]

        self.solution_counter = 0

        self.add_input("dafoam_vol_coords", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_input("dafoam_states", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])

    def mphys_set_dvs_and_cons(self):

        DVNames, DVSizes = self.solver.getDVsCons()

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
        # TODO: setStates are not implemented yet!
        self.solver.setStates(inputs["dafoam_states"])

    def compute(self, inputs, outputs):
        solver = self.solver

        funcs = {}

        if self.extra_funcs is not None:
            solver.evalFunctions(funcs, evalFuncs=self.extra_funcs)
            for f_name in self.extra_funcs:
                if f_name in funcs:
                    outputs[f_name] = funcs[f_name]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        solver = self.solver
        DVs = self.solver.DVGeo.getValues()

        if mode == "fwd":
            xDvDot = {}
            for key in DVs:
                if key in d_inputs:
                    mach_name = key.split("_")[0]
                    xDvDot[mach_name] = d_inputs[key]

            if "dafoam_states" in d_inputs:
                wDot = d_inputs["dafoam_states"]
            else:
                wDot = None

            if "dafoam_vol_coords" in d_inputs:
                xVDot = d_inputs["dafoam_vol_coords"]
            else:
                xVDot = None

            # TODO: computeJacobianVectorProductFwd is not implemented yet!
            funcsdot = solver.computeJacobianVectorProductFwd(xDvDot=xDvDot, xVDot=xVDot, wDot=wDot, funcDeriv=True)

            for func_name in funcsdot:
                if func_name in d_outputs:
                    d_outputs[func_name] += funcsdot[func_name]

        elif mode == "rev":
            funcsBar = {}

            # also do the same for prop functions
            if self.extra_funcs is not None:
                for func_name in self.extra_funcs:
                    if func_name in d_outputs and d_outputs[func_name] != 0.0:
                        funcsBar[func_name] = d_outputs[func_name][0]

            # print(funcsBar, flush=True)

            wBar = None
            xVBar = None
            xDVBar = None

            # TODO: computeJacobianVectorProductBwd is not implemented yet!!
            wBar, xVBar, xDVBar = solver.computeJacobianVectorProductBwd(
                funcsBar=funcsBar, wDeriv=True, xVDeriv=True, xDvDeriv=False, xDvDerivAero=True
            )
            if "dafoam_states" in d_inputs:
                d_inputs["dafoam_states"] += wBar
            if "dafoam_vol_coords" in d_inputs:
                d_inputs["dafoam_vol_coords"] += xVBar

            for dv_name, dv_bar in xDVBar.items():
                if dv_name in d_inputs:
                    d_inputs[dv_name] += dv_bar.flatten()


class DAFoamWarper(ExplicitComponent):
    """
    OpenMDAO component that wraps the warping.

    """

    def initialize(self):
        self.options.declare("aero_solver", recordable=False)

    def setup(self):

        self.solver = self.options["aero_solver"]
        solver = self.solver

        # self.ap_vars,_ = get_dvs_and_cons(ap=ap)

        # state inputs and outputs
        local_volume_coord_size = solver.mesh.getSolverGrid().size

        self.add_input("x_aero", distributed=True, shape_by_conn=True, tags=["mphys_coupling"])
        self.add_output("dafoam_vol_coords", distributed=True, shape=local_volume_coord_size, tags=["mphys_coupling"])

        # self.declare_partials(of='adflow_vol_coords', wrt='x_aero')

    def compute(self, inputs, outputs):

        solver = self.solver

        x_a = inputs["x_aero"].reshape((-1, 3))
        solver.setSurfaceCoordinates(x_a)
        outputs["dafoam_vol_coords"] = solver.mesh.getSolverGrid()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        if mode == "fwd":
            if "dafoam_vol_coords" in d_outputs:
                if "x_aero" in d_inputs:
                    dxS = d_inputs["x_aero"]
                    dxV = self.solver.mesh.warpDerivFwd(dxS)
                    d_outputs["dafoam_vol_coords"] += dxV

        elif mode == "rev":
            if "dafoam_vol_coords" in d_outputs:
                if "x_aero" in d_inputs:
                    dxV = d_outputs["dafoam_vol_coords"]
                    self.solver.mesh.warpDeriv(dxV)
                    dxS = self.solver.mesh.getdXs()
                    dxS = self.solver.mapVector(
                        dxS, self.solver.meshFamilyGroup, self.solver.designFamilyGroup
                    )
                    d_inputs["x_aero"] += dxS.flatten()
