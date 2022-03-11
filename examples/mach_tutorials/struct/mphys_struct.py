# rst Imports
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI

import openmdao.api as om

from mphys import Multipoint
from mphys.scenario_structural import ScenarioStructural

# these imports will be from the respective codes' repos rather than omfsi
from tacs.mphys import TacsBuilder
from tacs import elements, constitutive, functions

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank


class Top(om.Group):
    def setup(self):

        ################################################################################
        # TACS options
        ################################################################################

        # Material properties
        rho = 2500.0  # density kg/m^3
        E = 70.0e9  # Young's modulus (Pa)
        nu = 0.30  # Poisson's ratio
        kcorr = 5.0 / 6.0  # shear correction factor
        ys = 350e6  # yield stress

        # Shell thickness
        t = 0.01  # m
        tMin = 0.002  # m
        tMax = 0.05  # m

        # Callback function used to setup TACS element objects and DVs
        def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
            # Setup (isotropic) property and constitutive objects
            prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
            # Set one thickness dv for every component
            con = constitutive.IsoShellConstitutive(prop, t=t, tNum=dvNum)

            # Define reference axis for local shell stresses
            if "SKIN" in compDescript:  # USKIN + LSKIN
                sweep = 35.0 / 180.0 * np.pi
                refAxis = np.array([np.sin(sweep), np.cos(sweep), 0])
            else:  # RIBS + SPARS + ENGINE_MOUNT
                refAxis = np.array([0.0, 0.0, 1.0])

            # For each element type in this component,
            # pass back the appropriate tacs element object
            transform = elements.ShellRefAxisTransform(refAxis)
            elem = elements.Quad4Shell(transform, con)
            return elem

        def problem_setup(scenario_name, fea_assembler, problem):
            """
            Helper function to add fixed forces and eval functions
            to structural problems used in tacs builder
            """

            # Add TACS Functions
            problem.addFunction("mass", functions.StructuralMass)
            problem.addFunction("ks_vmfailure", functions.KSFailure, safetyFactor=1.0, ksWeight=100.0)

            # Add forces to static problem
            F = fea_assembler.createVec()
            ndof = fea_assembler.getVarsPerNode()
            F[2::ndof] = 100.0
            problem.addLoadToRHS(F)

        tacs_options = {
            "element_callback": element_callback,
            "problem_setup": problem_setup,
            "mesh_file": "wingbox.bdf",
        }

        tacs_builder = TacsBuilder(tacs_options, coupled=False)
        tacs_builder.initialize(self.comm)

        ################################################################################
        # MPHY setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # create the multiphysics multipoint group.
        mp = self.add_subsystem("mp_group", Multipoint())

        # add the structural thickness DVs
        ndv_struct = tacs_builder.get_ndv()
        self.dvs.add_output("dv_struct", np.array(ndv_struct * [0.01]))

        mp.add_subsystem("mesh", tacs_builder.get_mesh_coordinate_subsystem())
        # this is the method that needs to be called for every point in this mp_group
        mp.mphys_add_scenario("s0", ScenarioStructural(struct_builder=tacs_builder))
        mp.mphys_connect_scenario_coordinate_source("mesh", "s0", "struct")

        self.connect("dv_struct", "mp_group.s0.dv_struct")


################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model
prob.setup()
om.n2(prob, show_browser=False, outfile="mphys_struct.html")
prob.run_model()
# prob.model.list_outputs()
if MPI.COMM_WORLD.rank == 0:
    print("Scenario 0")
