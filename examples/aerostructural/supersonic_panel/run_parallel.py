import numpy as np
import openmdao.api as om
from aerodynamics_mphys import AeroBuilder
from geometry_morph import GeometryBuilder
from mpi4py import MPI
from structures_mphys import StructBuilder
from xfer_mphys import XferBuilder

from mphys import Multipoint, MultipointParallel
from mphys.scenarios.aerostructural import ScenarioAeroStructural

comm = MPI.COMM_WORLD
rank = comm.rank

# panel geometry
panel_chord = 0.3
panel_width = 0.01

# panel discretization
N_el_struct = 20
N_el_aero = 7

# Mphys parallel multipoint scenarios
class AerostructParallel(MultipointParallel):
    # class AerostructParallel(Multipoint):
    def __init__(
        self,
        aero_builder=None,
        struct_builder=None,
        xfer_builder=None,
        geometry_builder=None,
        scenario_names=None,
    ):
        super().__init__()
        self.aero_builder = aero_builder
        self.struct_builder = struct_builder
        self.xfer_builder = xfer_builder
        self.geometry_builder = geometry_builder
        self.scenario_names = scenario_names

    def setup(self):
        for i in range(len(self.scenario_names)):
            nonlinear_solver = om.NonlinearBlockGS(
                maxiter=100, iprint=2, use_aitken=True, aitken_initial_factor=0.5
            )
            linear_solver = om.LinearBlockGS(
                maxiter=40, iprint=2, use_aitken=True, aitken_initial_factor=0.5
            )
            self.mphys_add_scenario(
                self.scenario_names[i],
                ScenarioAeroStructural(
                    aero_builder=self.aero_builder,
                    struct_builder=self.struct_builder,
                    ldxfer_builder=self.xfer_builder,
                    geometry_builder=self.geometry_builder,
                    in_MultipointParallel=True,
                ),
                coupling_nonlinear_solver=nonlinear_solver,
                coupling_linear_solver=linear_solver,
            )


# OM group
class Model(om.Group):
    def setup(self):

        # ivc
        self.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
        self.ivc.add_output("modulus", val=70e9)
        self.ivc.add_output("yield_stress", val=270e6)
        self.ivc.add_output("density", val=2800.0)
        self.ivc.add_output("mach", val=[5.0, 3.0])
        self.ivc.add_output("qdyn", val=[3e4, 1e4])
        self.ivc.add_output("aoa", val=[3.0, 2.0], units="deg")
        self.ivc.add_output("geometry_morph_param", val=1.0)

        # create dv_struct, which is the thickness of each structural element
        thickness = 0.001 * np.ones(N_el_struct)
        self.ivc.add_output("dv_struct", thickness)

        # structure setup and builder
        structure_setup = {
            "panel_chord": panel_chord,
            "panel_width": panel_width,
            "N_el": N_el_struct,
        }
        struct_builder = StructBuilder(structure_setup)

        # aero builder
        aero_setup = {
            "panel_chord": panel_chord,
            "panel_width": panel_width,
            "N_el": N_el_aero,
        }
        aero_builder = AeroBuilder(aero_setup)

        # xfer builder
        xfer_builder = XferBuilder(
            aero_builder=aero_builder, struct_builder=struct_builder
        )

        # geometry
        builders = {"struct": struct_builder, "aero": aero_builder}
        geometry_builder = GeometryBuilder(builders)

        # list of scenario names
        scenario_names = ["aerostructural1", "aerostructural2"]

        # add parallel multipoint group
        self.add_subsystem(
            "multipoint",
            AerostructParallel(
                aero_builder=aero_builder,
                struct_builder=struct_builder,
                xfer_builder=xfer_builder,
                geometry_builder=geometry_builder,
                scenario_names=scenario_names,
            ),
        )

        for i in range(len(scenario_names)):

            # connect scalar inputs to the scenario
            for var in ["modulus", "yield_stress", "density", "dv_struct"]:
                self.connect(var, "multipoint." + scenario_names[i] + "." + var)

            # connect vector inputs
            for var in ["mach", "qdyn", "aoa"]:
                self.connect(
                    var, "multipoint." + scenario_names[i] + "." + var, src_indices=[i]
                )

            # connect top-level geom parameter
            self.connect(
                "geometry_morph_param",
                "multipoint." + scenario_names[i] + ".geometry.geometry_morph_param",
            )


# run model and check derivatives
if __name__ == "__main__":

    prob = om.Problem()
    prob.model = Model()
    prob.setup(mode="rev")

    om.n2(prob, show_browser=False, outfile="n2.html")

    prob.run_model()

    mass1 = prob.get_val("multipoint.aerostructural1.mass", get_remote=True)
    func_struct1 = prob.get_val(
        "multipoint.aerostructural1.func_struct", get_remote=True
    )
    C_L1 = prob.get_val("multipoint.aerostructural1.C_L", get_remote=True)
    mass2 = prob.get_val("multipoint.aerostructural2.mass", get_remote=True)
    func_struct2 = prob.get_val(
        "multipoint.aerostructural2.func_struct", get_remote=True
    )
    C_L2 = prob.get_val("multipoint.aerostructural2.C_L", get_remote=True)

    if rank == 0:
        print("mass1 =        " + str(mass1))
        print("func_struct1 = " + str(func_struct1))
        print("C_L1 =         " + str(C_L1))
        print("mass2 =        " + str(mass2))
        print("func_struct2 = " + str(func_struct2))
        print("C_L2 =         " + str(C_L2))

    prob.check_totals(
        of=[
            "multipoint.aerostructural1.mass",
            "multipoint.aerostructural1.func_struct",
            "multipoint.aerostructural1.C_L",
            "multipoint.aerostructural2.mass",
            "multipoint.aerostructural2.func_struct",
            "multipoint.aerostructural2.C_L",
        ],
        wrt=[
            "modulus",
            "yield_stress",
            "density",
            "mach",
            "qdyn",
            "aoa",
            "dv_struct",
            "geometry_morph_param",
        ],
        step_calc="rel_avg",
        compact_print=True,
    )

    prob.check_partials(compact_print=True, step_calc="rel_avg")