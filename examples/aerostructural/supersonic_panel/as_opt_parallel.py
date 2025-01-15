import os

import numpy as np
import openmdao.api as om
from aerodynamics_mphys import AeroBuilder
from geometry_morph import GeometryBuilder
from openmdao.core.constants import _DEFAULT_OUT_STREAM
from structures_mphys import StructBuilder
from xfer_mphys import XferBuilder

from mphys import Multipoint, MultipointParallel
from mphys.scenarios.aerostructural import ScenarioAeroStructural

check_totals = (
    False  # True=check objective/constraint derivatives, False=run optimization
)

# panel geometry
panel_chord = 0.3
panel_width = 0.01

# panel discretization
N_el_struct = 20
N_el_aero = 7

# Mphys parallel multipoint scenarios
class AerostructParallel(MultipointParallel):
    def initialize(self):
        self.options.declare("aero_builder")
        self.options.declare("struct_builder")
        self.options.declare("xfer_builder")
        self.options.declare("geometry_builder")
        self.options.declare("scenario_names")

    def setup(self):
        for i in range(len(self.options["scenario_names"])):

            # create the run directory
            if self.comm.rank == 0:
                if not os.path.isdir(self.options["scenario_names"][i]):
                    os.mkdir(self.options["scenario_names"][i])
            self.comm.Barrier()

            nonlinear_solver = om.NonlinearBlockGS(
                maxiter=100, iprint=2, use_aitken=True, aitken_initial_factor=0.5
            )
            linear_solver = om.LinearBlockGS(
                maxiter=40, iprint=2, use_aitken=True, aitken_initial_factor=0.5
            )
            self.mphys_add_scenario(
                self.options["scenario_names"][i],
                ScenarioAeroStructural(
                    aero_builder=self.options["aero_builder"],
                    struct_builder=self.options["struct_builder"],
                    ldxfer_builder=self.options["xfer_builder"],
                    geometry_builder=self.options["geometry_builder"],
                    in_MultipointParallel=True,
                    run_directory=self.options["scenario_names"][i],
                ),
                coupling_nonlinear_solver=nonlinear_solver,
                coupling_linear_solver=linear_solver,
            )


# OM group
class Model(om.Group):
    def initialize(self):
        self.options.declare(
            "scenario_names", default=["aerostructural1", "aerostructural2"]
        )

    def setup(self):
        self.scenario_names = self.options["scenario_names"]

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

        # add parallel multipoint group
        self.add_subsystem(
            "multipoint",
            AerostructParallel(
                aero_builder=aero_builder,
                struct_builder=struct_builder,
                xfer_builder=xfer_builder,
                geometry_builder=geometry_builder,
                scenario_names=self.scenario_names,
            ),
        )

        for i in range(len(self.scenario_names)):

            # connect scalar inputs to the scenario
            for var in ["modulus", "yield_stress", "density", "dv_struct"]:
                self.connect(var, "multipoint." + self.scenario_names[i] + "." + var)

            # connect vector inputs
            for var in ["mach", "qdyn", "aoa"]:
                self.connect(
                    var,
                    "multipoint." + self.scenario_names[i] + "." + var,
                    src_indices=[i],
                )

            # connect top-level geom parameter
            self.connect(
                "geometry_morph_param",
                "multipoint."
                + self.scenario_names[i]
                + ".geometry.geometry_morph_param",
            )

        # add design vars
        self.add_design_var("geometry_morph_param", lower=0.1, upper=10.0)
        self.add_design_var("dv_struct", lower=1.0e-4, upper=1.0e-2, ref=1.0e-3)
        self.add_design_var("aoa", lower=-20.0, upper=20.0)

        # add objective/constraints
        self.add_objective(f"multipoint.{self.scenario_names[0]}.mass", ref=0.01)
        self.add_constraint(
            f"multipoint.{self.scenario_names[0]}.func_struct",
            upper=1.0,
            parallel_deriv_color="struct_cons",
        )  # run func_struct derivatives in parallel
        self.add_constraint(
            f"multipoint.{self.scenario_names[1]}.func_struct",
            upper=1.0,
            parallel_deriv_color="struct_cons",
        )
        self.add_constraint(
            f"multipoint.{self.scenario_names[0]}.C_L",
            lower=0.15,
            ref=0.1,
            parallel_deriv_color="lift_cons",
        )  # run C_L derivatives in parallel
        self.add_constraint(
            f"multipoint.{self.scenario_names[1]}.C_L",
            lower=0.45,
            ref=0.1,
            parallel_deriv_color="lift_cons",
        )


def get_model(scenario_names):
    return Model(scenario_names=scenario_names)


# run model and check derivatives
if __name__ == "__main__":

    prob = om.Problem()
    prob.model = Model()

    if check_totals:
        prob.setup(mode="rev")
        om.n2(prob, show_browser=False, outfile="n2.html")
        prob.run_model()
        prob.check_totals(
            step_calc="rel_avg",
            compact_print=True,
            directional=False,
            show_progress=True,
            out_stream=None if prob.model.comm.rank > 0 else _DEFAULT_OUT_STREAM,
        )

    else:

        # setup optimization driver
        prob.driver = om.ScipyOptimizeDriver(
            debug_print=["nl_cons", "objs", "desvars", "totals"]
        )
        prob.driver.options["optimizer"] = "SLSQP"
        prob.driver.options["tol"] = 1e-5
        prob.driver.options["disp"] = True
        prob.driver.options["maxiter"] = 300

        # add optimization recorder
        prob.driver.recording_options["record_objectives"] = True
        prob.driver.recording_options["record_constraints"] = True
        prob.driver.recording_options["record_desvars"] = True
        prob.driver.recording_options["record_derivatives"] = True

        recorder = om.SqliteRecorder("optimization_history.sql")
        prob.driver.add_recorder(recorder)

        # run the optimization
        prob.setup(mode="rev")
        prob.run_driver()
        prob.cleanup()

        if prob.model.comm.rank == 0:  # write out data
            cr = om.CaseReader("optimization_history.sql")
            driver_cases = cr.list_cases("driver")

            case = cr.get_case(0)
            cons = case.get_constraints()
            dvs = case.get_design_vars()
            objs = case.get_objectives()

            f = open("optimization_history.dat", "w+")

            for i, k in enumerate(objs.keys()):
                f.write("objective: " + k + "\n")
                for j, case_id in enumerate(driver_cases):
                    f.write(
                        str(j)
                        + " "
                        + str(cr.get_case(case_id).get_objectives(scaled=False)[k][0])
                        + "\n"
                    )
                f.write(" " + "\n")

            for i, k in enumerate(cons.keys()):
                f.write("constraint: " + k + "\n")
                for j, case_id in enumerate(driver_cases):
                    f.write(
                        str(j)
                        + " "
                        + " ".join(
                            map(
                                str,
                                cr.get_case(case_id).get_constraints(scaled=False)[k],
                            )
                        )
                        + "\n"
                    )
                f.write(" " + "\n")

            for i, k in enumerate(dvs.keys()):
                f.write("DV: " + k + "\n")
                for j, case_id in enumerate(driver_cases):
                    f.write(
                        str(j)
                        + " "
                        + " ".join(
                            map(
                                str,
                                cr.get_case(case_id).get_design_vars(scaled=False)[k],
                            )
                        )
                        + "\n"
                    )
                f.write(" " + "\n")

            f.close()
