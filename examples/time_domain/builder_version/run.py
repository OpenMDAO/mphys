import numpy as np
import openmdao.api as om

from aero_solver import FakeAeroBuilder
from xfer_scheme import ModalXferBuilder
from struct_solver import ModalStructBuilder
from mphys.time_domain.integator_aerostructural import IntegratorAerostructural


class Model(om.Group):
    def setup(self):
        nsteps = 10
        dt = 1.0
        nmodes = 2
        root_name = "sphere_body1"

        aero_builder = FakeAeroBuilder(root_name, nmodes, dt)
        aero_builder.initialize(self.comm)

        struct_builder = ModalStructBuilder(nmodes, dt)
        struct_builder.initialize(self.comm)

        ldxfer_builder = ModalXferBuilder(root_name, nmodes)
        ldxfer_builder.initialize(self.comm)

        dvs = om.IndepVarComp()
        dvs.add_output("amplitude_aero", 1.0)
        dvs.add_output("freq_aero", 0.1)
        dvs.add_output("m", [1.0, 1.0])
        dvs.add_output("c", [0.0, 0.0])
        dvs.add_output("k", [1.0, 1.5])
        self.add_subsystem("design_variables", dvs, promotes=["*"])

        inputs = om.IndepVarComp()
        inputs.add_output("u_struct|0", [1.0, 2.0], distributed=True)
        inputs.add_output("x_aero0", np.ones(aero_builder.nnodes * 3), distributed=True)
        self.add_subsystem("analysis_inputs", inputs, promotes=["*"])

        self.add_subsystem(
            "integrator",
            IntegratorAerostructural(
                nsteps=nsteps,
                dt=dt,
                aero_builder=aero_builder,
                struct_builder=struct_builder,
                ldxfer_builder=ldxfer_builder,
                nonlinear_solver = om.NonlinearRunOnce(),
                linear_solver = om.LinearRunOnce()
            ),
            promotes=["*"],
        )


def main():
    prob = om.Problem()
    prob.model = Model()
    prob.setup()
    prob.run_model()


if __name__ == "__main__":
    main()
