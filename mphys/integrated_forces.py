import numpy as np
import openmdao.api as om

from mphys import MPhysVariables

AOA_NAME = MPhysVariables.Aerodynamics.FlowConditions.ANGLE_OF_ATTACK
YAW_NAME = MPhysVariables.Aerodynamics.FlowConditions.YAW_ANGLE
QINF_NAME = MPhysVariables.Aerodynamics.FlowConditions.DYNAMIC_PRESSURE

REF_AREA_NAME = MPhysVariables.Aerodynamics.ReferenceGeometry.REF_AREA
REF_LENGTH_X_NAME = MPhysVariables.Aerodynamics.ReferenceGeometry.REF_LENGTH_X
REF_LENGTH_Y_NAME = MPhysVariables.Aerodynamics.ReferenceGeometry.REF_LENGTH_Y
MOMENT_CENTER_NAME = MPhysVariables.Aerodynamics.ReferenceGeometry.MOMENT_CENTER

X_AERO_NAME = MPhysVariables.Aerodynamics.Surface.COORDINATES
F_AERO_NAME = MPhysVariables.Aerodynamics.Surface.LOADS


class IntegratedSurfaceForces(om.ExplicitComponent):
    def setup(self):
        self.add_input(
            AOA_NAME, desc="angle of attack", units="rad", tags=["mphys_input"]
        )
        self.add_input(YAW_NAME, desc="yaw angle", units="rad", tags=["mphys_input"])
        self.add_input(REF_AREA_NAME, val=1.0, tags=["mphys_input"])
        self.add_input(MOMENT_CENTER_NAME, shape=3, tags=["mphys_input"])
        self.add_input(REF_LENGTH_X_NAME, val=1.0, tags=["mphys_input"])
        self.add_input(REF_LENGTH_Y_NAME, val=1.0, tags=["mphys_input"])
        self.add_input(QINF_NAME, val=1.0, tags=["mphys_input"])

        self.add_input(
            X_AERO_NAME,
            shape_by_conn=True,
            distributed=True,
            desc="surface coordinates",
            tags=["mphys_coupling"],
        )
        self.add_input(
            F_AERO_NAME,
            shape_by_conn=True,
            distributed=True,
            desc="dimensional forces at nodes",
            tags=["mphys_coupling"],
        )

        self.add_output("C_L", desc="Lift coefficient", tags=["mphys_result"])
        self.add_output("C_D", desc="Drag coefficient", tags=["mphys_result"])
        self.add_output("C_X", desc="X Force coefficient", tags=["mphys_result"])
        self.add_output("C_Y", desc="Y Force coefficient", tags=["mphys_result"])
        self.add_output("C_Z", desc="Z Force coefficient", tags=["mphys_result"])
        self.add_output("CM_X", desc="X Moment coefficient", tags=["mphys_result"])
        self.add_output("CM_Y", desc="Y Moment coefficient", tags=["mphys_result"])
        self.add_output("CM_Z", desc="Z Moment coefficient", tags=["mphys_result"])

        self.add_output("Lift", desc="Total Lift", tags=["mphys_result"])
        self.add_output("Drag", desc="Total Drag", tags=["mphys_result"])
        self.add_output("F_X", desc="Total X Force", tags=["mphys_result"])
        self.add_output("F_Y", desc="Total Y Force", tags=["mphys_result"])
        self.add_output("F_Z", desc="Total Z Force", tags=["mphys_result"])
        self.add_output("M_X", desc="Total X Moment", tags=["mphys_result"])
        self.add_output("M_Y", desc="Total Y Moment", tags=["mphys_result"])
        self.add_output("M_Z", desc="Total Z Moment", tags=["mphys_result"])

    def compute(self, inputs, outputs):
        aoa = inputs[AOA_NAME]
        yaw = inputs[YAW_NAME]
        area = inputs[REF_AREA_NAME]
        q_inf = inputs[QINF_NAME]
        xc = inputs[MOMENT_CENTER_NAME][0]
        yc = inputs[MOMENT_CENTER_NAME][1]
        zc = inputs[MOMENT_CENTER_NAME][2]
        c = inputs[REF_LENGTH_X_NAME]
        span = inputs[REF_LENGTH_Y_NAME]

        x = inputs[X_AERO_NAME][0::3]
        y = inputs[X_AERO_NAME][1::3]
        z = inputs[X_AERO_NAME][2::3]

        fx = inputs[F_AERO_NAME][0::3]
        fy = inputs[F_AERO_NAME][1::3]
        fz = inputs[F_AERO_NAME][2::3]

        fx_total = self.comm.allreduce(np.sum(fx))
        fy_total = self.comm.allreduce(np.sum(fy))
        fz_total = self.comm.allreduce(np.sum(fz))

        outputs["F_X"] = fx_total
        outputs["F_Y"] = fy_total
        outputs["F_Z"] = fz_total
        outputs["C_X"] = fx_total / (q_inf * area)
        outputs["C_Y"] = fy_total / (q_inf * area)
        outputs["C_Z"] = fz_total / (q_inf * area)

        outputs["Lift"] = -fx_total * np.sin(aoa) + fz_total * np.cos(aoa)
        outputs["Drag"] = (
            fx_total * np.cos(aoa) * np.cos(yaw)
            - fy_total * np.sin(yaw)
            + fz_total * np.sin(aoa) * np.cos(yaw)
        )

        outputs["C_L"] = outputs["Lift"] / (q_inf * area)
        outputs["C_D"] = outputs["Drag"] / (q_inf * area)

        m_x = self.comm.allreduce(np.dot(fz, (y - yc)) - np.dot(fy, (z - zc)))
        m_y = self.comm.allreduce(-np.dot(fz, (x - xc)) + np.dot(fx, (z - zc)))
        m_z = self.comm.allreduce(np.dot(fy, (x - xc)) - np.dot(fx, (y - yc)))

        outputs["M_X"] = m_x
        outputs["M_Y"] = m_y
        outputs["M_Z"] = m_z

        outputs["CM_X"] = m_x / (q_inf * area * span)
        outputs["CM_Y"] = m_y / (q_inf * area * c)
        outputs["CM_Z"] = m_z / (q_inf * area * span)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        aoa = inputs[AOA_NAME]
        yaw = inputs[YAW_NAME]
        area = inputs[REF_AREA_NAME]
        q_inf = inputs[QINF_NAME]
        xc = inputs[MOMENT_CENTER_NAME][0]
        yc = inputs[MOMENT_CENTER_NAME][1]
        zc = inputs[MOMENT_CENTER_NAME][2]
        c = inputs[REF_LENGTH_X_NAME]
        span = inputs[REF_LENGTH_Y_NAME]

        x = inputs[X_AERO_NAME][0::3]
        y = inputs[X_AERO_NAME][1::3]
        z = inputs[X_AERO_NAME][2::3]

        fx = inputs[F_AERO_NAME][0::3]
        fy = inputs[F_AERO_NAME][1::3]
        fz = inputs[F_AERO_NAME][2::3]

        fx_total = self.comm.allreduce(np.sum(fx))
        fy_total = self.comm.allreduce(np.sum(fy))
        fz_total = self.comm.allreduce(np.sum(fz))

        lift = -fx_total * np.sin(aoa) + fz_total * np.cos(aoa)
        drag = (
            fx_total * np.cos(aoa) * np.cos(yaw)
            - fy_total * np.sin(yaw)
            + fz_total * np.sin(aoa) * np.cos(yaw)
        )

        m_x = self.comm.allreduce(np.dot(fz, (y - yc)) - np.dot(fy, (z - zc)))
        m_y = self.comm.allreduce(-np.dot(fz, (x - xc)) + np.dot(fx, (z - zc)))
        m_z = self.comm.allreduce(np.dot(fy, (x - xc)) - np.dot(fx, (y - yc)))

        if mode == "fwd":
            if AOA_NAME in d_inputs:
                daoa_rad = d_inputs[AOA_NAME]
                if "Lift" in d_outputs or "C_L" in d_outputs:
                    d_lift_d_aoa = (
                        -fx_total * np.cos(aoa) * daoa_rad
                        - fz_total * np.sin(aoa) * daoa_rad
                    )

                    if "Lift" in d_outputs:
                        d_outputs["Lift"] += d_lift_d_aoa
                    if "C_L" in d_outputs:
                        d_outputs["C_L"] += d_lift_d_aoa / (q_inf * area)
                if "Drag" in d_outputs or "C_D" in d_outputs:
                    d_drag_d_aoa = fx_total * (-np.sin(aoa) * daoa_rad) * np.cos(
                        yaw
                    ) + fz_total * (np.cos(aoa) * daoa_rad) * np.cos(yaw)
                    if "Drag" in d_outputs:
                        d_outputs["Drag"] += d_drag_d_aoa
                    if "C_D" in d_outputs:
                        d_outputs["C_D"] += d_drag_d_aoa / (q_inf * area)

            if YAW_NAME in d_inputs:
                dyaw_rad = d_inputs[YAW_NAME]
                if "Drag" in d_outputs or "C_D" in d_outputs:
                    d_drag_d_yaw = (
                        fx_total * np.cos(aoa) * (-np.sin(yaw) * dyaw_rad)
                        - fy_total * np.cos(yaw) * dyaw_rad
                        + fz_total * np.sin(aoa) * (-np.sin(yaw) * dyaw_rad)
                    )
                    if "Drag" in d_outputs:
                        d_outputs["Drag"] += d_drag_d_yaw
                    if "C_D" in d_outputs:
                        d_outputs["C_D"] += d_drag_d_yaw / (q_inf * area)

            if REF_AREA_NAME in d_inputs:
                d_nondim = -d_inputs[REF_AREA_NAME] / (q_inf * area**2.0)
                if "C_X" in d_outputs:
                    d_outputs["C_X"] += fx_total * d_nondim
                if "C_Y" in d_outputs:
                    d_outputs["C_Y"] += fy_total * d_nondim
                if "C_Z" in d_outputs:
                    d_outputs["C_Z"] += fz_total * d_nondim
                if "C_L" in d_outputs:
                    d_outputs["C_L"] += lift * d_nondim
                if "C_D" in d_outputs:
                    d_outputs["C_D"] += drag * d_nondim
                if "CM_X" in d_outputs:
                    d_outputs["CM_X"] += m_x * d_nondim / span
                if "CM_X" in d_outputs:
                    d_outputs["CM_Y"] += m_y * d_nondim / c
                if "CM_Z" in d_outputs:
                    d_outputs["CM_Z"] += m_z * d_nondim / span

            if MOMENT_CENTER_NAME in d_inputs:
                dxc = d_inputs[MOMENT_CENTER_NAME][0]
                dyc = d_inputs[MOMENT_CENTER_NAME][1]
                dzc = d_inputs[MOMENT_CENTER_NAME][2]
                if "M_X" in d_outputs:
                    d_outputs["M_X"] += -fz_total * dyc + fy_total * dzc
                if "M_Y" in d_outputs:
                    d_outputs["M_Y"] += fz_total * dxc - fx_total * dzc
                if "M_Z" in d_outputs:
                    d_outputs["M_Z"] += -fy_total * dxc + fx_total * dyc
                if "CM_X" in d_outputs:
                    d_outputs["CM_X"] += (-fz_total * dyc + fy_total * dzc) / (
                        q_inf * area * span
                    )
                if "CM_Y" in d_outputs:
                    d_outputs["CM_Y"] += (fz_total * dxc - fx_total * dzc) / (
                        q_inf * area * c
                    )
                if "CM_Z" in d_outputs:
                    d_outputs["CM_Z"] += (-fy_total * dxc + fx_total * dyc) / (
                        q_inf * area * span
                    )

            if REF_LENGTH_X_NAME in d_inputs:
                d_nondim = -d_inputs[REF_LENGTH_X_NAME] / (q_inf * area * c**2.0)
                if "CM_Y" in d_outputs:
                    d_outputs["CM_Y"] += m_y * d_nondim

            if REF_LENGTH_Y_NAME in d_inputs:
                d_nondim = -d_inputs[REF_LENGTH_Y_NAME] / (q_inf * area * span**2.0)
                if "CM_X" in d_outputs:
                    d_outputs["CM_X"] += m_x * d_nondim
                if "CM_Z" in d_outputs:
                    d_outputs["CM_Z"] += m_z * d_nondim

            if QINF_NAME in d_inputs:
                d_nondim = -d_inputs[QINF_NAME] / (q_inf**2.0 * area)
                if "C_X" in d_outputs:
                    d_outputs["C_X"] += fx_total * d_nondim
                if "C_Y" in d_outputs:
                    d_outputs["C_Y"] += fy_total * d_nondim
                if "C_Z" in d_outputs:
                    d_outputs["C_Z"] += fz_total * d_nondim
                if "C_L" in d_outputs:
                    d_outputs["C_L"] += lift * d_nondim
                if "C_D" in d_outputs:
                    d_outputs["C_D"] += drag * d_nondim
                if "CM_X" in d_outputs:
                    d_outputs["CM_X"] += m_x * d_nondim / span
                if "CM_X" in d_outputs:
                    d_outputs["CM_Y"] += m_y * d_nondim / c
                if "CM_Z" in d_outputs:
                    d_outputs["CM_Z"] += m_z * d_nondim / span

            if X_AERO_NAME in d_inputs:
                dx = d_inputs[X_AERO_NAME][0::3]
                dy = d_inputs[X_AERO_NAME][1::3]
                dz = d_inputs[X_AERO_NAME][2::3]
                if "M_X" in d_outputs:
                    d_outputs["M_X"] += np.dot(fz, dy) - np.dot(fy, dz)
                if "M_Y" in d_outputs:
                    d_outputs["M_Y"] += -np.dot(fz, dx) + np.dot(fx, dz)
                if "M_Z" in d_outputs:
                    d_outputs["M_Z"] += np.dot(fy, dx) - np.dot(fx, dy)
                if "CM_X" in d_outputs:
                    d_outputs["CM_X"] += (np.dot(fz, dy) - np.dot(fy, dz)) / (
                        q_inf * area * span
                    )
                if "CM_Y" in d_outputs:
                    d_outputs["CM_Y"] += (-np.dot(fz, dx) + np.dot(fx, dz)) / (
                        q_inf * area * c
                    )
                if "CM_Z" in d_outputs:
                    d_outputs["CM_Z"] += (np.dot(fy, dx) - np.dot(fx, dy)) / (
                        q_inf * area * span
                    )

            if F_AERO_NAME in d_inputs:
                dfx = d_inputs[F_AERO_NAME][0::3]
                dfy = d_inputs[F_AERO_NAME][1::3]
                dfz = d_inputs[F_AERO_NAME][2::3]
                dfx_total = np.sum(dfx)
                dfy_total = np.sum(dfy)
                dfz_total = np.sum(dfz)
                if "F_X" in d_outputs:
                    d_outputs["F_X"] += dfx_total
                if "F_Y" in d_outputs:
                    d_outputs["F_Y"] += dfy_total
                if "F_Z" in d_outputs:
                    d_outputs["F_Z"] += dfz_total
                if "C_X" in d_outputs:
                    d_outputs["C_X"] += dfx_total / (q_inf * area)
                if "C_Y" in d_outputs:
                    d_outputs["C_Y"] += dfy_total / (q_inf * area)
                if "C_Z" in d_outputs:
                    d_outputs["C_Z"] += dfz_total / (q_inf * area)
                if "Lift" in d_outputs:
                    d_outputs["Lift"] += -dfx_total * np.sin(aoa) + dfz_total * np.cos(
                        aoa
                    )
                if "Drag" in d_outputs:
                    d_outputs["Drag"] += (
                        dfx_total * np.cos(aoa) * np.cos(yaw)
                        - dfy_total * np.sin(yaw)
                        + dfz_total * np.sin(aoa) * np.cos(yaw)
                    )
                if "C_L" in d_outputs:
                    d_outputs["C_L"] += (
                        -dfx_total * np.sin(aoa) + dfz_total * np.cos(aoa)
                    ) / (q_inf * area)
                if "C_D" in d_outputs:
                    d_outputs["C_D"] += (
                        dfx_total * np.cos(aoa) * np.cos(yaw)
                        - dfy_total * np.sin(yaw)
                        + dfz_total * np.sin(aoa) * np.cos(yaw)
                    ) / (q_inf * area)

                if "M_X" in d_outputs:
                    d_outputs["M_X"] += np.dot(dfz, (y - yc)) - np.dot(dfy, (z - zc))
                if "M_Y" in d_outputs:
                    d_outputs["M_Y"] += -np.dot(dfz, (x - xc)) + np.dot(dfx, (z - zc))
                if "M_Z" in d_outputs:
                    d_outputs["M_Z"] += np.dot(dfy, (x - xc)) - np.dot(dfx, (y - yc))
                if "CM_X" in d_outputs:
                    d_outputs["CM_X"] += (
                        np.dot(dfz, (y - yc)) - np.dot(dfy, (z - zc))
                    ) / (q_inf * area * span)
                if "CM_Y" in d_outputs:
                    d_outputs["CM_Y"] += (
                        -np.dot(dfz, (x - xc)) + np.dot(dfx, (z - zc))
                    ) / (q_inf * area * c)
                if "CM_Z" in d_outputs:
                    d_outputs["CM_Z"] += (
                        np.dot(dfy, (x - xc)) - np.dot(dfx, (y - yc))
                    ) / (q_inf * area * span)

        elif mode == "rev":
            if AOA_NAME in d_inputs:
                if "Lift" in d_outputs or "C_L" in d_outputs:
                    d_lift = d_outputs["Lift"] if "Lift" in d_outputs else 0.0
                    d_cl = d_outputs["C_L"] if "C_L" in d_outputs else 0.0
                    d_inputs[AOA_NAME] += (
                        -fx_total * np.cos(aoa) - fz_total * np.sin(aoa)
                    ) * (d_lift + d_cl / (q_inf * area))

                if "Drag" in d_outputs or "C_D" in d_outputs:
                    d_drag = d_outputs["Drag"] if "Drag" in d_outputs else 0.0
                    d_cd = d_outputs["C_D"] if "C_D" in d_outputs else 0.0
                    d_inputs[AOA_NAME] += (
                        fx_total * (-np.sin(aoa)) * np.cos(yaw)
                        + fz_total * (np.cos(aoa)) * np.cos(yaw)
                    ) * (d_drag + d_cd / (q_inf * area))
            if YAW_NAME in d_inputs:
                if "Drag" in d_outputs or "C_D" in d_outputs:
                    d_drag = d_outputs["Drag"] if "Drag" in d_outputs else 0.0
                    d_cd = d_outputs["C_D"] if "C_D" in d_outputs else 0.0
                    d_inputs[YAW_NAME] += (
                        fx_total * np.cos(aoa) * (-np.sin(yaw))
                        - fy_total * np.cos(yaw)
                        + fz_total * np.sin(aoa) * (-np.sin(yaw))
                    ) * (d_drag + d_cd / (q_inf * area))

            if REF_AREA_NAME in d_inputs:
                d_nondim = -1.0 / (q_inf * area**2.0)
                if "C_X" in d_outputs:
                    d_inputs[REF_AREA_NAME] += d_outputs["C_X"] * fx_total * d_nondim
                if "C_Y" in d_outputs:
                    d_inputs[REF_AREA_NAME] += d_outputs["C_Y"] * fy_total * d_nondim
                if "C_Z" in d_outputs:
                    d_inputs[REF_AREA_NAME] += d_outputs["C_Z"] * fz_total * d_nondim
                if "C_L" in d_outputs:
                    d_inputs[REF_AREA_NAME] += d_outputs["C_L"] * lift * d_nondim
                if "C_D" in d_outputs:
                    d_inputs[REF_AREA_NAME] += d_outputs["C_D"] * drag * d_nondim
                if "CM_X" in d_outputs:
                    d_inputs[REF_AREA_NAME] += d_outputs["CM_X"] * m_x * d_nondim / span
                if "CM_X" in d_outputs:
                    d_inputs[REF_AREA_NAME] += d_outputs["CM_Y"] * m_y * d_nondim / c
                if "CM_Z" in d_outputs:
                    d_inputs[REF_AREA_NAME] += d_outputs["CM_Z"] * m_z * d_nondim / span

            if MOMENT_CENTER_NAME in d_inputs:
                if "M_X" in d_outputs:
                    d_inputs[MOMENT_CENTER_NAME][1] += -fz_total * d_outputs["M_X"]
                    d_inputs[MOMENT_CENTER_NAME][2] += fy_total * d_outputs["M_X"]
                if "M_Y" in d_outputs:
                    d_inputs[MOMENT_CENTER_NAME][0] += fz_total * d_outputs["M_Y"]
                    d_inputs[MOMENT_CENTER_NAME][2] += -fx_total * d_outputs["M_Y"]
                if "M_Z" in d_outputs:
                    d_inputs[MOMENT_CENTER_NAME][0] += -fy_total * d_outputs["M_Z"]
                    d_inputs[MOMENT_CENTER_NAME][1] += fx_total * d_outputs["M_Z"]
                if "CM_X" in d_outputs:
                    d_inputs[MOMENT_CENTER_NAME][1] += (
                        -fz_total * d_outputs["CM_X"] / (q_inf * area * span)
                    )
                    d_inputs[MOMENT_CENTER_NAME][2] += (
                        fy_total * d_outputs["CM_X"] / (q_inf * area * span)
                    )
                if "CM_Y" in d_outputs:
                    d_inputs[MOMENT_CENTER_NAME][0] += (
                        fz_total * d_outputs["CM_Y"] / (q_inf * area * c)
                    )
                    d_inputs[MOMENT_CENTER_NAME][2] += (
                        -fx_total * d_outputs["CM_Y"] / (q_inf * area * c)
                    )
                if "CM_Z" in d_outputs:
                    d_inputs[MOMENT_CENTER_NAME][0] += (
                        -fy_total * d_outputs["CM_Z"] / (q_inf * area * span)
                    )
                    d_inputs[MOMENT_CENTER_NAME][1] += (
                        fx_total * d_outputs["CM_Z"] / (q_inf * area * span)
                    )
            if REF_LENGTH_X_NAME in d_inputs:
                d_nondim = -1.0 / (q_inf * area * c**2.0)
                if "CM_Y" in d_outputs:
                    d_inputs[REF_LENGTH_X_NAME] += m_y * d_nondim * d_outputs["CM_Y"]

            if REF_LENGTH_Y_NAME in d_inputs:
                d_nondim = -1.0 / (q_inf * area * span**2.0)
                if "CM_X" in d_outputs:
                    d_inputs[REF_LENGTH_Y_NAME] += m_x * d_nondim * d_outputs["CM_X"]
                if "CM_Z" in d_outputs:
                    d_inputs[REF_LENGTH_Y_NAME] += m_z * d_nondim * d_outputs["CM_Z"]

            if QINF_NAME in d_inputs:
                d_nondim = -1.0 / (q_inf**2.0 * area)
                if "C_X" in d_outputs:
                    d_inputs[QINF_NAME] += d_outputs["C_X"] * fx_total * d_nondim
                if "C_Y" in d_outputs:
                    d_inputs[QINF_NAME] += d_outputs["C_Y"] * fy_total * d_nondim
                if "C_Z" in d_outputs:
                    d_inputs[QINF_NAME] += d_outputs["C_Z"] * fz_total * d_nondim
                if "C_L" in d_outputs:
                    d_inputs[QINF_NAME] += d_outputs["C_L"] * lift * d_nondim
                if "C_D" in d_outputs:
                    d_inputs[QINF_NAME] += d_outputs["C_D"] * drag * d_nondim
                if "CM_X" in d_outputs:
                    d_inputs[QINF_NAME] += d_outputs["CM_X"] * m_x * d_nondim / span
                if "CM_X" in d_outputs:
                    d_inputs[QINF_NAME] += d_outputs["CM_Y"] * m_y * d_nondim / c
                if "CM_Z" in d_outputs:
                    d_inputs[QINF_NAME] += d_outputs["CM_Z"] * m_z * d_nondim / span

            if X_AERO_NAME in d_inputs:
                nondim_c = 1.0 / (q_inf * area * c)
                nondim_span = 1.0 / (q_inf * area * span)
                dm_x = d_outputs["M_X"] if "M_X" in d_outputs else 0.0
                dm_y = d_outputs["M_Y"] if "M_Y" in d_outputs else 0.0
                dm_z = d_outputs["M_Z"] if "M_Z" in d_outputs else 0.0
                dcm_x = d_outputs["CM_X"] * nondim_span if "CM_X" in d_outputs else 0.0
                dcm_y = d_outputs["CM_Y"] * nondim_c if "CM_Y" in d_outputs else 0.0
                dcm_z = d_outputs["CM_Z"] * nondim_span if "CM_Z" in d_outputs else 0.0
                d_inputs[X_AERO_NAME][0::3] += -fz * (dm_y + dcm_y) + fy * (
                    dm_z + dcm_z
                )
                d_inputs[X_AERO_NAME][1::3] += fz * (dm_x + dcm_x) - fx * (dm_z + dcm_z)
                d_inputs[X_AERO_NAME][2::3] += -fy * (dm_x + dcm_x) + fx * (
                    dm_y + dcm_y
                )

            if F_AERO_NAME in d_inputs:
                if "F_X" in d_outputs:
                    d_inputs[F_AERO_NAME][0::3] += d_outputs["F_X"]
                if "F_Y" in d_outputs:
                    d_inputs[F_AERO_NAME][1::3] += d_outputs["F_Y"]
                if "F_Z" in d_outputs:
                    d_inputs[F_AERO_NAME][2::3] += d_outputs["F_Z"]
                if "C_X" in d_outputs:
                    d_inputs[F_AERO_NAME][0::3] += d_outputs["C_X"] / (q_inf * area)
                if "C_Y" in d_outputs:
                    d_inputs[F_AERO_NAME][1::3] += d_outputs["C_Y"] / (q_inf * area)
                if "C_Z" in d_outputs:
                    d_inputs[F_AERO_NAME][2::3] += d_outputs["C_Z"] / (q_inf * area)
                if "Lift" in d_outputs:
                    d_inputs[F_AERO_NAME][0::3] += -np.sin(aoa) * d_outputs["Lift"]
                    d_inputs[F_AERO_NAME][2::3] += np.cos(aoa) * d_outputs["Lift"]
                if "Drag" in d_outputs:
                    d_inputs[F_AERO_NAME][0::3] += (
                        np.cos(aoa) * np.cos(yaw) * d_outputs["Drag"]
                    )
                    d_inputs[F_AERO_NAME][1::3] += -np.sin(yaw) * d_outputs["Drag"]
                    d_inputs[F_AERO_NAME][2::3] += (
                        np.sin(aoa) * np.cos(yaw) * d_outputs["Drag"]
                    )
                if "C_L" in d_outputs:
                    d_inputs[F_AERO_NAME][0::3] += (
                        -np.sin(aoa) * d_outputs["C_L"] / (q_inf * area)
                    )
                    d_inputs[F_AERO_NAME][2::3] += (
                        np.cos(aoa) * d_outputs["C_L"] / (q_inf * area)
                    )
                if "C_D" in d_outputs:
                    d_inputs[F_AERO_NAME][0::3] += (
                        np.cos(aoa) * np.cos(yaw) * d_outputs["C_D"] / (q_inf * area)
                    )
                    d_inputs[F_AERO_NAME][1::3] += (
                        -np.sin(yaw) * d_outputs["C_D"] / (q_inf * area)
                    )
                    d_inputs[F_AERO_NAME][2::3] += (
                        np.sin(aoa) * np.cos(yaw) * d_outputs["C_D"] / (q_inf * area)
                    )

                if "M_X" in d_outputs:
                    d_inputs[F_AERO_NAME][1::3] += -(z - zc) * d_outputs["M_X"]
                    d_inputs[F_AERO_NAME][2::3] += (y - yc) * d_outputs["M_X"]
                if "M_Y" in d_outputs:
                    d_inputs[F_AERO_NAME][0::3] += (z - zc) * d_outputs["M_Y"]
                    d_inputs[F_AERO_NAME][2::3] += -(x - xc) * d_outputs["M_Y"]
                if "M_Z" in d_outputs:
                    d_inputs[F_AERO_NAME][0::3] += -(y - yc) * d_outputs["M_Z"]
                    d_inputs[F_AERO_NAME][1::3] += (x - xc) * d_outputs["M_Z"]
                if "CM_X" in d_outputs:
                    d_inputs[F_AERO_NAME][1::3] += (
                        -(z - zc) * d_outputs["CM_X"] / (q_inf * area * span)
                    )
                    d_inputs[F_AERO_NAME][2::3] += (
                        (y - yc) * d_outputs["CM_X"] / (q_inf * area * span)
                    )
                if "CM_Y" in d_outputs:
                    d_inputs[F_AERO_NAME][0::3] += (
                        (z - zc) * d_outputs["CM_Y"] / (q_inf * area * c)
                    )
                    d_inputs[F_AERO_NAME][2::3] += (
                        -(x - xc) * d_outputs["CM_Y"] / (q_inf * area * c)
                    )
                if "CM_Z" in d_outputs:
                    d_inputs[F_AERO_NAME][0::3] += (
                        -(y - yc) * d_outputs["CM_Z"] / (q_inf * area * span)
                    )
                    d_inputs[F_AERO_NAME][1::3] += (
                        (x - xc) * d_outputs["CM_Z"] / (q_inf * area * span)
                    )


def check_integrated_surface_force_partials():
    nnodes = 3
    prob = om.Problem()
    ivc = om.IndepVarComp()
    ivc.add_output(AOA_NAME, val=45.0, units="deg")
    ivc.add_output(YAW_NAME, val=135.0, units="deg")
    ivc.add_output(REF_AREA_NAME, val=0.2)
    ivc.add_output(MOMENT_CENTER_NAME, shape=3, val=np.zeros(3))
    ivc.add_output(REF_LENGTH_X_NAME, val=3.0)
    ivc.add_output(REF_LENGTH_Y_NAME, val=4.0)
    ivc.add_output(QINF_NAME, val=10.0)
    ivc.add_output(
        X_AERO_NAME, shape=3 * nnodes, val=np.random.rand(3 * nnodes), distributed=True
    )
    ivc.add_output(
        F_AERO_NAME, shape=3 * nnodes, val=np.random.rand(3 * nnodes), distributed=True
    )
    prob.model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
    prob.model.add_subsystem("forces", IntegratedSurfaceForces(), promotes_inputs=["*"])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True, method="cs")


if __name__ == "__main__":
    check_integrated_surface_force_partials()
