import numpy as np
import openmdao.api as om

class IntegratedSurfaceForces(om.ExplicitComponent):
    def setup(self):
        self.add_input('aoa',desc = 'angle of attack', units='rad',tags=['mphys_input'])
        self.add_input('yaw',desc = 'yaw angle',units='rad',tags=['mphys_input'])
        self.add_input('ref_area', val = 1.0,tags=['mphys_input'])
        self.add_input('moment_center',shape=3,tags=['mphys_input'])
        self.add_input('ref_length', val = 1.0,tags=['mphys_input'])
        self.add_input('q_inf', val = 1.0,tags=['mphys_input'])

        self.add_input('x_aero', shape_by_conn=True,
                                 distributed=True,
                                 desc = 'surface coordinates',
                                 tags=['mphys_coupling'])
        self.add_input('f_aero', shape_by_conn=True,
                                 distributed=True,
                                 desc = 'dimensional forces at nodes',
                                 tags=['mphys_coupling'])

        self.add_output('C_L', desc = 'Lift coefficient', tags=['mphys_result'])
        self.add_output('C_D', desc = 'Drag coefficient', tags=['mphys_result'])
        self.add_output('C_X', desc = 'X Force coefficient', tags=['mphys_result'])
        self.add_output('C_Y', desc = 'Y Force coefficient', tags=['mphys_result'])
        self.add_output('C_Z', desc = 'Z Force coefficient', tags=['mphys_result'])
        self.add_output('CM_X', desc = 'X Moment coefficient', tags=['mphys_result'])
        self.add_output('CM_Y', desc = 'Y Moment coefficient', tags=['mphys_result'])
        self.add_output('CM_Z', desc = 'Z Moment coefficient', tags=['mphys_result'])

        self.add_output('Lift', desc = 'Total Lift', tags=['mphys_result'])
        self.add_output('Drag', desc = 'Total Drag', tags=['mphys_result'])
        self.add_output('F_X', desc = 'Total X Force', tags=['mphys_result'])
        self.add_output('F_Y', desc = 'Total Y Force', tags=['mphys_result'])
        self.add_output('F_Z', desc = 'Total Z Force', tags=['mphys_result'])
        self.add_output('M_X', desc = 'Total X Moment', tags=['mphys_result'])
        self.add_output('M_Y', desc = 'Total Y Moment', tags=['mphys_result'])
        self.add_output('M_Z', desc = 'Total Z Moment', tags=['mphys_result'])

    def compute(self,inputs,outputs):
        aoa = inputs['aoa']
        yaw  = inputs['yaw']
        area = inputs['ref_area']
        q_inf = inputs['q_inf']
        xc = inputs['moment_center'][0]
        yc = inputs['moment_center'][1]
        zc = inputs['moment_center'][2]
        c = inputs['ref_length']

        x  = inputs['x_aero'][0::3]
        y  = inputs['x_aero'][1::3]
        z  = inputs['x_aero'][2::3]

        fx = inputs['f_aero'][0::3]
        fy = inputs['f_aero'][1::3]
        fz = inputs['f_aero'][2::3]

        fx_total = self.comm.allreduce(np.sum(fx))
        fy_total = self.comm.allreduce(np.sum(fy))
        fz_total = self.comm.allreduce(np.sum(fz))

        outputs['F_X'] = fx_total
        outputs['F_Y'] = fy_total
        outputs['F_Z'] = fz_total
        outputs['C_X'] = fx_total / (q_inf * area)
        outputs['C_Y'] = fy_total / (q_inf * area)
        outputs['C_Z'] = fz_total / (q_inf * area)

        outputs['Lift'] = -fx_total * np.sin(aoa) + fz_total * np.cos(aoa)
        outputs['Drag'] = ( fx_total * np.cos(aoa) * np.cos(yaw)
                          - fy_total * np.sin(yaw)
                          + fz_total * np.sin(aoa) * np.cos(yaw)
                          )

        outputs['C_L'] = outputs['Lift'] / (q_inf * area)
        outputs['C_D'] = outputs['Drag'] / (q_inf * area)

        m_x = self.comm.allreduce( np.dot(fz,(y-yc)) - np.dot(fy,(z-zc)))
        m_y = self.comm.allreduce(-np.dot(fz,(x-xc)) + np.dot(fx,(z-zc)))
        m_z = self.comm.allreduce( np.dot(fy,(x-xc)) - np.dot(fx,(y-yc)))

        outputs['M_X'] =  m_x
        outputs['M_Y'] =  m_y
        outputs['M_Z'] =  m_z

        outputs['CM_X'] = m_x / (q_inf * area * c)
        outputs['CM_Y'] = m_y / (q_inf * area * c)
        outputs['CM_Z'] = m_z / (q_inf * area * c)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        aoa = inputs['aoa']
        yaw  = inputs['yaw']
        area = inputs['ref_area']
        q_inf = inputs['q_inf']
        xc = inputs['moment_center'][0]
        yc = inputs['moment_center'][1]
        zc = inputs['moment_center'][2]
        c = inputs['ref_length']

        x  = inputs['x_aero'][0::3]
        y  = inputs['x_aero'][1::3]
        z  = inputs['x_aero'][2::3]

        fx = inputs['f_aero'][0::3]
        fy = inputs['f_aero'][1::3]
        fz = inputs['f_aero'][2::3]

        fx_total = self.comm.allreduce(np.sum(fx))
        fy_total = self.comm.allreduce(np.sum(fy))
        fz_total = self.comm.allreduce(np.sum(fz))

        lift = -fx_total * np.sin(aoa) + fz_total * np.cos(aoa)
        drag = ( fx_total * np.cos(aoa) * np.cos(yaw)
               - fy_total * np.sin(yaw)
               + fz_total * np.sin(aoa) * np.cos(yaw)
               )

        m_x = self.comm.allreduce( np.dot(fz,(y-yc)) - np.dot(fy,(z-zc)))
        m_y = self.comm.allreduce(-np.dot(fz,(x-xc)) + np.dot(fx,(z-zc)))
        m_z = self.comm.allreduce( np.dot(fy,(x-xc)) - np.dot(fx,(y-yc)))

        if mode == 'fwd':
            if 'aoa' in d_inputs:
                daoa_rad = d_inputs['aoa']
                if 'Lift' in d_outputs or 'C_L' in d_outputs:
                    d_lift_d_aoa = ( - fx_total * np.cos(aoa) * daoa_rad
                                       - fz_total * np.sin(aoa) * daoa_rad )

                    if 'Lift' in d_outputs:
                        d_outputs['Lift'] += d_lift_d_aoa
                    if 'C_L' in d_outputs:
                        d_outputs['C_L'] += d_lift_d_aoa / (q_inf * area)
                if 'Drag' in d_outputs or 'C_D' in d_outputs:
                    d_drag_d_aoa = ( fx_total * (-np.sin(aoa) * daoa_rad) * np.cos(yaw)
                                     + fz_total * ( np.cos(aoa) * daoa_rad) * np.cos(yaw))
                    if 'Drag' in d_outputs:
                        d_outputs['Drag'] += d_drag_d_aoa
                    if 'C_D' in d_outputs:
                        d_outputs['C_D'] += d_drag_d_aoa / (q_inf * area)

            if 'yaw' in d_inputs:
                dyaw_rad = d_inputs['yaw']
                if 'Drag' in d_outputs or 'C_D' in d_outputs:
                    d_drag_d_yaw = ( fx_total * np.cos(aoa) * (-np.sin(yaw) * dyaw_rad)
                                    - fy_total * np.cos(yaw) * dyaw_rad
                                    + fz_total * np.sin(aoa) * (-np.sin(yaw) * dyaw_rad)
                                    )
                    if 'Drag' in d_outputs:
                        d_outputs['Drag'] += d_drag_d_yaw
                    if 'C_D' in d_outputs:
                        d_outputs['C_D'] += d_drag_d_yaw / (q_inf * area)

            if 'ref_area' in d_inputs:
                d_nondim = - d_inputs['ref_area'] / (q_inf * area**2.0)
                if 'C_X' in d_outputs:
                    d_outputs['C_X'] += fx_total * d_nondim
                if 'C_Y' in d_outputs:
                    d_outputs['C_Y'] += fy_total * d_nondim
                if 'C_Z' in d_outputs:
                    d_outputs['C_Z'] += fz_total * d_nondim
                if 'C_L' in d_outputs:
                    d_outputs['C_L'] += lift * d_nondim
                if 'C_D' in d_outputs:
                    d_outputs['C_D'] += drag * d_nondim
                if 'CM_X' in d_outputs:
                    d_outputs['CM_X'] += m_x * d_nondim / c
                if 'CM_X' in d_outputs:
                    d_outputs['CM_Y'] += m_y * d_nondim / c
                if 'CM_Z' in d_outputs:
                    d_outputs['CM_Z'] += m_z * d_nondim / c
            if 'moment_center' in d_inputs:
                dxc = d_inputs['moment_center'][0]
                dyc = d_inputs['moment_center'][1]
                dzc = d_inputs['moment_center'][2]
                if 'M_X' in d_outputs:
                    d_outputs['M_X'] += -fz_total * dyc + fy_total * dzc
                if 'M_Y' in d_outputs:
                    d_outputs['M_Y'] +=  fz_total * dxc - fx_total * dzc
                if 'M_Z' in d_outputs:
                    d_outputs['M_Z'] += -fy_total * dxc + fx_total * dyc
                if 'CM_X' in d_outputs:
                    d_outputs['CM_X'] += (-fz_total * dyc + fy_total * dzc) / (q_inf * area * c)
                if 'CM_Y' in d_outputs:
                    d_outputs['CM_Y'] += ( fz_total * dxc - fx_total * dzc) / (q_inf * area * c)
                if 'CM_Z' in d_outputs:
                    d_outputs['CM_Z'] += (-fy_total * dxc + fx_total * dyc) / (q_inf * area * c)

            if 'ref_length' in d_inputs:
                d_nondim = - d_inputs['ref_length'] / (q_inf * area * c**2.0)
                if 'CM_X' in d_outputs:
                    d_outputs['CM_X'] += m_x * d_nondim
                if 'CM_X' in d_outputs:
                    d_outputs['CM_Y'] += m_y * d_nondim
                if 'CM_Z' in d_outputs:
                    d_outputs['CM_Z'] += m_z * d_nondim

            if 'q_inf' in d_inputs:
                d_nondim = - d_inputs['q_inf'] / (q_inf**2.0 * area)
                if 'C_X' in d_outputs:
                    d_outputs['C_X'] += fx_total * d_nondim
                if 'C_Y' in d_outputs:
                    d_outputs['C_Y'] += fy_total * d_nondim
                if 'C_Z' in d_outputs:
                    d_outputs['C_Z'] += fz_total * d_nondim
                if 'C_L' in d_outputs:
                    d_outputs['C_L'] += lift * d_nondim
                if 'C_D' in d_outputs:
                    d_outputs['C_D'] += drag * d_nondim
                if 'CM_X' in d_outputs:
                    d_outputs['CM_X'] += m_x * d_nondim / c
                if 'CM_X' in d_outputs:
                    d_outputs['CM_Y'] += m_y * d_nondim / c
                if 'CM_Z' in d_outputs:
                    d_outputs['CM_Z'] += m_z * d_nondim / c

            if 'x_aero' in d_inputs:
                dx = d_inputs['x_aero'][0::3]
                dy = d_inputs['x_aero'][1::3]
                dz = d_inputs['x_aero'][2::3]
                if 'M_X' in d_outputs:
                    d_outputs['M_X'] +=  np.dot(fz,dy) - np.dot(fy,dz)
                if 'M_Y' in d_outputs:
                    d_outputs['M_Y'] += -np.dot(fz,dx) + np.dot(fx,dz)
                if 'M_Z' in d_outputs:
                    d_outputs['M_Z'] +=  np.dot(fy,dx) - np.dot(fx,dy)
                if 'CM_X' in d_outputs:
                    d_outputs['CM_X'] += ( np.dot(fz,dy) - np.dot(fy,dz)) / (q_inf * area * c)
                if 'CM_Y' in d_outputs:
                    d_outputs['CM_Y'] += (-np.dot(fz,dx) + np.dot(fx,dz)) / (q_inf * area * c)
                if 'CM_Z' in d_outputs:
                    d_outputs['CM_Z'] += ( np.dot(fy,dx) - np.dot(fx,dy)) / (q_inf * area * c)

            if 'f_aero' in d_inputs:
                dfx = d_inputs['f_aero'][0::3]
                dfy = d_inputs['f_aero'][1::3]
                dfz = d_inputs['f_aero'][2::3]
                dfx_total = np.sum(dfx)
                dfy_total = np.sum(dfy)
                dfz_total = np.sum(dfz)
                if 'F_X' in d_outputs:
                    d_outputs['F_X'] += dfx_total
                if 'F_Y' in d_outputs:
                    d_outputs['F_Y'] += dfy_total
                if 'F_Z' in d_outputs:
                    d_outputs['F_Z'] += dfz_total
                if 'C_X' in d_outputs:
                    d_outputs['C_X'] += dfx_total / (q_inf * area)
                if 'C_Y' in d_outputs:
                    d_outputs['C_Y'] += dfy_total / (q_inf * area)
                if 'C_Z' in d_outputs:
                    d_outputs['C_Z'] += dfz_total / (q_inf * area)
                if 'Lift' in d_outputs:
                    d_outputs['Lift'] += -dfx_total * np.sin(aoa) + dfz_total * np.cos(aoa)
                if 'Drag' in d_outputs:
                    d_outputs['Drag'] += ( dfx_total * np.cos(aoa) * np.cos(yaw)
                                         - dfy_total * np.sin(yaw)
                                         + dfz_total * np.sin(aoa) * np.cos(yaw)
                                         )
                if 'C_L' in d_outputs:
                    d_outputs['C_L'] += (-dfx_total * np.sin(aoa) + dfz_total * np.cos(aoa)) / (q_inf * area)
                if 'C_D' in d_outputs:
                    d_outputs['C_D'] += ( dfx_total * np.cos(aoa) * np.cos(yaw)
                                        - dfy_total * np.sin(yaw)
                                        + dfz_total * np.sin(aoa) * np.cos(yaw)
                                        ) / (q_inf * area)

                if 'M_X' in d_outputs:
                    d_outputs['M_X'] +=  np.dot(dfz,(y-yc)) - np.dot(dfy,(z-zc))
                if 'M_Y' in d_outputs:
                    d_outputs['M_Y'] += -np.dot(dfz,(x-xc)) + np.dot(dfx,(z-zc))
                if 'M_Z' in d_outputs:
                    d_outputs['M_Z'] +=  np.dot(dfy,(x-xc)) - np.dot(dfx,(y-yc))
                if 'CM_X' in d_outputs:
                    d_outputs['CM_X'] += ( np.dot(dfz,(y-yc)) - np.dot(dfy,(z-zc))) / (q_inf * area * c)
                if 'CM_Y' in d_outputs:
                    d_outputs['CM_Y'] += (-np.dot(dfz,(x-xc)) + np.dot(dfx,(z-zc))) / (q_inf * area * c)
                if 'CM_Z' in d_outputs:
                    d_outputs['CM_Z'] += ( np.dot(dfy,(x-xc)) - np.dot(dfx,(y-yc))) / (q_inf * area * c)

        elif mode == 'rev':
            if 'aoa' in d_inputs:
                if 'Lift' in d_outputs or 'C_L' in d_outputs:
                    d_lift = d_outputs['Lift'] if 'Lift' in d_outputs else 0.0
                    d_cl   = d_outputs['C_L']  if 'C_L'  in d_outputs else 0.0
                    d_inputs['aoa'] += ( - fx_total * np.cos(aoa)
                                         - fz_total * np.sin(aoa)
                                       ) * (d_lift + d_cl / (q_inf * area))

                if 'Drag' in d_outputs or 'C_D' in d_outputs:
                    d_drag = d_outputs['Drag'] if 'Drag' in d_outputs else 0.0
                    d_cd   = d_outputs['C_D']  if 'C_D'  in d_outputs else 0.0
                    d_inputs['aoa'] += ( fx_total * (-np.sin(aoa)) * np.cos(yaw)
                                       + fz_total * ( np.cos(aoa)) * np.cos(yaw)
                                       ) * (d_drag + d_cd / (q_inf * area))
            if 'yaw' in d_inputs:
                if 'Drag' in d_outputs or 'C_D' in d_outputs:
                    d_drag = d_outputs['Drag'] if 'Drag' in d_outputs else 0.0
                    d_cd   = d_outputs['C_D']  if 'C_D'  in d_outputs else 0.0
                    d_inputs['yaw'] += ( fx_total * np.cos(aoa) * (-np.sin(yaw))
                                       - fy_total * np.cos(yaw)
                                       + fz_total * np.sin(aoa) * (-np.sin(yaw))
                                       ) * (d_drag + d_cd / (q_inf * area))

            if 'ref_area' in d_inputs:
                d_nondim = - 1.0 / (q_inf * area**2.0)
                if 'C_X' in d_outputs:
                    d_inputs['ref_area'] += d_outputs['C_X'] * fx_total * d_nondim
                if 'C_Y' in d_outputs:
                    d_inputs['ref_area'] += d_outputs['C_Y'] * fy_total * d_nondim
                if 'C_Z' in d_outputs:
                    d_inputs['ref_area'] += d_outputs['C_Z'] * fz_total * d_nondim
                if 'C_L' in d_outputs:
                    d_inputs['ref_area'] += d_outputs['C_L'] * lift * d_nondim
                if 'C_D' in d_outputs:
                    d_inputs['ref_area'] += d_outputs['C_D'] * drag * d_nondim
                if 'CM_X' in d_outputs:
                    d_inputs['ref_area'] += d_outputs['CM_X'] * m_x * d_nondim / c
                if 'CM_X' in d_outputs:
                    d_inputs['ref_area'] += d_outputs['CM_Y'] * m_y * d_nondim / c
                if 'CM_Z' in d_outputs:
                    d_inputs['ref_area'] += d_outputs['CM_Z'] * m_z * d_nondim / c

            if 'moment_center' in d_inputs:
                if 'M_X' in d_outputs:
                    d_inputs['moment_center'][1] += -fz_total * d_outputs['M_X']
                    d_inputs['moment_center'][2] +=  fy_total * d_outputs['M_X']
                if 'M_Y' in d_outputs:
                    d_inputs['moment_center'][0] +=  fz_total * d_outputs['M_Y']
                    d_inputs['moment_center'][2] += -fx_total * d_outputs['M_Y']
                if 'M_Z' in d_outputs:
                    d_inputs['moment_center'][0] += -fy_total * d_outputs['M_Z']
                    d_inputs['moment_center'][1] +=  fx_total * d_outputs['M_Z']
                if 'CM_X' in d_outputs:
                    d_inputs['moment_center'][1] += -fz_total * d_outputs['CM_X'] / (q_inf * area * c)
                    d_inputs['moment_center'][2] +=  fy_total * d_outputs['CM_X'] / (q_inf * area * c)
                if 'CM_Y' in d_outputs:
                    d_inputs['moment_center'][0] +=  fz_total * d_outputs['CM_Y'] / (q_inf * area * c)
                    d_inputs['moment_center'][2] += -fx_total * d_outputs['CM_Y'] / (q_inf * area * c)
                if 'CM_Z' in d_outputs:
                    d_inputs['moment_center'][0] += -fy_total * d_outputs['CM_Z'] / (q_inf * area * c)
                    d_inputs['moment_center'][1] +=  fx_total * d_outputs['CM_Z'] / (q_inf * area * c)
            if 'ref_length' in d_inputs:
                d_nondim = - 1.0 / (q_inf * area * c**2.0)
                if 'CM_X' in d_outputs:
                    d_inputs['ref_length'] += m_x * d_nondim * d_outputs['CM_X']
                if 'CM_X' in d_outputs:
                    d_inputs['ref_length'] += m_y * d_nondim * d_outputs['CM_Y']
                if 'CM_Z' in d_outputs:
                    d_inputs['ref_length'] += m_z * d_nondim * d_outputs['CM_Z']

            if 'q_inf' in d_inputs:
                d_nondim = - 1.0 / (q_inf**2.0 * area)
                if 'C_X' in d_outputs:
                    d_inputs['q_inf'] += d_outputs['C_X'] * fx_total * d_nondim
                if 'C_Y' in d_outputs:
                    d_inputs['q_inf'] += d_outputs['C_Y'] * fy_total * d_nondim
                if 'C_Z' in d_outputs:
                    d_inputs['q_inf'] += d_outputs['C_Z'] * fz_total * d_nondim
                if 'C_L' in d_outputs:
                    d_inputs['q_inf'] += d_outputs['C_L'] * lift * d_nondim
                if 'C_D' in d_outputs:
                    d_inputs['q_inf'] += d_outputs['C_D'] * drag * d_nondim
                if 'CM_X' in d_outputs:
                    d_inputs['q_inf'] += d_outputs['CM_X'] * m_x * d_nondim / c
                if 'CM_X' in d_outputs:
                    d_inputs['q_inf'] += d_outputs['CM_Y'] * m_y * d_nondim / c
                if 'CM_Z' in d_outputs:
                    d_inputs['q_inf'] += d_outputs['CM_Z'] * m_z * d_nondim / c

            if 'x_aero' in d_inputs:
                nondim = 1.0 / (q_inf * area * c)
                dm_x = d_outputs['M_X'] if 'M_X' in d_outputs else 0.0
                dm_y = d_outputs['M_Y'] if 'M_Y' in d_outputs else 0.0
                dm_z = d_outputs['M_Z'] if 'M_Z' in d_outputs else 0.0
                dcm_x = d_outputs['CM_X']*nondim if 'CM_X' in d_outputs else 0.0
                dcm_y = d_outputs['CM_Y']*nondim if 'CM_Y' in d_outputs else 0.0
                dcm_z = d_outputs['CM_Z']*nondim if 'CM_Z' in d_outputs else 0.0
                d_inputs['x_aero'][0::3] += -fz * (dm_y + dcm_y) + fy * (dm_z + dcm_z)
                d_inputs['x_aero'][1::3] +=  fz * (dm_x + dcm_x) - fx * (dm_z + dcm_z)
                d_inputs['x_aero'][2::3] += -fy * (dm_x + dcm_x) + fx * (dm_y + dcm_y)

            if 'f_aero' in d_inputs:
                if 'F_X' in d_outputs:
                    d_inputs['f_aero'][0::3] += d_outputs['F_X']
                if 'F_Y' in d_outputs:
                    d_inputs['f_aero'][1::3] += d_outputs['F_Y']
                if 'F_Z' in d_outputs:
                    d_inputs['f_aero'][2::3] += d_outputs['F_Z']
                if 'C_X' in d_outputs:
                    d_inputs['f_aero'][0::3] += d_outputs['C_X'] / (q_inf * area)
                if 'C_Y' in d_outputs:
                    d_inputs['f_aero'][1::3] += d_outputs['C_Y'] / (q_inf * area)
                if 'C_Z' in d_outputs:
                    d_inputs['f_aero'][2::3] += d_outputs['C_Z'] / (q_inf * area)
                if 'Lift' in d_outputs:
                    d_inputs['f_aero'][0::3] += -np.sin(aoa) * d_outputs['Lift']
                    d_inputs['f_aero'][2::3] +=  np.cos(aoa) * d_outputs['Lift']
                if 'Drag' in d_outputs:
                    d_inputs['f_aero'][0::3] +=  np.cos(aoa) * np.cos(yaw) * d_outputs['Drag']
                    d_inputs['f_aero'][1::3] += -np.sin(yaw) * d_outputs['Drag']
                    d_inputs['f_aero'][2::3] +=  np.sin(aoa) * np.cos(yaw) * d_outputs['Drag']
                if 'C_L' in d_outputs:
                    d_inputs['f_aero'][0::3] += -np.sin(aoa) * d_outputs['C_L'] / (q_inf * area)
                    d_inputs['f_aero'][2::3] +=  np.cos(aoa) * d_outputs['C_L'] / (q_inf * area)
                if 'C_D' in d_outputs:
                    d_inputs['f_aero'][0::3] +=  np.cos(aoa) * np.cos(yaw) * d_outputs['C_D'] / (q_inf * area)
                    d_inputs['f_aero'][1::3] += -np.sin(yaw) * d_outputs['C_D'] / (q_inf * area)
                    d_inputs['f_aero'][2::3] +=  np.sin(aoa) * np.cos(yaw) * d_outputs['C_D'] / (q_inf * area)

                if 'M_X' in d_outputs:
                    d_inputs['f_aero'][1::3] += -(z-zc) * d_outputs['M_X']
                    d_inputs['f_aero'][2::3] +=  (y-yc) * d_outputs['M_X']
                if 'M_Y' in d_outputs:
                    d_inputs['f_aero'][0::3] +=  (z-zc) * d_outputs['M_Y']
                    d_inputs['f_aero'][2::3] += -(x-xc) * d_outputs['M_Y']
                if 'M_Z' in d_outputs:
                    d_inputs['f_aero'][0::3] += -(y-yc) * d_outputs['M_Z']
                    d_inputs['f_aero'][1::3] +=  (x-xc) * d_outputs['M_Z']
                if 'CM_X' in d_outputs:
                    d_inputs['f_aero'][1::3] += -(z-zc) * d_outputs['CM_X'] / (q_inf * area * c)
                    d_inputs['f_aero'][2::3] +=  (y-yc) * d_outputs['CM_X'] / (q_inf * area * c)
                if 'CM_Y' in d_outputs:
                    d_inputs['f_aero'][0::3] +=  (z-zc) * d_outputs['CM_Y'] / (q_inf * area * c)
                    d_inputs['f_aero'][2::3] += -(x-xc) * d_outputs['CM_Y'] / (q_inf * area * c)
                if 'CM_Z' in d_outputs:
                    d_inputs['f_aero'][0::3] += -(y-yc) * d_outputs['CM_Z'] / (q_inf * area * c)
                    d_inputs['f_aero'][1::3] +=  (x-xc) * d_outputs['CM_Z'] / (q_inf * area * c)

def check_integrated_surface_force_partials():
    nnodes = 3
    prob = om.Problem()
    ivc = om.IndepVarComp()
    ivc.add_output('aoa',val=45.0, units='deg')
    ivc.add_output('yaw',val=135.0, units='deg')
    ivc.add_output('ref_area',val=0.2)
    ivc.add_output('moment_center',shape=3,val=np.zeros(3))
    ivc.add_output('ref_length', val = 3.0)
    ivc.add_output('q_inf',val=10.0)
    ivc.add_output('x_aero',shape=3*nnodes,val=np.random.rand(3*nnodes),distributed=True)
    ivc.add_output('f_aero',shape=3*nnodes,val=np.random.rand(3*nnodes),distributed=True)
    prob.model.add_subsystem('ivc',ivc,promotes_outputs=['*'])
    prob.model.add_subsystem('forces',IntegratedSurfaceForces(),
                                      promotes_inputs=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True, method='cs')

if __name__ == '__main__':
    check_integrated_surface_force_partials()
