
class MPhysVariables:
    # coordinates
    coordinates_initial_aerodynamic = 'x_aero0'
    coordinates_initial_structural = 'x_struct0'
    coordinates_deformed_aerodynamic = 'x_aero'
    coordinates_deformed_structural = 'x_struct'

    # aerostructural coupling states
    displacements_aerodynamic = 'u_aero'
    displacements_structural = 'u_struct'
    loads_aerodynamic = 'f_aero'
    loads_structural_from_aerodynamics = 'f_aero_struct'

    # thermal coupling states
    temperature_conduction ='T_conduct'
    heat_flow_conduction ='q_conduct'
    temperature_convection ='T_convect'
    heat_flow_convection ='q_convect'
