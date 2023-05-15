
class MPhysVariables:

    class Aerodynamics:
        #: Surface coordinates, jig shape (includes geometry changes)
        coordinates_initial = 'x_aero0'

        #: Surface coordinates, deformed by geometry or structures
        coordinates_deformed = 'x_aero'

        #: Surface displacements
        displacements = 'u_aero'

        #: Surface forces
        loads = 'f_aero'

        #: Surface temperature distribution
        temperature_convection ='T_convect'

        #: Surface distribution of heat flux * local surface area
        heat_flow_convection ='q_convect'

    class Structures:
        #: Coordinates, jig shape (including geometry changes)
        coordinates_initial = 'x_struct0'

        #: displacements at mesh nodes
        displacements = 'u_struct'

        #: loads at mesh nodes
        loads_from_aerodynamics = 'f_aero_struct'

        #: Temperature at mesh nodes
        temperature_conduction ='T_conduct'

        #: Heat flux * local surface area at mesh nodes
        heat_flow_conduction ='q_conduct'
