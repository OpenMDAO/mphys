class MPhysVariables:
    class Aerodynamics:
        class Surface:
            #: node coordinates, jig shape (includes geometry changes)
            COORDINATES_INITIAL = "x_aero0"

            #: node coordinates, deformed by geometry or structures
            COORDINATES_DEFORMED = "x_aero"

            #: displacement distribution
            DISPLACEMENTS = "u_aero"

            #: force distribution
            LOADS = "f_aero"

            #: temperature distribution
            TEMPERATURE = "T_convect"

            #: distribution of heat flux * local surface area
            HEAT_FLOW = "q_convect"

    class Structures:
        #: Coordinates, jig shape (including geometry changes)
        COORDINATES_INITIAL = "x_struct0"

        #: displacements at mesh nodes
        DISPLACEMENTS = "u_struct"

        #: loads at mesh nodes
        LOADS_FROM_AERODYNAMICS = "f_aero_struct"

        #: temperature at mesh nodes
        TEMPERATURE = "T_conduct"

        #: heat flux * local surface area at mesh nodes
        HEAT_FLOW = "q_conduct"
