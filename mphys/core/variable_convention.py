class MPhysVariables:
    class Aerodynamics:
        class FlowConditions:
            ANGLE_OF_ATTACK = "angle_of_attack"
            YAW_ANGLE = "yaw_angle"
            MACH_NUMBER = "mach_number"
            REYNOLDS_NUMBER = "reynolds_number"
            DYNAMIC_PRESSURE = "dynamic_pressure"

        #TODO add propulsion coupling variables

        class Surface:
            #: displacement distribution
            DISPLACEMENTS = "u_aero"

            #: force distribution
            LOADS = "f_aero"

            #: temperature distribution
            TEMPERATURE = "T_aero"

            #: distribution of heat flux * local surface area
            HEAT_FLOW = "q_aero"

            #: node coordinates at start of the analysis (jig shape)
            COORDINATES_INITIAL = "x_aero0"

            #: current node coordinates, deformed by geometry and/or structures
            COORDINATES = "x_aero"

            class Geometry:
                #: node coordinates, input to geometry subsystem
                COORDINATES_INPUT = "x_aero0_geometry_input"

                #: node coordinates, output of geometry subsystem
                COORDINATES_OUTPUT = "x_aero0_geometry_output"

            class Mesh:
                #: node coordinates, original surface from mesh file
                COORDINATES = "x_aero0_mesh"

    class Structures:
        #: displacements at mesh nodes
        DISPLACEMENTS = "u_struct"

        #: loads at mesh nodes
        LOADS_FROM_AERODYNAMICS = "f_aero_struct"

        #: temperature at mesh nodes
        TEMPERATURE = "T_struct"

        #: heat flux * local surface area at mesh nodes
        HEAT_FLOW = "q_aero_struct"

        #: Coordinates at start of analysis
        COORDINATES = "x_struct0"

        class Geometry:
            #: node coordinates, input to geometry subsystem
            COORDINATES_INPUT = "x_struct0_geometry_input"

            #: node coordinates, output of geometry subsystem
            COORDINATES_OUTPUT = "x_struct0_geometry_output"

        class Mesh:
            #: node coordinates, original (no geometry changes, no deflections)
            COORDINATES = "x_struct0_mesh"

        class Thermal:
            """
            In physical space, the thermal domain is the same as the structures
            domain, but in some situations a different mesh is used.
            """
            #: Coordinates at start of analysis
            COORDINATES = "x_thermal0"

            class Geometry:
                #: node coordinates, input to geometry subsystem
                COORDINATES_INPUT = "x_thermal0_geometry_input"

                #: node coordinates, output of geometry subsystem
                COORDINATES_OUTPUT = "x_thermal0_geometry_output"

            class Mesh:
                #: node coordinates, original (no geometry changes)
                COORDINATES = "x_thermal0_mesh"

    #TODO add propulsion
    #class Propulsion:
