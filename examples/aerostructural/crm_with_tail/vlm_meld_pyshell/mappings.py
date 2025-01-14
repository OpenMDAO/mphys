import numpy as np

## create a function which takes in elemental dv_struct, X, Y, and Z
## computes A/B/D and the mass-per-area of each element
## function must be complex-steppable across all inputs


def dv_map(dv_struct_el, Xel, Yel, Zel):

    ## material properties

    E_modulus = 73.1e9
    poisson = 0.33
    rho = 2780.0

    ## stress-strain Q matrix

    Q = (E_modulus / (1 - poisson**2)) * np.array(
        [[1, poisson, 0], [poisson, 1, 0], [0, 0, (1 - poisson) / 2]]
    )

    ## allocate

    if np.iscomplexobj(dv_struct_el) != 0 or np.iscomplexobj(np.c_[Xel, Yel, Zel]) != 0:
        ABD = np.zeros([6, 6, len(dv_struct_el)], dtype=complex)
    else:
        ABD = np.zeros([6, 6, len(dv_struct_el)])

    ## A and D matrices

    ABD[0:3, 0:3, :] = np.tile(dv_struct_el[:, 0], [3, 3, 1]) * np.dstack(
        [Q] * len(dv_struct_el)
    )
    ABD[3:6, 3:6, :] = np.tile(
        dv_struct_el[:, 0] * dv_struct_el[:, 0] * dv_struct_el[:, 0] / 12, [3, 3, 1]
    ) * np.dstack([Q] * len(dv_struct_el))

    ## mass per area

    mass_per_area = rho * dv_struct_el[:, 0]

    return ABD, mass_per_area


## create a function which takes in elemental dv_struct, X, Y, Z, strain, and curvature
## computes the failure function of each element
## function must be complex-steppable across all inputs


def f_struct_map(dv_struct_el, Xel, Yel, Zel, strain, curvature):

    ## material properties

    E_modulus = 73.1e9
    poisson = 0.33
    YS = 420e6

    ## stress-strain Q matrix

    Q = (E_modulus / (1 - poisson**2)) * np.array(
        [[1, poisson, 0], [poisson, 1, 0], [0, 0, (1 - poisson) / 2]]
    )

    ## stress computation

    stress = (strain @ Q) / YS

    ## failure function

    f_struct = np.sqrt(
        stress[:, 0] ** 2
        - stress[:, 0] * stress[:, 1]
        + stress[:, 1] ** 2
        + 3 * stress[:, 2] ** 2
    )

    return f_struct
