import numpy as np
from mpi4py import MPI

class Beam:
    def __init__(self, panel_chord, panel_width, N_el, comm=MPI.COMM_WORLD):
        self.panel_chord = panel_chord
        self.panel_width = panel_width
        self.N_el = N_el

        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()

        # create mesh
        x = np.linspace(0, self.panel_chord, num=self.N_el+1)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        
        # partitioning
        if self.rank == 0:
            self.owned = np.arange(self.N_el+1)
            ave, res = divmod(self.N_el+1, self.nprocs)
            counts = [ave + 1 if p < res else ave for p in range(self.nprocs)]

            starts = [sum(counts[:p]) for p in range(self.nprocs)]
            ends = [sum(counts[:p+1]) for p in range(self.nprocs)]

            self.owned = [self.owned[starts[p]:ends[p]] for p in range(self.nprocs)]

        else:
            self.owned = None

        # distribute mesh
        self.owned = comm.scatter(self.owned, root=0)
        if np.size(self.owned) == 0:
            self.owned = None
            self.n_nodes = 0
            self.x = np.zeros(0)
            self.y = np.zeros(0)
            self.z = np.zeros(0)
        else:
            self.n_nodes = len(self.owned)
            self.x = x[self.owned]
            self.y = y[self.owned]
            self.z = z[self.owned]

        self.n_dof = 6

        # free degrees of freedom for the FEA: simply-supported ends
        n = self.comm.allreduce(self.n_nodes, op=MPI.SUM)
        self.fdof = np.arange(n*2)
        self.fdof = np.delete(self.fdof, [0,-2])

    def compute_stiffness_matrix(self, x, EI):
        K = np.zeros([len(x)*2,len(x)*2])
        for i in range(len(x)-1):
            L = x[i+1] - x[i]
            K[i*2:i*2+4,i*2:i*2+4] += EI[i]*np.array([
                [12/L**3,  6/L**2, -12/L**3, 6/L**2],
                [6/L**2,   4/L,     -6/L**2, 2/L],
                [-12/L**3, -6/L**2, 12/L**3, -6/L**2],
                [6/L**2,   2/L,     -6/L**2, 4/L],
            ])
        return K

    def solve_system(self, f):
        x = self.comm.gather(self.xyz[0::3], root=0)
        f = self.comm.gather(f, root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)
            f = np.concatenate(f, axis=0)

            # compute stiffness matrix
            EI = (1/12)*self.modulus*self.panel_width*self.dv_struct**3
            K = self.compute_stiffness_matrix(x, EI)
            K = K[self.fdof,:][:,self.fdof]

            # extract the forces needed: dof 2 and 4
            F = self._extract_2dof(size=len(x)*2, mask=self.fdof, x=f)

            # solve
            U = np.zeros(len(x)*2)
            U[self.fdof] = np.linalg.solve(K, F)

        else:
            U = None

        # distribute output
        U = self._distribute_output_2dof(U)

        # pad outputs into dof 2 and 4
        U = self._pad_output_2dof(size=self.n_dof*self.n_nodes, x=U)

        return U

    def compute_residual(self, u, f):
        x = self.comm.gather(self.xyz[0::3], root=0)
        u = self.comm.gather(u, root=0)
        f = self.comm.gather(f, root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)
            u = np.concatenate(u, axis=0)
            f = np.concatenate(f, axis=0)

            # compute stiffness matrix
            EI = (1/12)*self.modulus*self.panel_width*self.dv_struct**3
            K = self.compute_stiffness_matrix(x, EI)
            K = K[self.fdof,:][:,self.fdof]

            # extract the forces needed: dof 2 and 4
            F = self._extract_2dof(size=len(x)*2, mask=self.fdof, x=f) 

            # extract the displacements needed: dof 2 and 4
            U = self._extract_2dof(size=len(x)*2, mask=self.fdof, x=u)

            # residual
            R = np.zeros(len(x)*2)
            R[self.fdof] = K@U - F

        else:
            R = None

        # distribute output
        R = self._distribute_output_2dof(R)

        # pad outputs into dof 2 and 4
        R = self._pad_output_2dof(size=self.n_dof*self.n_nodes, x=R)

        return R

    def bc_correction(self, u):
        x = self.comm.gather(self.x, root=0)
        u = self.comm.gather(u, root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)
            u = np.concatenate(u, axis=0)

            U = self._extract_2dof(size=len(x)*2, mask=None, x=u)
            U[self.fdof] = 0.

        else:
            U = None

        # distribute output
        U = self._distribute_output_2dof(U)

        # pad outputs into dof 2 and 4
        U = self._pad_output_2dof(size=self.n_dof*self.n_nodes, x=U)

        return U

    def set_adjoint(self, adjoint):
        adjoint = self.comm.gather(adjoint, root=0)

        # only run on rank 0
        if self.rank == 0:
            adjoint = np.concatenate(adjoint, axis=0)

            Adjoint = self._extract_2dof(size=len(adjoint)*2//self.n_dof, mask=None, x=adjoint)
            Adjoint[np.setdiff1d(np.arange(len(Adjoint)), self.fdof)] = 0.

        else:
            Adjoint = None

        # distribute output
        Adjoint = self._distribute_output_2dof(Adjoint)

        # pad outputs into dof 2 and 4
        Adjoint = self._pad_output_2dof(size=self.n_dof*self.n_nodes, x=Adjoint)

        return Adjoint

    def compute_stiffness_derivatives(self, u, adjoint):
        x = self.comm.gather(self.xyz[0::3], root=0)
        u = self.comm.gather(u, root=0)
        adjoint = self.comm.gather(adjoint, root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)
            u = np.concatenate(u, axis=0)
            adjoint = np.concatenate(adjoint, axis=0)

            # extract the terms needed: dof 2 and 4
            U = self._extract_2dof(size=len(x)*2, mask=None, x=u)
            Adjoint = self._extract_2dof(size=len(x)*2, mask=None, x=adjoint)

            # allocate output 
            d_dv_struct = np.zeros_like(self.dv_struct)
            d_x = np.zeros_like(x)

            EI = (1/12)*self.modulus*self.panel_width*self.dv_struct**3
            for i in range(len(x)-1):
                L = x[i+1] - x[i]

                # dv_struct derivatives
                EI_wrt_dv_struct = (1/12)*self.modulus*self.panel_width*3*self.dv_struct**2
                K_wrt_dv_struct = EI_wrt_dv_struct[i]*np.array([
                    [12/L**3,  6/L**2, -12/L**3, 6/L**2],
                    [6/L**2,   4/L,     -6/L**2, 2/L],
                    [-12/L**3, -6/L**2, 12/L**3, -6/L**2],
                    [6/L**2,   2/L,     -6/L**2, 4/L],
                ])
                d_dv_struct[i] += Adjoint[i*2:i*2+4]@K_wrt_dv_struct@U[i*2:i*2+4]

                # x derivatives
                K_wrt_L = EI[i]*np.array([
                    [-36/L**4, -12/L**3, 36/L**4,  -12/L**3],
                    [-12/L**3, -4/L**2,  12/L**3,  -2/L**2],
                    [36/L**4,  12/L**3,  -36/L**4, 12/L**3],
                    [-12/L**3, -2/L**2,  12/L**3,  -4/L**2],
                ])
                d_x[i+1] += Adjoint[i*2:i*2+4]@K_wrt_L@U[i*2:i*2+4]
                d_x[i]   -= Adjoint[i*2:i*2+4]@K_wrt_L@U[i*2:i*2+4]

            # modulus derivatives
            K = self.compute_stiffness_matrix(x, EI)
            K = K[self.fdof,:][:,self.fdof]
            d_modulus = Adjoint[self.fdof]@K@U[self.fdof]/self.modulus

        else:
            d_dv_struct = None
            d_x = None
            d_modulus = None

        # distribute output
        d_dv_struct = self.comm.bcast(d_dv_struct, root=0)
        d_x = self._distribute_output(d_x)
        d_modulus = self.comm.bcast(d_modulus, root=0)

        # pad outputs into correct dof
        d_xs = self._pad_output(size=3*self.n_nodes, n1=0, n2=3, x=d_x)

        return d_dv_struct, d_xs, d_modulus

    def compute_stress(self, u, aggregation_parameter):
        x = self.comm.gather(self.xyz[0::3], root=0)
        rotations = self.comm.gather(u[4::6], root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)
            rotations = np.concatenate(rotations, axis=0)

            # compute stresses
            stress = np.zeros(len(x)-1)
            for i in range(len(x)-1):
                stress[i] = self.modulus*self.dv_struct[i]*(rotations[i+1]-rotations[i])/(x[i+1] - x[i])/2

        else:
            stress = None

        stress = self.comm.bcast(stress, root=0)

        f = np.r_[stress,-stress]/self.yield_stress
        KS = np.log(np.sum(np.exp(f*aggregation_parameter)))/aggregation_parameter

        return stress, KS

    def compute_stress_derivatives(self, u, stress, aggregation_parameter, adjoint):
        x = self.comm.gather(self.xyz[0::3], root=0)
        rotations = self.comm.gather(u[4::6], root=0)

        f = np.r_[stress,-stress]/self.yield_stress

        func_struct_wrt_stress = (
            np.exp((aggregation_parameter*stress)/self.yield_stress) - 
            np.exp(-(aggregation_parameter*stress)/self.yield_stress)
            )/self.yield_stress/np.sum(np.exp(f*aggregation_parameter))

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)
            rotations = np.concatenate(rotations, axis=0)

            # allocate output 
            d_dv_struct = np.zeros_like(self.dv_struct)
            d_x = np.zeros_like(x)
            d_rotation = np.zeros_like(rotations)
            d_modulus = 0.

            for i in range(len(x)-1):
                # dv_struct derivatives
                d_dv_struct[i] += adjoint*func_struct_wrt_stress[i]*self.modulus*(rotations[i+1]-rotations[i])/(x[i+1] - x[i])/2

                # x derivatives
                stress_wrt_L = -stress[i]/(x[i+1] - x[i])
                d_x[i+1] += adjoint*func_struct_wrt_stress[i]*stress_wrt_L
                d_x[i]   -= adjoint*func_struct_wrt_stress[i]*stress_wrt_L

                # rotation derivatives
                stress_wrt_rotation = stress[i]/(rotations[i+1]-rotations[i])
                d_rotation[i+1] += adjoint*func_struct_wrt_stress[i]*stress_wrt_rotation
                d_rotation[i]   -= adjoint*func_struct_wrt_stress[i]*stress_wrt_rotation 

                # modulus derivatives
                stress_wrt_modulus = stress[i]/self.modulus
                d_modulus += adjoint*func_struct_wrt_stress[i]*stress_wrt_modulus     

            d_yield_stress = adjoint*np.dot(np.exp(f*aggregation_parameter)/np.sum(np.exp(f*aggregation_parameter)), -f/self.yield_stress)
     
        else:
            d_dv_struct = None
            d_x = None
            d_rotation = None
            d_modulus = None
            d_yield_stress = None

        # distribute output
        d_dv_struct = self.comm.bcast(d_dv_struct, root=0)
        d_x = self._distribute_output(d_x)
        d_rotation = self._distribute_output(d_rotation)
        d_modulus = self.comm.bcast(d_modulus, root=0)
        d_yield_stress = self.comm.bcast(d_yield_stress, root=0)

        # pad outputs into correct dof
        d_xs = self._pad_output(size=3*self.n_nodes, n1=0, n2=3, x=d_x)
        d_us = self._pad_output(size=6*self.n_nodes, n1=4, n2=6, x=d_rotation)

        return d_dv_struct, d_xs, d_us, d_modulus, d_yield_stress

    def compute_mass(self):
        x = self.comm.gather(self.xyz[0::3], root=0)

        # only run on rank 0
        mass = 0.
        if self.rank == 0:
            x = np.concatenate(x, axis=0)

            for i in range(len(x)-1):
                mass += (x[i+1]-x[i])*self.panel_width*self.dv_struct[i]*self.density

        mass = self.comm.bcast(mass)
        return mass

    def compute_mass_derivatives(self, adjoint):
        x = self.comm.gather(self.xyz[0::3], root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)

            # allocate output 
            d_dv_struct = np.zeros_like(self.dv_struct)
            d_x = np.zeros_like(x)
            d_density = 0.

            for i in range(len(x)-1):
                # dv_struct derivatives
                d_dv_struct[i] += adjoint*(x[i+1]-x[i])*self.panel_width*self.density

                # x derivatives
                d_x[i+1] += adjoint*self.panel_width*self.dv_struct[i]*self.density
                d_x[i]   -= adjoint*self.panel_width*self.dv_struct[i]*self.density

                # density derivatives
                d_density += adjoint*(x[i+1]-x[i])*self.panel_width*self.dv_struct[i]

        else:
            d_dv_struct = None
            d_x = None
            d_density = None

        # distribute output
        d_dv_struct = self.comm.bcast(d_dv_struct, root=0)
        d_x = self._distribute_output(d_x)
        d_density = self.comm.bcast(d_density, root=0)

        # pad outputs into correct dof
        d_xs = self._pad_output(size=3*self.n_nodes, n1=0, n2=3, x=d_x)

        return d_dv_struct, d_xs, d_density

    def _distribute_output(self, x):
        x = self.comm.bcast(x, root=0)
        if self.owned is not None:
            x = x[self.owned]
        return x

    def _distribute_output_2dof(self, x):
        x = self.comm.bcast(x, root=0)
        if self.owned is not None:
            owned = np.c_[self.owned*2,self.owned*2+1].flatten()
            x = x[owned]
        return x

    def _pad_output(self, size, n1, n2, x):
        if self.owned is not None:
            X = np.zeros(size)
            X[n1::n2] = x
        else:
            X = np.zeros(0)
        return X

    def _pad_output_2dof(self, size, x):
        if self.owned is not None:
            X = np.zeros(size)
            X[2::6] = x[0::2]
            X[4::6] = x[1::2]
        else:
            X = np.zeros(0)
        return X

    def _extract_2dof(self, size, mask, x):
        X = np.zeros(size)
        X[0::2] = x[2::6]
        X[1::2] = x[4::6]
        if mask is not None:
            X = X[mask]
        return X

    def write_output(self, u, stress):
        x = self.comm.gather(self.xyz[0::3], root=0)
        u = self.comm.gather(u[2::6], root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)
            u = np.concatenate(u, axis=0)

            x = np.c_[x[0:-1],x[1:]].flatten()/self.panel_chord
            u = np.c_[u[0:-1],u[1:]].flatten()/self.panel_chord
            stress = np.c_[stress,stress].flatten()/self.yield_stress

            f = open('structures_output.dat',"w+")
            f.write('TITLE = "structural beam data"\n')
            f.write('VARIABLES = "x", "w", "stress"\n')
            f.write('ZONE I=' + str(len(x)) + ', F=POINT\n')
            for i in range(len(x)):
                f.write(str(x[i]) + ' ' + str(u[i]) + ' ' + str(stress[i]) + '\n')
            f.close()

