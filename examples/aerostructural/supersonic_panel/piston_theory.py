import numpy as np
from mpi4py import MPI


class PistonTheory:
    def __init__(self, panel_chord, panel_width, N_el, comm=MPI.COMM_WORLD):
        self.panel_chord = panel_chord
        self.panel_width = panel_width
        self.N_el = N_el

        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()

        # create_mesh
        x = np.linspace(0, self.panel_chord, num=self.N_el + 1)
        y = np.zeros_like(x)
        z = np.zeros_like(x)

        # partitioning
        if self.rank == 0:
            self.owned = np.arange(self.N_el + 1)
            ave, res = divmod(self.N_el + 1, self.nprocs)
            counts = [ave + 1 if p < res else ave for p in range(self.nprocs)]

            starts = [sum(counts[:p]) for p in range(self.nprocs)]
            ends = [sum(counts[: p + 1]) for p in range(self.nprocs)]

            self.owned = [self.owned[starts[p] : ends[p]] for p in range(self.nprocs)]

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

        self.n_dof = 3

    def compute_pressure(self):
        x = self.comm.gather(self.xyz[0::3], root=0)
        z = self.comm.gather(self.xyz[2::3], root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)
            z = np.concatenate(z, axis=0)

            pressure = np.zeros(len(x) - 1)
            for i in range(len(x) - 1):
                # compute panel slope
                dzdx = (z[i + 1] - z[i]) / (x[i + 1] - x[i])

                # compute panel pressure with steady piston theory
                pressure[i] = (
                    4
                    * self.qdyn
                    / np.sqrt(self.mach**2 - 1)
                    * (self.aoa * np.pi / 180 - dzdx)
                )

        else:
            pressure = np.zeros(0)

        return pressure

    def compute_residual(self, pressure):
        x = self.comm.gather(self.xyz[0::3], root=0)
        z = self.comm.gather(self.xyz[2::3], root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)
            z = np.concatenate(z, axis=0)

            residual = np.zeros(len(x) - 1)
            for i in range(len(x) - 1):
                # compute panel slope
                dzdx = (z[i + 1] - z[i]) / (x[i + 1] - x[i])

                # residual
                residual[i] = pressure[i] - 4 * self.qdyn / np.sqrt(
                    self.mach**2 - 1
                ) * (self.aoa * np.pi / 180 - dzdx)

        else:
            residual = np.zeros(0)

        return residual

    def compute_pressure_derivatives(self, adjoint):
        x = self.comm.gather(self.xyz[0::3], root=0)
        z = self.comm.gather(self.xyz[2::3], root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)
            z = np.concatenate(z, axis=0)

            # allocate output
            d_x = np.zeros_like(x)
            d_z = np.zeros_like(z)
            d_aoa = 0.0
            d_qdyn = 0.0
            d_mach = 0.0

            for i in range(len(x) - 1):
                # compute panel slope
                dzdx = (z[i + 1] - z[i]) / (x[i + 1] - x[i])

                # x derivatives
                d_x[i] += (
                    adjoint[i]
                    * 4
                    * self.qdyn
                    / np.sqrt(self.mach**2 - 1)
                    * dzdx
                    / (x[i + 1] - x[i])
                )
                d_x[i + 1] -= (
                    adjoint[i]
                    * 4
                    * self.qdyn
                    / np.sqrt(self.mach**2 - 1)
                    * dzdx
                    / (x[i + 1] - x[i])
                )

                # z derivatives
                d_z[i] -= (
                    adjoint[i]
                    * 4
                    * self.qdyn
                    / np.sqrt(self.mach**2 - 1)
                    / (x[i + 1] - x[i])
                )
                d_z[i + 1] += (
                    adjoint[i]
                    * 4
                    * self.qdyn
                    / np.sqrt(self.mach**2 - 1)
                    / (x[i + 1] - x[i])
                )

                # aoa derivatives
                d_aoa += (
                    adjoint[i]
                    * -4
                    * self.qdyn
                    / np.sqrt(self.mach**2 - 1)
                    * np.pi
                    / 180
                )

                # qdyn derivatives
                d_qdyn += (
                    adjoint[i]
                    * -4
                    / np.sqrt(self.mach**2 - 1)
                    * (self.aoa * np.pi / 180 - dzdx)
                )

                # mach derivatives
                d_mach += (
                    adjoint[i]
                    * 4
                    * self.qdyn
                    * self.mach
                    * (self.aoa * np.pi / 180 - dzdx)
                    / (self.mach**2 - 1) ** (3 / 2)
                )

        else:
            d_x = None
            d_z = None
            d_aoa = None
            d_qdyn = None
            d_mach = None

        # distribute output
        d_x = self._distribute_output(d_x)
        d_z = self._distribute_output(d_z)
        d_aoa = self.comm.bcast(d_aoa, root=0)
        d_qdyn = self.comm.bcast(d_qdyn, root=0)
        d_mach = self.comm.bcast(d_mach, root=0)

        # pad outputs into correct dof
        d_xa = self._pad_output(
            size=3 * self.n_nodes, n1=0, n2=3, x=d_x
        ) + self._pad_output(size=3 * self.n_nodes, n1=2, n2=3, x=d_z)

        return d_xa, d_aoa, d_qdyn, d_mach

    def compute_force(self):
        x = self.comm.gather(self.xyz[0::3], root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)

            f = np.zeros_like(x)
            for i in range(len(x) - 1):
                # distribute force to the two end nodes
                f[i : i + 2] += (
                    self.pressure[i] * self.panel_width * (x[i + 1] - x[i]) / 2
                )

        else:
            f = None

        # distribute output
        f = self._distribute_output(f)

        # pad outputs into dof 2
        f = self._pad_output(size=self.n_dof * self.n_nodes, n1=2, n2=3, x=f)

        return f

    def compute_force_derivatives(self, adjoint):
        x = self.comm.gather(self.xyz[0::3], root=0)

        # extract the adjoints needed: dof 2
        adjoint = self.comm.gather(adjoint[2::3], root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)
            adjoint = np.concatenate(adjoint, axis=0)

            # allocate output
            d_x = np.zeros_like(x)
            d_p = np.zeros_like(self.pressure)

            for i in range(len(x) - 1):
                # x derivatives
                d_x[i] -= self.pressure[i] * self.panel_width / 2 * adjoint[i]
                d_x[i] -= self.pressure[i] * self.panel_width / 2 * adjoint[i + 1]
                d_x[i + 1] += self.pressure[i] * self.panel_width / 2 * adjoint[i]
                d_x[i + 1] += self.pressure[i] * self.panel_width / 2 * adjoint[i + 1]

                # pressure derivatives
                d_p[i] += self.panel_width * (x[i + 1] - x[i]) / 2 * adjoint[i]
                d_p[i] += self.panel_width * (x[i + 1] - x[i]) / 2 * adjoint[i + 1]

        else:
            d_x = None
            d_p = np.zeros(0)

        # distribute output
        d_x = self._distribute_output(d_x)

        # pad outputs into correct dof
        d_xa = self._pad_output(size=3 * self.n_nodes, n1=0, n2=3, x=d_x)

        return d_xa, d_p

    def compute_lift(self):
        C_L = self.comm.bcast(np.mean(self.pressure) / self.qdyn, root=0)
        return C_L

    def compute_lift_derivatives(self, adjoint):
        d_p = np.ones_like(self.pressure) / self.qdyn / len(self.pressure) * adjoint

        if self.rank == 0:
            d_qdyn = -np.mean(self.pressure) / self.qdyn / self.qdyn * adjoint
        else:
            d_qdyn = None
        d_qdyn = self.comm.bcast(d_qdyn, root=0)

        return d_p, d_qdyn

    def _distribute_output(self, x):
        x = self.comm.bcast(x, root=0)
        if self.owned is not None:
            x = x[self.owned]
        return x

    def _pad_output(self, size, n1, n2, x):
        if self.owned is not None:
            X = np.zeros(size)
            X[n1::n2] = x
        else:
            X = np.zeros(0)
        return X

    def write_output(self):
        x = self.comm.gather(self.xyz[0::3], root=0)
        z = self.comm.gather(self.xyz[2::3], root=0)

        # only run on rank 0
        if self.rank == 0:
            x = np.concatenate(x, axis=0)
            z = np.concatenate(z, axis=0)

            x = np.c_[x[0:-1], x[1:]].flatten() / self.panel_chord
            z = np.c_[z[0:-1], z[1:]].flatten() / self.panel_chord
            pressure = np.c_[self.pressure, self.pressure].flatten() / self.qdyn

            f = open("aerodynamics_output.dat", "w+")
            f.write('TITLE = "piston theory data"\n')
            f.write('VARIABLES = "x", "z", "pressure"\n')
            f.write("ZONE I=" + str(len(x)) + ", F=POINT\n")
            for i in range(len(x)):
                f.write(str(x[i]) + " " + str(z[i]) + " " + str(pressure[i]) + "\n")
            f.close()
