import numpy as np
from mpi4py import MPI


class Xfer:
    def __init__(self, aero, struct, comm=MPI.COMM_WORLD):
        self.aero = aero
        self.struct = struct

        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()

    def _linear_interpolate(self, x, grid):
        if x < np.min(grid):
            left = np.argmin(x)
        else:
            left = np.argwhere(grid <= x)[-1][0]
        if left >= len(grid) - 1:
            left -= 1
        right = left + 1
        eta = (x - grid[left]) / (grid[right] - grid[left])

        return right, left, eta

    def transfer_displacements(self):
        xs = self.comm.gather(self.xs[0::3], root=0)
        xa = self.comm.gather(self.xa[0::3], root=0)

        # extract the displacements needed: dof 2
        us = self.comm.gather(self.us[2::6], root=0)

        # only run on rank 0
        if self.rank == 0:
            xs = np.concatenate(xs, axis=0)
            xa = np.concatenate(xa, axis=0)
            us = np.concatenate(us, axis=0)

            # allocate output
            ua = np.zeros_like(xa)

            for i in range(len(xa)):
                right, left, eta = self._linear_interpolate(xa[i], xs)
                ua[i] = (1 - eta) * us[left] + eta * us[right]

        else:
            ua = None

        # distribute output
        ua = self._distribute_output(self.aero.owned, ua)

        # pad outputs into dof 2
        ua = self._pad_output(
            self.aero.owned, size=self.aero.n_dof * self.aero.n_nodes, n1=2, n2=3, x=ua
        )

        return ua

    def transfer_displacements_derivatives(self, adjoint):
        xs = self.comm.gather(self.xs[0::3], root=0)
        xa = self.comm.gather(self.xa[0::3], root=0)

        # extract the displacements needed: dof 2
        us = self.comm.gather(self.us[2::6], root=0)

        # extract the adjoints needed: dof 2
        adjoint = self.comm.gather(adjoint[2::3], root=0)

        # only run on rank 0
        if self.rank == 0:
            xs = np.concatenate(xs, axis=0)
            xa = np.concatenate(xa, axis=0)
            us = np.concatenate(us, axis=0)
            adjoint = np.concatenate(adjoint, axis=0)

            # allocate output
            d_xs = np.zeros_like(xs)
            d_xa = np.zeros_like(xa)
            d_us = np.zeros_like(us)

            for i in range(len(xa)):
                right, left, eta = self._linear_interpolate(xa[i], xs)

                # xs derivatives
                eta_wrt_xs_left = (
                    1 / (xs[left] - xs[right])
                    + (xa[i] - xs[left]) / (xs[left] - xs[right]) ** 2
                )
                eta_wrt_xs_right = -(xa[i] - xs[left]) / (xs[right] - xs[left]) ** 2

                d_xs[left] += adjoint[i] * (
                    -eta_wrt_xs_left * us[left] + eta_wrt_xs_left * us[right]
                )
                d_xs[right] += adjoint[i] * (
                    -eta_wrt_xs_right * us[left] + eta_wrt_xs_right * us[right]
                )

                # xa derivatives
                eta_wrt_xa = 1 / (xs[right] - xs[left])

                d_xa[i] += adjoint[i] * (
                    -eta_wrt_xa * us[left] + eta_wrt_xa * us[right]
                )

                # us derivatives
                d_us[left] += adjoint[i] * (1 - eta)
                d_us[right] += adjoint[i] * eta

        else:
            d_xs = None
            d_xa = None
            d_us = None

        # distribute output
        d_xs = self._distribute_output(self.struct.owned, d_xs)
        d_xa = self._distribute_output(self.aero.owned, d_xa)
        d_us = self._distribute_output(self.struct.owned, d_us)

        # pad outputs into correct dof
        d_xs = self._pad_output(
            self.struct.owned, size=3 * self.struct.n_nodes, n1=0, n2=3, x=d_xs
        )
        d_xa = self._pad_output(
            self.aero.owned, size=3 * self.aero.n_nodes, n1=0, n2=3, x=d_xa
        )
        d_us = self._pad_output(
            self.struct.owned,
            size=self.struct.n_dof * self.struct.n_nodes,
            n1=2,
            n2=6,
            x=d_us,
        )

        return d_xs, d_xa, d_us

    def transfer_loads(self):
        xs = self.comm.gather(self.xs[0::3], root=0)
        xa = self.comm.gather(self.xa[0::3], root=0)

        # extract the forces needed: dof 2
        fa = self.comm.gather(self.fa[2::3], root=0)

        # only run on rank 0
        if self.rank == 0:
            xs = np.concatenate(xs, axis=0)
            xa = np.concatenate(xa, axis=0)
            fa = np.concatenate(fa, axis=0)

            # allocate output
            fs = np.zeros_like(xs)

            for i in range(len(xa)):
                right, left, eta = self._linear_interpolate(xa[i], xs)
                fs[left] = (1 - eta) * fa[i]
                fs[right] = eta * fa[i]

        else:
            fs = None

        # distribute output
        fs = self._distribute_output(self.struct.owned, fs)

        # pad outputs into dof 2
        fs = self._pad_output(
            self.struct.owned,
            size=self.struct.n_dof * self.struct.n_nodes,
            n1=2,
            n2=6,
            x=fs,
        )

        return fs

    def transfer_loads_derivatives(self, adjoint):
        xs = self.comm.gather(self.xs[0::3], root=0)
        xa = self.comm.gather(self.xa[0::3], root=0)

        # extract the forces needed: dof 2
        fa = self.comm.gather(self.fa[2::3], root=0)

        # extract the adjoints needed: dof 2
        adjoint = self.comm.gather(adjoint[2::6], root=0)

        # only run on rank 0
        if self.rank == 0:
            xs = np.concatenate(xs, axis=0)
            xa = np.concatenate(xa, axis=0)
            fa = np.concatenate(fa, axis=0)
            adjoint = np.concatenate(adjoint, axis=0)

            # allocate output
            d_xs = np.zeros_like(xs)
            d_xa = np.zeros_like(xa)
            d_fa = np.zeros_like(fa)

            for i in range(len(xa)):
                right, left, eta = self._linear_interpolate(xa[i], xs)

                # xs derivatives
                eta_wrt_xs_left = (
                    1 / (xs[left] - xs[right])
                    + (xa[i] - xs[left]) / (xs[left] - xs[right]) ** 2
                )
                eta_wrt_xs_right = -(xa[i] - xs[left]) / (xs[right] - xs[left]) ** 2

                d_xs[left] += adjoint[left] * -eta_wrt_xs_left * fa[i]
                d_xs[right] += adjoint[left] * -eta_wrt_xs_right * fa[i]

                d_xs[left] += adjoint[right] * eta_wrt_xs_left * fa[i]
                d_xs[right] += adjoint[right] * eta_wrt_xs_right * fa[i]

                # xa derivatives
                eta_wrt_xa = 1 / (xs[right] - xs[left])

                d_xa[i] += adjoint[left] * -eta_wrt_xa * fa[i]
                d_xa[i] += adjoint[right] * eta_wrt_xa * fa[i]

                # fa derivatives
                d_fa[i] += adjoint[left] * (1 - eta)
                d_fa[i] += adjoint[right] * eta

        else:
            d_xs = None
            d_xa = None
            d_fa = None

        # distribute output
        d_xs = self._distribute_output(self.struct.owned, d_xs)
        d_xa = self._distribute_output(self.aero.owned, d_xa)
        d_fa = self._distribute_output(self.aero.owned, d_fa)

        # pad outputs into correct dof
        d_xs = self._pad_output(
            self.struct.owned, size=3 * self.struct.n_nodes, n1=0, n2=3, x=d_xs
        )
        d_xa = self._pad_output(
            self.aero.owned, size=3 * self.aero.n_nodes, n1=0, n2=3, x=d_xa
        )
        d_fa = self._pad_output(
            self.aero.owned,
            size=self.aero.n_dof * self.aero.n_nodes,
            n1=2,
            n2=3,
            x=d_fa,
        )

        return d_xs, d_xa, d_fa

    def _distribute_output(self, owned, x):
        x = self.comm.bcast(x, root=0)
        if owned is not None:
            x = x[owned]
        return x

    def _pad_output(self, owned, size, n1, n2, x):
        if owned is not None:
            X = np.zeros(size)
            X[n1::n2] = x
        else:
            X = np.zeros(0)
        return X
