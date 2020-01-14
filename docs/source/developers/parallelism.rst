*****************************
Parallelism and Communicators
*****************************
Coupling with high-fidelity analysis components requires communication of parallel data in OpenMDAO.
This can create difficulty coupling when mesh sizes or internal solver differences make the solvers more efficient with an unequal number of processors.
To make this problem less extensive, all components in OMFSI should expect to be called on every MPI rank.
If the component is desired to operate on a subset of those processors, it is should specify that on the extra ranks the output exists but is of size zero on that rank.
Serial codes can operate on a single rank or be duplicated on all ranks.
