.. _remote_components:

*****************
Remote Components
*****************

The purpose of remote components is to provide a means of adding a remote physics analysis to a local OpenMDAO problem.
One situation in which this may be desirable is when the time to carry out a full optimization exceeds an HPC job time limit.
Such a situation, without remote components, may normally require manual restarts of the optimization, and would thus limit one to optimizers with this capability.
Using remote components, one can keep a serial OpenMDAO optimization running continuously on a login node while the parallel physics analyses are evaluated across several HPC jobs.
Another situation where these components may be advantageous is when the OpenMDAO problem contains components not streamlined for massively parallel environments.

In general, remote components use nested OpenMDAO problems in a server-client arrangement.
The outer, client-side OpenMDAO model serves as the overarching analysis/optimization problem while the inner, server-side model serves as the isolated high-fidelity analysis.
The server inside the HPC job remains open to evaluate function or gradient calls.

Three general base classes are used to achieve this.

* :class:`~mphys.network.remote_component.RemoteComp`: Explicit component that wraps communication with server, replicating inputs/outputs to/from server-side group and requesting new a server when estimated analysis time exceeds remaining job time.
* :class:`~mphys.network.server_manager.ServerManager`: Used by ``RemoteComp`` to control and communicate with the server.
* :class:`~mphys.network.server.Server`: Loads the inner OpenMDAO problem and evaluates function or gradient calls as requested by the ``ServerManager``.

Currently, there is one derived class for each, which use pbs4py for HPC job control and ZeroMQ for network communication.

* :class:`~mphys.network.zmq_pbs.RemoteZeroMQComp`: Through the use of ``MPhysZeroMQServerManager``, uses encoded JSON dictionaries to send and receive necessary information to and from the server.
* :class:`~mphys.network.zmq_pbs.MPhysZeroMQServerManager`: Uses ZeroMQ socket and ssh port forwarding from login to compute node to communicate with server, and pbs4py to start, stop, and check status of HPC jobs.
* :class:`~mphys.network.zmq_pbs.MPhysZeroMQServer`: Uses ZeroMQ socket to send and receive encoded JSON dictionaries.

.. embed-options::
    mphys.network.remote_component
    RemoteComp
    options

.. autoclass:: mphys.network.remote_component.RemoteComp
    :members:

.. autoclass:: mphys.network.server_manager.ServerManager
    :members:

.. autoclass:: mphys.network.server.Server
    :members:

.. autoclass:: mphys.network.zmq_pbs.RemoteZeroMQComp
    :members:

.. autoclass:: mphys.network.zmq_pbs.MPhysZeroMQServerManager
    :members:

.. autoclass:: mphys.network.zmq_pbs.MPhysZeroMQServer
    :members:
