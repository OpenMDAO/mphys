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

RemoteZeroMQComp Options
========================
.. embed-options::
    mphys.network.zmq_pbs
    RemoteZeroMQComp
    options

Usage
=====
When adding a :code:`RemoteZeroMQComp` component, the two required options are :code:`run_server_filename`, which is the server to be launched on an HPC job, and :code:`pbs`, which is the pbs4py Launcher object.
The server file should accept port number as an argument to facilitate communication with the client.
Within this file, the :code:`MPhysZeroMQServer` class's :code:`get_om_group_function_pointer` option is the pointer to the OpenMDAO Group or Multipoint class to be evaluated.
By default, any design variables, objectives, and constraints defined in the group will be added on the client side.
Any other desired inputs or outputs must be added in the :code:`additional_remote_inputs` or :code:`additional_remote_outputs` options.
On the client side, any "." characters in these input and output names will be replaced by :code:`var_naming_dot_replacement`.

Troubleshooting
===============
The :code:`dump_json` option for :code:`RemoteZeroMQComp` will make the component write input and output JSON files, which contain all data sent to and received from the server.
The one exception is the :code:`wall_time` entry (given in seconds) in the output JSON file, which is added on the client-side.
Another entry that is only provided for informational purposes is :code:`design_counter`, which keeps track of how many different designs have been evaluated on the current server.
If :code:`dump_separate_json` is also set to True, then separate files will be written for each design evaluation.
On the server side, an n2 file titled :code:`n2_inner_analysis.html` will be written after each evaluation.

Current Limitations
===================
* A pbs4py Launcher must be implemented for your HPC environment
* On the client side, :code:`RemoteZeroMQComp.stop_server()` should be added after your analysis/optimization; the HPC job and/or ssh port forwarding may otherwise need to be halted manually.
* Currently, the :code:`of` and :code:`wrt` inputs for :code:`check_totals` are not used by the remote component; on the server side, :code:`compute_totals` will be evaluated for all design variables and responses.

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
