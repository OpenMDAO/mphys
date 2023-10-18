.. _remote_components:

*****************
Remote Components
*****************

The purpose of remote components is to provide a means of adding a remote physics analysis to a local OpenMDAO problem.
One situation in which this may be desirable is when the time to carry out a full optimization exceeds an HPC job time limit.
Such a situation, without remote components, may normally require manual restarts of the optimization, and would thus limit one to optimizers with this capability.
Using remote components, one can keep a serial OpenMDAO optimization running continuously on a login node (e.g., using the nohup or screen Linux commands) while the parallel physics analyses are evaluated across several HPC jobs.
Another situation where these components may be advantageous is when the OpenMDAO problem contains components not streamlined for massively parallel environments.

In general, remote components use nested OpenMDAO problems in a server-client arrangement.
The outer, client-side OpenMDAO model serves as the overarching analysis/optimization problem while the inner, server-side model serves as the isolated high-fidelity analysis.
The server inside the HPC job remains open to evaluate function or gradient calls.
Wall times for function and gradient calls are saved, and when the maximum previous time multiplied by a scale factor exceeds the remaining job time, the server will be relaunched.

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

The screen output from a particular remote component's Nth server will be sent to :code:`mphys_<component name>_serverN.out`, where :code:`component name` is the subsystem name of the :code:`RemoteZeroMQComp` instance.
Searching for the keyword "SERVER" will display what the server is currently doing; the keyword "CLIENT" will do the same on the client-side.
The HPC job for the component's server is named :code:`MPhys<port number>`; the pbs4py-generated job submission script is the same followed by ".pbs".
Note that running the remote component in parallel is not supported, and a SystemError will be triggered otherwise.

Example
=======
Two examples are provided for the `supersonic panel aerostructural case <https://github.com/OpenMDAO/mphys/tree/main/examples/aerostructural/supersonic_panel>`_: :code:`as_opt_remote_serial.py` and :code:`as_opt_remote_parallel.py`.
Both run the optimization problem defined in :code:`as_opt_parallel.py`, which contains a :code:`MultipointParallel` class and thus evaluates two aerostructural scenarios in parallel.
The serial remote example runs this group on one server.
The parallel remote example, on the other hand, contains an OpenMDAO parallel group which runs two servers in parallel.
Both examples use the same server file, :code:`mphys_server.py`, but point to either :code:`as_opt_parallel.py` or :code:`run.py` by sending the model's filename through the use of the :code:`RemoteZeroMQComp`'s :code:`additional_server_args` option.
As demonstrated in this server file, additional configuration options may be sent to the server-side OpenMDAO group through the use of a functor (called :code:`GetModel` in this case) in combination with :code:`additional_server_args`.
In this particular case, scenario name(s) are sent as :code:`additional_server_args` from the client side; on the server side, the :code:`GetModel` functor allows the scenario name(s) to be sent as OpenMDAO options to the server-side group.
Using the scenario :code:`run_directory` option, the scenarios can then be evaluated in different directories.
In both examples, the remote component(s) use a :code:`K4` pbs4py Launcher object, which will launch, monitor, and stop jobs using the K4 queue of the NASA K-cluster.

Troubleshooting
===============
The :code:`dump_json` option for :code:`RemoteZeroMQComp` will make the component write input and output JSON files, which contain all data sent to and received from the server.
An exception is the :code:`wall_time` entry (given in seconds) in the output JSON file, which is added on the client-side after the server has completed the design evaluation.
Another entry that is only provided for informational purposes is :code:`design_counter`, which keeps track of how many different designs have been evaluated on the current server.
If :code:`dump_separate_json` is set to True, then separate files will be written for each design evaluation.
On the server side, an n2 file titled :code:`n2_inner_analysis_<component name>.html` will be written after each evaluation.

Current Limitations
===================
* A pbs4py Launcher must be implemented for your HPC environment
* On the client side, :code:`RemoteZeroMQComp.stop_server()` should be added after your analysis/optimization to stop the HPC job and ssh port forwarding, which the server manager starts as a background process.
* If :code:`stop_server` is not called or the server stops unexpectedly, stopping the port forwarding manually is difficult, as it involves finding the ssh process associated with the remote server's port number. This must be done on the same login node that the server was launched from.
* Stopping the HPC job is somewhat easier as the job name will be :code:`MPhys` followed by the port number; however, if runs are launched from multiple login nodes then one may have multiple jobs with the same name.
* Currently, the :code:`of` option (as well as :code:`wrt`) for :code:`check_totals` or :code:`compute_totals` is not used by the remote component; on the server side, :code:`compute_totals` will be evaluated for all responses (objectives, constraints, and :code:`additional_remote_outputs`). Depending on how many :code:`of` responses are desired, this may be more costly than not using remote components.
* The HPC environment must allow ssh port forwarding from the login node to a compute node.

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
