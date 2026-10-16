.. _remote_components:

=================
Remote Components
=================

The purpose of remote components is to provide a means of adding a remote physics analysis to a local OpenMDAO problem.
One situation in which this may be desirable is when the time to carry out a full optimization exceeds an HPC job time limit.
Such a situation, without remote components, may normally require manual restarts of the optimization, and would thus limit one to optimizers with this capability.
Using remote components, one can keep a serial OpenMDAO optimization running continuously on a login node (e.g., using the nohup or screen Linux commands) while the parallel physics analyses are evaluated across several HPC jobs.
Another situation where these components may be advantageous is when the OpenMDAO problem contains components not streamlined for massively parallel environments, or if only a subset of the analysis requires GPU HPC nodes.

In general, remote components use nested OpenMDAO problems in a server-client arrangement.
The outer, client-side OpenMDAO model serves as the overarching analysis/optimization problem while the inner, server-side model serves as the isolated high-fidelity analysis.
The server inside the HPC job remains open to evaluate function or gradient calls.
Wall times for function and gradient calls are saved, and when the maximum previous time multiplied by a scale factor exceeds the remaining job time, the server will be relaunched (see :ref:`remote_server_lifetime`).

Three general base classes are used to achieve this.

* :class:`~mphys.network.remote_component.RemoteComp`: Explicit component that wraps communication with server, replicating inputs/outputs to/from server-side group and requesting new a server when estimated analysis time exceeds remaining job time.
* :class:`~mphys.network.server_manager.ServerManager`: Used by ``RemoteComp`` to control and communicate with the server.
* :class:`~mphys.network.server.Server`: Loads the inner OpenMDAO problem and evaluates function or gradient calls as requested by the ``ServerManager``.

Currently, there is one derived class for each, which use pbs4py for HPC job control and ZeroMQ for network communication.

* :class:`~mphys.network.zmq_pbs.RemoteZeroMQComp`: Through the use of ``MPhysZeroMQServerManager``, uses encoded JSON dictionaries to send and receive necessary information to and from the server.
* :class:`~mphys.network.zmq_pbs.MPhysZeroMQServerManager`: Uses ZeroMQ socket and ssh port forwarding from login to compute node to communicate with server, and pbs4py to start, stop, and check status of HPC jobs.
* :class:`~mphys.network.zmq_pbs.MPhysZeroMQServer`: Uses ZeroMQ socket to send and receive encoded JSON dictionaries.

------------------------
RemoteZeroMQComp Options
------------------------
.. embed-options::
    mphys.network.zmq_pbs
    RemoteZeroMQComp
    options

----------------------------
MPhysZeroMQServer Parameters
----------------------------
The server is configured through its constructor arguments rather than OpenMDAO options:

* :code:`port`: TCP port to listen on; passed to the server script by the client as :code:`--port`.
* :code:`get_om_group_function_pointer`: function (or functor) returning the OpenMDAO Group to evaluate.
* :code:`ignore_setup_warnings`, :code:`ignore_runtime_warnings`: suppress OpenMDAO warnings during :code:`setup` and :code:`run_model`/:code:`compute_totals`, respectively (default False).
* :code:`rerun_initial_design`: evaluate the baseline design before any new ones, for cases where the analysis somehow depends on the starting design (default False).
* :code:`write_n2`: write :code:`n2_inner_analysis_<component name>.html` after each evaluation (default False).
* :code:`report_errors`: send the traceback of any server-side error back to the client, which then stops the job and raises the error (default True).

-----
Usage
-----
When adding a :code:`RemoteZeroMQComp` component, the two required options are :code:`run_server_filename`, which is the server to be launched on an HPC job, and :code:`pbs`, which is the pbs4py Launcher object.
The server file should accept port number as an argument to facilitate communication with the client; :code:`get_default_zmq_pbs_argparser` provides an argument parser with this option, to which further options may be added and passed from the client with :code:`additional_server_args`.
Within this file, the :code:`MPhysZeroMQServer` class's :code:`get_om_group_function_pointer` option is the pointer to the OpenMDAO group to be evaluated.
By default, any design variables, objectives, and constraints defined in the group will be added on the client side, including their bounds, scaling, and units.
Setting :code:`skip_objective_constraint_definition` adds the objective and constraint quantities as outputs only, so that the objective and constraints can instead be defined at the top level of the client model (e.g., when an objective combines the outputs of several remote components).
Any other desired inputs or outputs must be added in the :code:`additional_remote_inputs` or :code:`additional_remote_outputs` options; derivatives of all responses and additional outputs are computed with respect to the design variables and additional inputs.
Inputs whose derivatives are not needed, e.g. operating conditions that remain fixed during an optimization, should be added as :code:`additional_remote_constants` instead.
On the client side, any "." characters in these input and output names will be replaced by :code:`var_naming_dot_replacement`.
The :code:`run_directory` option makes the component run (and write its files) in a different directory.

The remote component may be added to a model that runs on several MPI ranks, e.g., as one of several remote components in an OpenMDAO ParallelGroup, as in the parallel example below.
Only the component's root rank launches a server and communicates with it; the results are broadcast to the component's other ranks.
When each component in such a parallel group defines its own objective and constraints, :code:`use_derivative_coloring` assigns a :code:`parallel_deriv_color` to each of them such that the Nth response of every component shares a color, allowing OpenMDAO to compute the components' derivatives simultaneously.

The screen output from a particular remote component's Nth server will be sent to :code:`mphys_<component name>_serverN.out`, where :code:`component name` is the subsystem name of the :code:`RemoteZeroMQComp` instance.
Searching for the keyword "SERVER" will display what the server is currently doing; the keyword "CLIENT" will do the same on the client-side.
The HPC job for the component's server is named :code:`MPhys<port number>`; the pbs4py-generated job submission script is the same followed by ".pbs".

.. _remote_server_lifetime:

---------------
Server Lifetime
---------------
A server is launched when the client-side component is first set up, and its baseline design is evaluated to obtain the design problem information (design variables, responses, additional inputs/outputs and their metadata).
Since the PBS job may report as running well before the server has loaded its model and bound its port (e.g., depending on MPI startup time), the client pings the server every :code:`startup_ping_interval` seconds until it replies.
By default it waits indefinitely (with periodic checks that the job is still running); :code:`startup_timeout` limits the wait.

Before each function or gradient evaluation, the client estimates the evaluation time as :code:`time_estimate_multiplier` times the longest previous evaluation of that type, plus :code:`time_estimate_buffer` seconds.
If this exceeds the job's remaining wall time, or the job is no longer running, the server is first stopped and replaced with a new one.
With :code:`reboot_only_on_function_call` set to true (the default), the restart-check is skipped before gradient calls; before function calls, the estimated time includes a potential gradient evaluation following the function call (i.e., the previous estimate plus the maximum gradient wall time times the multiplier).
This may shorten the current job, but avoids a situation where a newly launched server needs to reevaluate the same design before gradients can be computed.
Each relaunch caused by a job that has ended unexpectedly (whether detected before an evaluation or while waiting for its result) counts toward :code:`job_expiration_max_restarts`; once exceeded, an error is raised.
If significant idle time between evaluations is expected, which would typically waste compute hours on an HPC for example, :code:`stop_server_for_down_time` stops the job after every evaluation if set to 1, or only after gradient evaluations if set to 2.

Servers are stopped (shutdown message, ssh tunnel killed, job deleted) when :code:`stop_server()` is called, when the client process exits normally, or when the client receives SIGTERM or SIGHUP.
On Linux, the ssh tunnel process is also set up to die with the client even if the client is killed abruptly.
However, in certain cases the HPC job can persist depending on how the client process is killed and should in this case be deleted manually.

HPC and network problems
========================
The client tolerates temporary problems with the PBS server and the network:

* :code:`qsub`, :code:`qstat` and :code:`qdel` are given a timeout and retried (by default 10 attempts a minute apart for submission and status queries, 3 for deletion) if PBS cannot be reached. Submission errors that cannot resolve themselves (e.g., an invalid queue) are raised immediately with qsub's message. If the job state cannot be determined after all retries, the server is assumed to be still running.
* While waiting for a reply, the client checks every :code:`receive_timeout` seconds that the job is still running and that the ssh tunnel is alive. A dead tunnel is restarted and the request re-sent. A job that has ended (confirmed by several consecutive checks) is relaunched and the request re-sent, counting towards :code:`job_expiration_max_restarts`. A server that is simply busy with a long analysis is left alone; the request is never duplicated.
* :code:`forward_through_frontend` routes the ssh tunnel through the PBS front-end node (:code:`$PBS_O_HOST`) for systems whose compute nodes cannot reach each other directly, e.g. when the client itself runs inside a PBS job.

-------
Example
-------
Two examples are provided for the `supersonic panel aerostructural case <https://github.com/OpenMDAO/mphys/tree/main/examples/aerostructural/supersonic_panel>`_: :code:`as_opt_remote_serial.py` and :code:`as_opt_remote_parallel.py`.
Both run the optimization problem defined in :code:`as_opt_parallel.py`, which contains a :code:`MultipointParallel` class and thus evaluates two aerostructural scenarios in parallel.
The serial remote example runs this group on one server.
The parallel remote example, on the other hand, contains an OpenMDAO parallel group which runs two servers in parallel.
Both examples use the same server file, :code:`mphys_server.py`, but point to either :code:`as_opt_parallel.py` or :code:`run.py` by sending the model's filename through the use of the :code:`RemoteZeroMQComp`'s :code:`additional_server_args` option.
As demonstrated in this server file, additional configuration options may be sent to the server-side OpenMDAO group through the use of a functor (called :code:`GetModel` in this case) in combination with :code:`additional_server_args`.
In this particular case, scenario name(s) are sent as :code:`additional_server_args` from the client side; on the server side, the :code:`GetModel` functor allows the scenario name(s) to be sent as OpenMDAO options to the server-side group.
Using the scenario :code:`run_directory` option, the scenarios can then be evaluated in different directories.
The remote component(s) use a pbs4py Launcher object for either the NASA K cluster (:code:`PBS.k4`) or the NAS Pleiades cluster (:code:`PBS.nas`), chosen from the login node's hostname.

---------------
Troubleshooting
---------------
The :code:`dump_json` option for :code:`RemoteZeroMQComp` will make the component write input and output JSON files, which contain all data sent to and received from the server.
An exception is the :code:`wall_time` entry (given in seconds) in the output JSON file, which is added on the client-side after the server has completed the design evaluation.
Similarly, the :code:`down_time` entry keeps track of the elapsed time between the end of the previous design evaluation and the beginning of the current one.
Another entry that is only provided for informational purposes is :code:`design_counter`, which keeps track of how many different designs have been evaluated on the current server.
By default the files :code:`<component name>_inputs.json` and :code:`<component name>_outputs.json` are overwritten on every evaluation.
If :code:`dump_separate_json` is set to True, then separate files will be written for each design evaluation, in a :code:`remote_json_files` directory, as :code:`<component name>_<inputs|outputs>_<function|derivative><N>.json`.

The :code:`reuse_dumped_json` option (which internally turns the :code:`dump_separate_json` option on) makes the component look through these output files before contacting the server.
In the case of server initialization, the file :code:`<component name>_outputs_function0.json`, if it exists, will be used to set all inputs and outputs on the client side, without having to run an analysis on the server side.
For a normal function or gradient evaluation, a JSON output file corresponding to the current design (i.e., with the same design variables and additional remote inputs/constants), if it exists, will provide the outputs.
This allows for various troubleshooting on the client-side without having to re-run costly analyses on the server-side.
Alternatively, this also allows an interrupted optimization to be re-run up to the point where it stopped, without re-evaluating designs on the HPC; a server is only launched once a new design is encountered.

On the server side, an n2 file titled :code:`n2_inner_analysis_<component name>.html` will be written after each evaluation if the server's :code:`write_n2` argument is set.
Each status message printed by the client is prefixed with :code:`CLIENT (subsystem <component name>)`, so that the messages of several remote components can be told apart; server messages in :code:`mphys_<component name>_serverN.out` are prefixed with :code:`SERVER`.

Current Limitations
===================
* A pbs4py Launcher must be implemented for your HPC environment.
* The HPC environment must allow ssh port forwarding from the login node to a compute node.
* Servers are stopped automatically when the client exits, including on SIGTERM/SIGHUP, but not if the client is killed with SIGKILL or the login node fails. The ssh tunnel then still dies with the client (on Linux), but the HPC job keeps running until its wall time expires and should be deleted by hand; its name is :code:`MPhys` followed by the port number, and if runs are launched from multiple login nodes then one may have multiple jobs with the same name.
* Currently, the :code:`of` option (as well as :code:`wrt`) for :code:`check_totals` or :code:`compute_totals` is not used by the remote component; on the server side, :code:`compute_totals` will be evaluated for all responses (objectives, constraints, and :code:`additional_remote_outputs`). Depending on how many :code:`of` responses are desired, this may be more costly than not using remote components.

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
