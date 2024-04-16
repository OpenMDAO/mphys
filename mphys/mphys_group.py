from openmdao.api import Group


class MphysGroup(Group):
    """
    A OpenMDAO class for automated promotion of MPhys variables
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mphys_subsystems = []

    def mphys_add_subsystem(self, name, subsystem):
        """
        Adding an mphys subsystem will add the subsystem and then set the group
        to automatically promote the mphys tagged variables
        """
        subsystem = self.add_subsystem(name, subsystem)
        self.mphys_subsystems.append(subsystem)
        return subsystem

    def configure(self):
        """
        Promote the mphys-tagged variables of subsystems added by :func:`~MphysGroup.mphys_add_subsystem`
        """
        self._mphys_promote_coupling_variables()
        self._mphys_promote_inputs()
        self._mphys_promote_mesh_coordinates()
        self._mphys_promote_results()
        self._mphys_promote_time_derivatives()

    def _mphys_promote_by_tag(self, iotype, tag):
        for subsystem in self.mphys_subsystems:
            promoted = []
            tagged_variables = subsystem.get_io_metadata(iotypes=iotype,
                                                         metadata_keys=['tags'],
                                                         tags=tag)
            for val in tagged_variables.values():
                variable = val['prom_name']
                if variable not in promoted:
                    self.promotes(subsystem.name, any=[variable])
                    promoted.append(variable)

    def _mphys_promote_time_derivatives(self):
        self._mphys_promote_by_tag(['input', 'output'], 'mphys_time_derivative')

    def _mphys_promote_coupling_variables(self):
        self._mphys_promote_by_tag(['input', 'output'], 'mphys_coupling')

    def _mphys_promote_inputs(self):
        self._mphys_promote_by_tag('input', 'mphys_input')

    def _mphys_promote_mesh_coordinates(self):
        self._mphys_promote_by_tag(['input', 'output'], 'mphys_coordinates')

    def _mphys_promote_results(self):
        self._mphys_promote_by_tag('output', 'mphys_result')
