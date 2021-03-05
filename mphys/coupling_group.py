from .mphys_group import MphysGroup

class CouplingGroup(MphysGroup):
    def _mphys_variable_is_in_subsystem_inputs(self, element, variable_name):
        meta_data = element.get_io_metadata(iotypes='input', includes=variable_name)
        return bool(meta_data)
