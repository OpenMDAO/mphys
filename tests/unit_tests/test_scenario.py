import unittest
from typing import List
import openmdao.api as om

from mphys.scenario import Scenario
from mphys import Builder

from common_methods import CommonMethods

class PreComp(om.IndepVarComp):
    def setup(self):
        self.add_output('out_pre', val=1.0, tags=['mphys_coupling'])

class PostComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('out_pre', tags=['mphys_coupling'])
        self.add_output('out_post', val=1.0, tags=['mphys_coupling'])

class UserPostCompPromoteInputsAndOutputs(om.ExplicitComponent):
    def setup(self):
        self.add_input('out_post')
        self.add_output('out_user', val=1.0)

class UserPostCompMphysPromote(om.ExplicitComponent):
    def setup(self):
        self.add_input('out_post', tags=['mphys_coupling'])
        self.add_output('out_user2', val=2.0, tags=['mphys_result'])

class UserPostCompUseGeneralPromotes(om.ExplicitComponent):
    def setup(self):
        self.add_input('out_post')
        self.add_output('out_user3', val=3.0)

class BuilderA(Builder):
    def __init__(self):
        self.name = 'BuilderA'
    def get_pre_coupling_subsystem(self, scenario_name=None):
        return PreComp()

class BuilderB(Builder):
    def __init__(self):
        self.name = 'BuilderB'
    def get_post_coupling_subsystem(self, scenario_name=None):
        return PostComp()

class FakeScenario(Scenario):
    def initialize(self):
        self.options.declare('builders')
        super().initialize()

    def _mphys_scenario_setup(self):
        for builder in self.options['builders']:
            self._mphys_add_pre_coupling_subsystem_from_builder(builder.name, builder, self.name)

        for builder in self.options['builders']:
            self._mphys_add_post_coupling_subsystem_from_builder(builder.name, builder, self.name)



class TestScenario(unittest.TestCase):
    def setUp(self) -> None:
        builders: List[Builder] = [BuilderA(), BuilderB()]

        scenario = FakeScenario(builders=builders)
        scenario.mphys_add_post_subsystem('user_post', UserPostCompPromoteInputsAndOutputs(), promotes_inputs=['out_post'], promotes_outputs=['out_user'])
        scenario.mphys_add_post_subsystem('user_post2', UserPostCompMphysPromote())
        scenario.mphys_add_post_subsystem('user_post3', UserPostCompUseGeneralPromotes(), promotes=['*'])

        self.common = CommonMethods()

        self.prob = om.Problem()
        self.prob.model.add_subsystem('scenario',scenario)
        self.prob.setup()

    def test_run_model(self):
        self.common.test_run_model(self)

    def test_components_were_added(self):
        self.assertIsInstance(self.prob.model.scenario.BuilderA_pre, PreComp)
        self.assertIsInstance(self.prob.model.scenario.BuilderB_post, PostComp)
        self.assertIsInstance(self.prob.model.scenario.user_post, UserPostCompPromoteInputsAndOutputs)
        self.assertIsInstance(self.prob.model.scenario.user_post2, UserPostCompMphysPromote)
        self.assertIsInstance(self.prob.model.scenario.user_post3, UserPostCompUseGeneralPromotes)

    def test_scenario_subsystem_order(self):
        expected_order = ['BuilderA_pre', 'BuilderB_post', 'user_post','user_post2', 'user_post3']
        self.common.test_subsystem_order(self, self.prob.model.scenario, expected_order)

    def test_no_autoivcs(self):
        self.common.test_no_autoivcs(self)

    def test_promoted_post_user_subsystem_outputs(self):
        self.assertAlmostEqual(self.prob.get_val('scenario.out_user'), 1.0)
        self.assertAlmostEqual(self.prob.get_val('scenario.out_user2'), 2.0)
        self.assertAlmostEqual(self.prob.get_val('scenario.out_user3'), 3.0)


if __name__ == '__main__':
    unittest.main()
