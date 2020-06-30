import openmdao.api as om
from pprint import pprint 
class C1(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        self.add_input('x4', shape=0)
        self.add_input('step', val= 1)
        self.add_output('x1', shape=0)

    def compute(self, inputs, outputs):

        outputs['x1'] = inputs['x4'] + inputs['step']


class C2(om.ExplicitComponent):
    def setup(self):
        self.add_input('x1', val=0.0)

        self.add_output('x2', val=0.0)

    def compute(self, inputs, outputs):
        outputs['x2'] = inputs['x1'] 



class C3(om.ExplicitComponent):
    def setup(self):
        self.add_input('x3', val=0.0)

        self.add_output('x4', val=0.0)

    def compute(self, inputs, outputs):
        outputs['x4'] = inputs['x3'] 


class C4(om.ExplicitComponent):
    def setup(self):
        self.add_input('x2', val=0.0)

        self.add_output('x3', val=0.0)

    def compute(self, inputs, outputs):
        outputs['x3'] = inputs['x2'] + 1 



class Cycle(om.Group):
    def configure(self):
        print('hi')
        self._setup_var_data()

        for sys in self.system_iter():
            if isinstance(sys, C1):

                # look at the inputs
                for var in sys._var_rel_names['input']:
                    # if the var shape shape zero



                    if sys._var_rel2meta[var]['size'] == 0:
                        print(sys.name, var)
                        var_abs = self._var_allprocs_prom2abs_list['output'][var]
                        var_meta = self._var_abs2meta[var_abs[0]]

                        # set the size using meta info
                        for data in ['value', 'shape', 'size']:
                            sys._var_rel2meta[var][data] = var_meta[data]


                    # set the shape to the shape of the other promoted var


                # look at the outputs
                for var in sys._var_rel_names['output']:

                    if sys._var_rel2meta[var]['size'] == 0:
                        print(sys.name, var)
                        var_abs = self._var_allprocs_prom2abs_list['input'][var]
                        var_meta = self._var_abs2meta[var_abs[0]]

                        # set the size using meta info
                        for data in ['value', 'shape', 'size']:
                            sys._var_rel2meta[var][data] = var_meta[data]



                # import ipdb; ipdb.set_trace()

                # look for the variable else where in the system 

                # for var in sys._var_rel_names:






model = om.Group()
ivc = om.IndepVarComp()
ivc.add_output('x1', 3.0)
model.add_subsystem('des_vars', ivc)
cycle = model.add_subsystem('cycle', Cycle())
cycle.add_subsystem('C1', C1(), promotes = ['*'])
cycle.add_subsystem('C2', C2(), promotes = ['*'])
cycle.add_subsystem('C3', C3(), promotes = ['*'])
cycle.add_subsystem('C4', C4(), promotes = ['*'])


model.connect('des_vars.x1', 'cycle.step')
# model.set_order(['cycle.C1', 'des_vars',  'cycle.C2', 'cycle.C3', 'cycle.C4'])

# Nonlinear Block Gauss-Seidel is a gradient-free solver
cycle.nonlinear_solver = om.NonlinearBlockGS()



prob = om.Problem(model)
prob.setup()
om.n2(prob, show_browser=False, outfile='test.html')
prob.run_model()