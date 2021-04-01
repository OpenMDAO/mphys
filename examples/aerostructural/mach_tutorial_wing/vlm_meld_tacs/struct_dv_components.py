import  openmdao.api as om

# Number of components by type
struct_comps = {'ribs':18,
                'le_spar':18,
                'te_spar':18,
                'up_skin':162,
                'lo_skin':162,
                'up_stringer':(8,18),
                'lo_stringer':(8,18)}

class StructDvMapper(om.ExplicitComponent):
    def initialize(self):
        self.ndvs = 810

        var_map = []
        fh = open('component_list.txt')
        while True:
            line = fh.readline()
            if not line:
                break
            index = int(line.split()[0]) - 1
            descript = line.split()[1]

            if 'RIBS' in descript:
                rib = int(descript.split('.')[1][:2]) - 1
                var_map.append(('ribs',rib))

            elif 'SPAR.00' in descript:
                seg = int(descript.split('.')[-1])
                var_map.append(('le_spar',seg))

            elif 'SPAR.09' in descript:
                seg = int(descript.split('.')[-1])
                var_map.append(('te_spar',seg))

            elif 'U_SKIN' in descript:
                patch = int(descript.split('.')[1][:3]) - 1
                var_map.append(('up_skin',patch))

            elif 'L_SKIN' in descript:
                patch = int(descript.split('.')[1][:3]) - 1
                var_map.append(('lo_skin',patch))

            elif 'U_STRING' in descript:
                stringer = int(descript.split('.')[1][:2]) - 1
                seg = int(descript.split('.')[-1])
                var_map.append(('up_stringer',(stringer,seg)))

            elif 'L_STRING' in descript:
                stringer = int(descript.split('.')[1][:2]) - 1
                seg = int(descript.split('.')[-1])
                var_map.append(('lo_stringer',(stringer,seg)))

            else:
                print('UNKNOWN COMPONENT')
        self.comps  = struct_comps
        self.var_map = var_map

    def setup(self):

        for comp_name, count in self.comps.items():
            self.add_input(comp_name,shape=count)

        self.add_output('dv_struct',shape=self.ndvs)

    def compute(self,inputs,outputs):
        for i in range(self.ndvs):
            comp = self.var_map[i][0]
            ind = self.var_map[i][1]
            if 'stringer' in comp:
                outputs['dv_struct'][i] = inputs[comp][ind[0],ind[1]]
            else:
                outputs['dv_struct'][i] = inputs[comp][ind]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'dv_struct' in d_outputs:
                for i in range(self.ndvs):
                    comp = self.var_map[i][0]
                    ind = self.var_map[i][1]
                    if comp in d_inputs:
                        if 'stringer' in comp:
                            d_outputs['dv_struct'][i] += d_inputs[comp][ind[0],ind[1]]
                        else:
                            d_outputs['dv_struct'][i] += d_inputs[comp][ind]
        if mode == 'rev':
            if 'dv_struct' in d_outputs:
                for i in range(self.ndvs):
                    comp = self.var_map[i][0]
                    ind = self.var_map[i][1]
                    if comp in d_inputs:
                        if 'stringer' in comp:
                            d_inputs[comp][ind[0],ind[1]] += d_outputs['dv_struct'][i]
                        else:
                            d_inputs[comp][ind] += d_outputs['dv_struct'][i]

class SmoothnessEvaluatorGrid(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('delta',default=0.001) # [m]
        self.options.declare('rows',default=1)
        self.options.declare('columns',default=1)

    def setup(self):
        self.rows = self.options['rows']
        self.columns = self.options['columns']

        self.ndiffs = 2 * (self.columns-1) * self.rows + 2 * self.columns * (self.rows-1)

        # row major thicknesses
        self.add_input('thickness',shape=self.rows*self.columns)
        self.add_output('diff',shape=self.ndiffs)

    def compute(self,inputs,outputs):
        delta = self.options['delta']

        j = 0
        for row in range(self.rows):
            for col in range(self.columns-1):
                i = self.columns * row + col
                outputs['diff'][j]   = inputs['thickness'][i]   - inputs['thickness'][i+1] - delta
                outputs['diff'][j+1] = inputs['thickness'][i+1] - inputs['thickness'][i]   - delta
                j += 2

        for row in range(self.rows-1):
            for col in range(self.columns):
                i  = self.columns * row + col
                i2 = self.columns * (row+1) + col
                outputs['diff'][j]   = inputs['thickness'][i]  - inputs['thickness'][i2] - delta
                outputs['diff'][j+1] = inputs['thickness'][i2] - inputs['thickness'][i]  - delta
                j += 2

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'diff' in d_outputs:
                if 'thickness' in d_inputs:
                    j = 0
                    for row in range(self.rows):
                        for col in range(self.columns-1):
                            i  = self.columns * row + col
                            d_outputs['diff'][j]   += d_inputs['thickness'][i]   - d_inputs['thickness'][i+1]
                            d_outputs['diff'][j+1] += d_inputs['thickness'][i+1] - d_inputs['thickness'][i]
                            j += 2

                    for row in range(self.rows-1):
                        for col in range(self.columns):
                            i  = self.columns * row + col
                            i2 = self.columns * (row+1) + col
                            d_outputs['diff'][j]   += d_inputs['thickness'][i]  - d_inputs['thickness'][i2]
                            d_outputs['diff'][j+1] += d_inputs['thickness'][i2] - d_inputs['thickness'][i]
                            j += 2
        if mode == 'rev':
            if 'diff' in d_outputs:
                if 'thickness' in d_inputs:
                    j = 0
                    for row in range(self.rows):
                        for col in range(self.columns-1):
                            i  = self.columns * row + col
                            d_inputs['thickness'][i]   += d_outputs['diff'][j]
                            d_inputs['thickness'][i+1] -= d_outputs['diff'][j]
                            d_inputs['thickness'][i+1] += d_outputs['diff'][j+1]
                            d_inputs['thickness'][i]   -= d_outputs['diff'][j+1]
                            j += 2

                    for row in range(self.rows-1):
                        for col in range(self.columns):
                            i  = self.columns * row + col
                            i2 = self.columns * (row+1) + col
                            d_inputs['thickness'][i]  += d_outputs['diff'][j]
                            d_inputs['thickness'][i2] -= d_outputs['diff'][j]
                            d_inputs['thickness'][i2] += d_outputs['diff'][j+1]
                            d_inputs['thickness'][i]  -= d_outputs['diff'][j+1]
                            j += 2

class StructDistributor(om.ExplicitComponent):
    def setup(self):
        self.add_input('struct_dv')

        self.add_output('ribs', shape=18)
        self.add_output('le_spar', shape=18)
        self.add_output('te_spar', shape=18)
        self.add_output('up_skin', shape=162)
        self.add_output('lo_skin', shape=162)
        self.add_output('up_stringer', shape=(8,18))
        self.add_output('lo_stringer', shape=(8,18))

        self.declare_partials(of=['*'], wrt=['struct_dv'], method='fd')

    def compute(self, inputs, outputs):
        outputs['ribs'][:] = inputs['struct_dv']
        outputs['le_spar'][:] = inputs['struct_dv']
        outputs['te_spar'][:] = inputs['struct_dv']
        outputs['up_skin'][:] = inputs['struct_dv']
        outputs['lo_skin'][:] = inputs['struct_dv']
        outputs['up_stringer'][:,:] = inputs['struct_dv']
        outputs['lo_stringer'][:,:] = inputs['struct_dv']


if __name__ == "__main__":
    from openmdao.api import Problem
    prob = Problem()
    prob.model.add_subsystem('StructDistributor',StructDistributor())
    prob.model.add_subsystem('StructDvDistributor',StructDvMapper())
    prob.model.add_subsystem('smoothness_grid',SmoothnessEvaluatorGrid(rows=3,columns=4))

    prob.setup(force_alloc_complex=True)
    prob.check_partials(method='cs',compact_print=True)
