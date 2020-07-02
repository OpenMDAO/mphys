import numpy as np
import openmdao.api as om

class PatchList:
    
    def __init__(self,bdf):
        
        self.bdf = bdf
        
        f = open(self.bdf, "r")
        self.contents = f.read().split()
        f.close()
        
    def read_families(self):
                
        a = [i for i, s in enumerate(self.contents) if 'family' in s]
        families = [0 for x in range(len(a))]
        family_ID = np.zeros(len(a),'int')
        for i in range(0,len(a)):
            families[i] = self.contents[a[i]+1]
            family_ID[i] = int(self.contents[a[i]+4])
        
        self.families = families
        self.family_ID = family_ID
        
    def create_DVs(self):
        
        upper_skin = np.zeros([len(self.families),10],'int')
        lower_skin = np.zeros([len(self.families),10],'int')
        le_spar = np.zeros([len(self.families),10],'int')
        te_spar = np.zeros([len(self.families),10],'int')
        rib = np.zeros([len(self.families),10],'int')
        
        for i in range(0,len(self.families)):
            _,comp,seg = self.families[i].split('/')
            
            if 'IMPD' in seg:
                seg = seg[0:seg.find(':')]
            
            if 'U_SKIN' in comp:
                _,c_id = comp.split('.')
                _,s_id = seg.split('.')
                upper_skin[int(c_id),int(s_id)] = self.family_ID[i]
            
            if 'L_SKIN' in comp:
                _,c_id = comp.split('.')
                _,s_id = seg.split('.')
                lower_skin[int(c_id),int(s_id)] = self.family_ID[i]
                
            if 'LE_SPAR' in comp:
                _,s_id = seg.split('.')
                le_spar[int(s_id),0] = self.family_ID[i]   
                
            if 'TE_SPAR' in comp:
                _,s_id = seg.split('.')
                te_spar[int(s_id),0] = self.family_ID[i] 

            if 'RIB' in comp:
                _,c_id = comp.split('.')
                _,s_id = seg.split('.')
                rib[int(c_id),int(s_id)] = self.family_ID[i]
            
        upper_skin = upper_skin[np.sum(upper_skin,axis=1)>0,:]
        self.upper_skin = upper_skin[:,np.sum(upper_skin,axis=0)>0]
        self.upper_skin = np.flip(self.upper_skin,axis=0)
        self.n_us = np.size(self.upper_skin,axis=0)

        lower_skin = lower_skin[np.sum(lower_skin,axis=1)>0,:]
        self.lower_skin = lower_skin[:,np.sum(lower_skin,axis=0)>0]
        self.lower_skin = np.flip(self.lower_skin,axis=0)
        self.n_ls = np.size(self.lower_skin,axis=0)
        
        le_spar = le_spar[np.sum(le_spar,axis=1)>0,:]
        self.le_spar = le_spar[:,np.sum(le_spar,axis=0)>0]
        self.n_le = np.size(self.le_spar,axis=0)
        
        te_spar = te_spar[np.sum(te_spar,axis=1)>0,:]
        self.te_spar = te_spar[:,np.sum(te_spar,axis=0)>0]
        self.n_te = np.size(self.te_spar,axis=0)
        
        rib = rib[np.sum(rib,axis=1)>0,:]
        self.rib = rib[:,np.sum(rib,axis=0)>0]
        self.n_rib = np.size(self.rib,axis=0)
        
        self.n_dvs = (self.upper_skin>0).sum() + (self.lower_skin>0).sum() + \
            (self.le_spar>0).sum() + (self.te_spar>0).sum() + \
            (self.rib>0).sum()
        self.n_dvs = int(self.n_dvs)


class DesignPatches(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('bdf', types=str)
        self.options.declare('patch_list')
        
    def setup(self):
        
        patches = self.options['patch_list']
        
        self.add_input('upper_skin_thickness',shape=patches.n_us)
        self.add_input('lower_skin_thickness',shape=patches.n_ls)
        self.add_input('le_spar_thickness',shape=patches.n_le)
        self.add_input('te_spar_thickness',shape=patches.n_te)
        self.add_input('rib_thickness',shape=patches.n_rib)
        
        self.add_output('dv_struct',shape=patches.n_dvs)
        
        self.declare_partials('dv_struct','upper_skin_thickness')
        self.declare_partials('dv_struct','lower_skin_thickness')
        self.declare_partials('dv_struct','le_spar_thickness')
        self.declare_partials('dv_struct','te_spar_thickness')
        self.declare_partials('dv_struct','rib_thickness')
        
    def compute(self,inputs,outputs):
        
        patches = self.options['patch_list']
        
        self.T_us = np.zeros([patches.n_dvs,patches.n_us])
        self.T_us[patches.upper_skin[np.nonzero(patches.upper_skin)[0],np.nonzero(patches.upper_skin)[1]]-1,np.nonzero(patches.upper_skin)[0]] = 1
        
        self.T_ls = np.zeros([patches.n_dvs,patches.n_ls])
        self.T_ls[patches.lower_skin[np.nonzero(patches.lower_skin)[0],np.nonzero(patches.lower_skin)[1]]-1,np.nonzero(patches.lower_skin)[0]] = 1
        
        self.T_le = np.zeros([patches.n_dvs,patches.n_le])
        self.T_le[patches.le_spar[np.nonzero(patches.le_spar)[0],np.nonzero(patches.le_spar)[1]]-1,np.nonzero(patches.le_spar)[0]] = 1
        
        self.T_te = np.zeros([patches.n_dvs,patches.n_te])
        self.T_te[patches.te_spar[np.nonzero(patches.te_spar)[0],np.nonzero(patches.te_spar)[1]]-1,np.nonzero(patches.te_spar)[0]] = 1
        
        self.T_rib = np.zeros([patches.n_dvs,patches.n_rib])
        self.T_rib[patches.rib[np.nonzero(patches.rib)[0],np.nonzero(patches.rib)[1]]-1,np.nonzero(patches.rib)[0]] = 1
                 
        outputs['dv_struct'] = self.T_us@inputs['upper_skin_thickness'] + \
                         self.T_ls@inputs['lower_skin_thickness'] + \
                         self.T_le@inputs['le_spar_thickness'] + \
                         self.T_te@inputs['te_spar_thickness'] + \
                         self.T_rib@inputs['rib_thickness']
    
    def compute_partials(self,inputs,partials):
        
        partials['dv_struct','upper_skin_thickness'] = self.T_us
        partials['dv_struct','lower_skin_thickness'] = self.T_ls
        partials['dv_struct','le_spar_thickness'] = self.T_le
        partials['dv_struct','te_spar_thickness'] = self.T_te
        partials['dv_struct','rib_thickness'] = self.T_rib
        

class PatchSmoothness(om.ExplicitComponent):
    
    def initialize(self):
        
        self.options.declare('N', types=int)
        self.options.declare('delta',default=0.001)
        
    def setup(self): 
        
        self.add_input('thickness',shape=self.options['N'])
        self.add_output('diff',shape=2*(self.options['N']-1))
        self.declare_partials('diff','thickness')
        
    def compute(self,inputs,outputs):
        
        j = 0
        for i in range(0,self.options['N']-1):
            outputs['diff'][j]   = inputs['thickness'][i]   - inputs['thickness'][i+1] - self.options['delta']
            outputs['diff'][j+1] = inputs['thickness'][i+1] - inputs['thickness'][i]   - self.options['delta']
            j += 2
            
    def compute_partials(self,inputs,partials):
        
        partials['diff','thickness'] = np.zeros([2*(self.options['N']-1),len(inputs['thickness'])])
        
        j = 0
        for i in range(0,self.options['N']-1):
            partials['diff','thickness'][j,i] = 1
            partials['diff','thickness'][j,i+1] = -1
            partials['diff','thickness'][j+1,i+1] = 1
            partials['diff','thickness'][j+1,i] = -1  
            j += 2

   