import openmdao.api as om
from pygeo import DVGeometry, DVConstraints
from mpi4py import MPI
import numpy as np

# class that actually calls the dvgeometry methods
class OM_DVGEOCOMP(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('ffd_file', allow_none=False)
        self.options['distributed'] = True

    def setup(self):
        # create the DVGeo object that does the computations
        ffd_file = self.options['ffd_file']
        self.DVGeo = DVGeometry(ffd_file)
        self.DVCon = DVConstraints()
        self.DVCon.setDVGeo(self.DVGeo)
        self.omPtSetList = []

    def compute(self, inputs, outputs):

        # inputs are the geometric design variables
        self.DVGeo.setDesignVars(inputs)

        # ouputs are the coordinates of the pointsets we have
        for ptName in self.DVGeo.points:
            if ptName in self.omPtSetList:
                # update this pointset and write it as output
                outputs[ptName] = self.DVGeo.update(ptName).flatten()

        # compute the DVCon constraint values
        constraintfunc = dict()
        self.DVCon.evalFunctions(constraintfunc)
        for constraintname in constraintfunc:
            outputs[constraintname] = constraintfunc[constraintname]

    def nom_addPointSet(self, points, ptName, **kwargs):
        # add the points to the dvgeo object
        self.DVGeo.addPointSet(points.reshape(len(points)//3, 3), ptName, **kwargs)
        self.omPtSetList.append(ptName)

        # add an output to the om component
        self.add_output(ptName, val=points.flatten())

    def nom_add_point_dict(self, point_dict):
        # add every pointset in the dict, and set the ptset name as the key
        for k,v in point_dict.items():
            self.nom_addPointSet(v, k)

    def nom_addGeoDVGlobal(self, dvName, value, func):
        # define the input
        self.add_input(dvName, shape=value.shape)

        # call the dvgeo object and add this dv
        self.DVGeo.addGeoDVGlobal(dvName, value, func)

    def nom_addGeoDVLocal(self, dvName, axis='y'):
        nVal = self.DVGeo.addGeoDVLocal(dvName, axis=axis)
        self.add_input(dvName, shape=nVal)

    def nom_addThicknessConstraints2D(self, name, leList, teList, nSpan=10, nChord=10):
        self.DVCon.addThicknessConstraints2D(leList, teList, nSpan, nChord, lower=1.0, name=name)
        # TODO add output to openmdao
        self.add_output(name, val=np.ones((nSpan*nChord,)), shape=nSpan*nChord)


    def nom_addVolumeConstraint(self, name, leList, teList, nSpan=10, nChord=10):
        self.DVCon.addVolumeConstraint(leList, teList, nSpan=nSpan, nChord=nChord, name=name)
        # TODO add output to openmdao
        # self.constraints[constrainttype][constraintname]
        self.add_output(name, val=1.0)

    def nom_addRefAxis(self, **kwargs):
        # we just pass this through
        return self.DVGeo.addRefAxis(**kwargs)

    def nom_setConstraintSurface(self, surface):
        # constraint needs a triangulated reference surface at initialization
        self.DVCon.setSurface(surface)

    

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        print(self.comm.rank)
        # only do the computations when we have more than zero entries in d_inputs in the reverse mode
        ni = len(list(d_inputs.keys()))

        if mode == 'rev' and ni > 0:
            constraintfuncsens = dict()
            self.DVCon.evalFunctionsSens(constraintfuncsens)
            for constraintname in constraintfuncsens:
                for dvname in constraintfuncsens[constraintname]:
                    dcdx = constraintfuncsens[constraintname][dvname]
                    dout = d_outputs[constraintname]
                    d_inputs[dvname] += np.dot(np.transpose(dcdx),dout)

            for ptSetName in self.DVGeo.ptSetNames:
                if ptSetName in self.omPtSetList:
                    dout = d_outputs[ptSetName].reshape(len(d_outputs[ptSetName])//3, 3)
                    xdot = self.DVGeo.totalSensitivityTransProd(dout, ptSetName)

                    # loop over dvs and accumulate
                    xdotg = {}
                    for k in xdot:
                        # check if this dv is present
                        if k in d_inputs:
                            # do the allreduce
                            # TODO reove the allreduce when this is fixed in openmdao
                            # reduce the result ourselves for now. ideally, openmdao will do the reduction itself when this is fixed. this is because the bcast is also done by openmdao (pyoptsparse, but regardless, it is not done here, so reduce should also not be done here)
                            xdotg[k] = self.comm.allreduce(xdot[k], op=MPI.SUM)

                            # accumulate in the dict
                            d_inputs[k] += xdotg[k]
