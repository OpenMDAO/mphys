import openmdao.api as om
from pygeo import DVGeometry
from mpi4py import MPI

# class that actually calls the dvgeometry methods
class OM_DVGEOCOMP(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('ffd_file', allow_none=False)
        self.options['distributed'] = True

    def setup(self):
        # create the DVGeo object that does the computations
        ffd_file = self.options['ffd_file']
        self.DVGeo = DVGeometry(ffd_file)

    def compute(self, inputs, outputs):

        # inputs are the geometric design variables
        self.DVGeo.setDesignVars(inputs)

        # ouputs are the coordinates of the pointsets we have
        for ptSet in self.DVGeo.points:
            # update this pointset and write it as output
            outputs[ptSet] = self.DVGeo.update(ptSet).flatten()

    def nom_addPointSet(self, points, ptName, **kwargs):
        # add the points to the dvgeo object
        self.DVGeo.addPointSet(points.reshape(len(points)//3, 3), ptName, **kwargs)

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

    def nom_addRefAxis(self, **kwargs):
        # we just pass this through
        return self.DVGeo.addRefAxis(**kwargs)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # only do the computations when we have more than zero entries in d_inputs in the reverse mode
        ni = len(list(d_inputs.keys()))

        if mode == 'rev' and ni > 0:
            for ptSetName in self.DVGeo.ptSetNames:
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
