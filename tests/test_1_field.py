import numpy as np
import pytest
import flowdyn.mesh  as mesh
import flowdyn.modelphy.convection as conv
#import flowdyn.modeldisc as modeldisc
import flowdyn.field as field
#import flowdyn.xnum  as xnum
#import flowdyn.integration as tnum

# @pytest.fixture
# def convmodel():
#     return conv.model(1.)


class Test_fdataclass():

    convmodel = conv.model(convcoef=1.)
    curmesh = mesh.unimesh(ncell=50, length=1.)

    def test_init_empty(self):
        f = field.fdata(self.convmodel, self.curmesh, [])
        assert f.time == 0. # default value
        assert f.data == []
        f.set_time(10.)
        assert f.time == 10. 

    def test_init_expand(self):
        f = field.fdata(self.convmodel, self.curmesh, [ 1. ])
        assert f.time == 0. # default value
        assert np.size(f.data[0]) == self.curmesh.ncell
        assert np.average(f.data[0]) == 1.

# class Test_scalar1d():

#     xsch = xnum.extrapol3()
#     curmesh = mesh.unimesh(ncell=50, length=1.)
#     convmodel = conv.model(convcoef=1.)

#     def init_sinperk(self, mesh, k):
#         return np.sin(2*k*np.pi/mesh.length*mesh.centers())

#     # def test_checkend_tottime(self):
#     #     endtime = 5.
#     #     cfl     = .5
#     #     stop_directive = { 'tottime': endtime }
#     #     finit = field.fdata(self.convmodel, self.curmesh, [ self.init_sinperk(self.curmesh, k=4) ] )
#     #     rhs = modeldisc.fvm(self.convmodel, self.curmesh, self.xsch)
#     #     solver = tnum.rk4(self.curmesh, rhs)
#     #     nsol = 11
#     #     tsave = np.linspace(0, 2*endtime, nsol, endpoint=True)
#     #     fsol = solver.solve(finit, cfl, tsave, stop=stop_directive)
#     #     assert len(fsol) < nsol # end before expected by tsave
#     #     assert not fsol[-1].isnan()
#     #     assert fsol[-1].time < 2*endtime

class Test_fieldlistclass():

    convmodel = conv.model(convcoef=1.)
    curmesh = mesh.unimesh(ncell=50, length=1.)

    def test_init_scalararray(self):
        flist = field.fieldlist()
        f1 = field.fdata(self.convmodel, self.curmesh, [1.])
        flist.append(f1)
        f2 = f1.copy()
        f2.set_time(10.)
        flist.append(f2)
        assert len(flist) == 2
        assert flist[0].time == 0.
        assert flist[-1].time == 10.
