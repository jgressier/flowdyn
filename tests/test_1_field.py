import numpy as np
import pytest
import flowdyn.mesh  as mesh
import flowdyn.mesh2d as mesh2d
import flowdyn.modelphy.convection as conv
import flowdyn.modelphy.euler as euler
import flowdyn.field as field
import matplotlib.pyplot as plt
# pytest --mpl # to compare images
# pytest --mpl-generate-path=baseline # to generate images in specific folder

# @pytest.fixture
# def convmodel():
#     return conv.model(1.)

class Test_fdataclass_scalar():

    convmodel = conv.model(convcoef=1.)
    curmesh = mesh.unimesh(ncell=50, length=1.)

    def test_init_empty(self):
        f = field.fdata(self.convmodel, self.curmesh, [])
        assert f.time == 0. # default value
        assert f.it == -1 # default value
        assert f.data == []
        f.set_time(10.)
        assert f.time == 10.

    def test_reset(self):
        f = field.fdata(self.convmodel, self.curmesh, [])
        f.set_time(10.)
        f.reset(t=5.)
        assert f.time == 5.
        assert f.it == -1 # default value
        f.reset(it=20)
        assert f.time == 0. # default value
        assert f.it == 20

    def test_init_expand(self):
        f = field.fdata(self.convmodel, self.curmesh, [ 1. ])
        assert f.time == 0. # default value
        assert np.size(f.data[0]) == self.curmesh.ncell
        assert np.average(f.data[0]) == 1.

    @pytest.mark.mpl_image_compare
    def test_plotdata_sca(self):
        def fn(x):
            return np.exp(-2*np.square(x-.5))*np.sin(20*(x-.5))
        f = field.fdata(self.convmodel, self.curmesh, 
                [ fn(self.curmesh.centers()) ] )
        fig, ax = plt.subplots(1,1)
        f.plot('q')
        return fig

class Test_fdataclass_vec():

    model = euler.model()
    mesh1d = mesh.unimesh(ncell=100, length=10.)
    mesh2d = mesh2d.mesh2d(60, 50, 20., 10.)

    def fn(self, x):
        return np.exp(-np.square(x)/2)*np.sin(5*x)

    @pytest.mark.mpl_image_compare
    def test_plotdata_vec(self):
        f = field.fdata(self.model, self.mesh1d, 
                [ 1., self.fn(self.mesh1d.centers()-5.), 5. ])
        fig, ax = plt.subplots(1,1)
        f.plot('mach', axes=ax, style='r-')
        return fig

    @pytest.mark.mpl_image_compare
    def test_plot2dcontour(self):
        xc, yc = self.mesh2d.centers()
        f = field.fdata(self.model, self.mesh2d, 
                [ 1., euler.datavector(0., 0.), 
                self.fn(np.sqrt(np.square(xc-8)+np.square(yc-4))) ])
        fig, ax = plt.subplots(1,1)
        f.contour('pressure', axes=ax, style='r-')
        return fig

    @pytest.mark.mpl_image_compare
    def test_plot2dcontourf(self):
        xc, yc = self.mesh2d.centers()
        f = field.fdata(self.model, self.mesh2d, 
                [ 1., euler.datavector(0., 0.), 
                self.fn(np.sqrt(np.square(xc-8)+np.square(yc-4))) ])
        fig, ax = plt.subplots(1,1)
        f.contourf('pressure', axes=ax, style='r-')
        return fig

class Test_fieldlistclass():

    convmodel = conv.model(convcoef=1.)
    curmesh = mesh.unimesh(ncell=50, length=1.)

    def test_append_scalararray(self):
        flist = field.fieldlist()
        f1 = field.fdata(self.convmodel, self.curmesh, [1.])
        flist.append(f1)
        f2 = f1.copy()
        f2.set_time(10.)
        flist.append(f2)
        assert len(flist) == 2
        assert flist[0].time == 0.
        assert flist[-1].time == 10.

    def test_extend_scalararray(self):
        flist = field.fieldlist()
        f1 = field.fdata(self.convmodel, self.curmesh, [1.])
        flist.append(f1)
        f2 = f1.copy()
        f2.set_time(10.)
        flist.append(f2)
        newlist = field.fieldlist()
        newlist.append(f2)
        newlist.append(f2)
        flist.extend(newlist)
        f2.set_time(100.)
        assert len(flist) == 4
        assert flist[0].time == 0.
        assert flist[-1].time == 100.

    @pytest.mark.mpl_image_compare
    def test_flist_plotxtcontour(self):
        def fn(x, t):
            return np.exp(-2*np.square(x-.5+.2*np.sin(10*t)))
        times = np.linspace(0., 5., 11, endpoint=True)
        flist = field.fieldlist()
        for t in times:
            f = field.fdata(self.convmodel, self.curmesh, 
                [ fn(self.curmesh.centers(),t) ] )
            f.set_time(t)
            flist.append(f)
        fig, ax = plt.subplots(1,1)
        flist.xtcontour('q')
        return fig

    @pytest.mark.mpl_image_compare
    def test_flist_plotxtcontourf(self):
        def fn(x, t):
            return np.exp(-2*np.square(x-.5+.2*np.sin(10*t)))
        times = np.linspace(0., 5., 11, endpoint=True)
        flist = field.fieldlist()
        for t in times:
            f = field.fdata(self.convmodel, self.curmesh, 
                [ fn(self.curmesh.centers(),t) ] )
            f.set_time(t)
            flist.append(f)
        fig, ax = plt.subplots(1,1)
        flist.xtcontourf('q')
        return fig