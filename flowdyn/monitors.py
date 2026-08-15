# -*- coding: utf-8 -*-
"""Provide monitors for iterative time integration.

Specific monitor implementations are configured by :mod:`flowdyn.integration`.

Example:
    Pass a monitor directive to an integrator's ``solve`` method. The directive's
    ``output`` entry then contains the resulting :class:`monitor` instance.
"""

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("unable to import matplotlib, some features will be missing")

# --------------------------------------------------------------------
# class monitor


class monitor():
    """Store sampled values and their iteration and time coordinates."""
    def __init__(self, name):
        self._name = name
        self.reset()
        
    def name(self):
        """get monitor name"""
        return self._name

    def reset(self):
        self._it = []
        self._time = []
        self._value = []

    def append(self, it, time, value):
        """Append a sampled value to the monitor.

        Args:
            it: Iteration number of the sample.
            time: Physical time of the sample.
            value: Monitored value.
        """
        self._it.append(it)
        self._time.append(time)
        self._value.append(value)

    def lastratio(self):
        return self._value[-1]/self._value[0]

    def plot_it(self, ax=plt, **kwargs):
        ax.plot(self._it, self._value, **kwargs)

    def plot_time(self, ax=plt, **kwargs):
        ax.plot(self._time, self._value, **kwargs)

    def semilogplot_it(self, ax=plt, **kwargs):
        ax.semilogy(self._it, self._value, **kwargs)

    def semilogplot_time(self, ax=plt, **kwargs):
        ax.semilogy(self._time, self._value, **kwargs)
