# -*- coding: utf-8 -*-
"""module monitors

This module implements generic monotoring of iterative integration. Specific
implementation is done in `flowdyn.integration`. A monitor directive is passed to `*.solve`
integration as a dictionary with its own parameters. A `monitor class`
is returned with 'output' key.

Example:

        $ python example_google.py

"""

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("unable to import matplotlib, some features will be missing")

# --------------------------------------------------------------------
# class monitor


class monitor():
    """ """
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
        """add it, time, value to monitor

        Args:
          it: 
          time: 
          value: 

        Returns:

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