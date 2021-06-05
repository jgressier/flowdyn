# -*- coding: utf-8 -*-
"""module monitors

This module implements generic monotoring of iterative integration. Specific
implementation is done in `flowdyn.integration`. A monitor directive is passed to `*.solve`
integration as a dictionary with its own parameters. A `monitor class`
is returned with 'output' key.

Example:

        $ python example_google.py

"""

# --------------------------------------------------------------------
# class monitor


class monitor():
    """ """
    def __init__(self, name):
        self._name = name
        self._it = []
        self._time = []
        self._value = []

    def name(self):
        """get monitor name"""
        return self._name

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
