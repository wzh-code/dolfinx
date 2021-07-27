# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Support for representing Dirichlet boundary conditions that are enforced
via modification of linear systems.

"""

import collections.abc
import types
import typing

import numpy as np
import ufl
from dolfinx import cpp
from dolfinx.fem.function import Function, FunctionSpace


def locate_dofs_geometrical(V: typing.Iterable[typing.Union[cpp.fem.FunctionSpace, FunctionSpace]],
                            marker: types.FunctionType) -> typing.Union[np.ndarray,
                                                                        typing.Tuple[np.ndarray, np.ndarray]]:
    """Locate degrees-of-freedom geometrically using a marker function.

    Parameters
    ----------
    V
        Function space(s) in which to search for degree-of-freedom indices.

    marker
        A function that takes an array of points ``x`` with shape
        ``(gdim, num_points)`` and returns an array of booleans of length
        ``num_points``, evaluating to ``True`` for entities whose
        degree-of-freedom should be returned.

    Returns
    -------
    numpy.ndarray
        An array of degree-of-freedom indices (local to the process)
        for degrees-of-freedom whose coordinate evaluates to True for the
        marker function.

        If ``V`` is a list of two function spaces, then a pair of arrays is returned.

        Returned degree-of-freedom indices are unique the first array is sorted.
    """

    if isinstance(V, collections.abc.Sequence):
        _V = []
        for space in V:
            try:
                _V.append(space._cpp_object)
            except AttributeError:
                _V.append(space)
        return cpp.fem.locate_dofs_geometrical(_V, marker)
    else:
        try:
            _V = V._cpp_object
        except AttributeError:
            _V = V
        return cpp.fem.locate_dofs_geometrical(_V, marker)


def locate_dofs_topological(V: typing.Iterable[typing.Union[cpp.fem.FunctionSpace, FunctionSpace]],
                            entity_dim: int,
                            entities: typing.Sequence[int],
                            remote: bool = True) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray]]:
    """Locate degrees-of-freedom belonging to mesh entities topologically.

    Parameters
    ----------
    V
        Function space(s) in which to search for degree-of-freedom indices.
    entity_dim
        Topological dimension of entities where degrees-of-freedom are located.
    entities
        Indices of mesh entities of dimension ``entity_dim`` where
        degrees-of-freedom are located.
    remote : True
        True to also return "remotely located" degree-of-freedom indices.

    Returns
    -------
    numpy.ndarray
        An array of degree-of-freedom indices (local to the process) for
        degrees-of-freedom topologically belonging to mesh entities.

        If ``V`` is a list of two function spaces, then a pair of arrays is returned.

        Returned degree-of-freedom indices are unique the first array is sorted.
    """

    _entities = np.asarray(entities, dtype=np.int32)
    if isinstance(V, collections.abc.Sequence):
        _V = []
        for space in V:
            try:
                _V.append(space._cpp_object)
            except AttributeError:
                _V.append(space)
        return cpp.fem.locate_dofs_topological(_V, entity_dim, _entities, remote)
    else:
        try:
            _V = V._cpp_object
        except AttributeError:
            _V = V
        return cpp.fem.locate_dofs_topological(_V, entity_dim, _entities, remote)


class DirichletBC(cpp.fem.DirichletBC):
    def __init__(
            self,
            value: typing.Union[ufl.Coefficient, Function, cpp.fem.Function],
            dofs: typing.Union[typing.Sequence[int], typing.Tuple[typing.Sequence[int], typing.Sequence[int]]],
            V: typing.Union[FunctionSpace] = None):
        """Representation of a Dirichlet boundary condition that is
        imposed on a linear system.

        Parameters
        ----------
        value
            Lifted boundary values function.
        dofs
            Local indices of degrees-of-freedom in function space to which
            boundary condition is applied.
            Expects a tuple of length two of local indices if function
            space ``V`` is provides, in which case the first array
            contains the local indices for the function space that
            ``value`` is defined on and the second array in the tuple
            contains the corresponding local degree-of-freedom indices
            in ``V``.
        V : optional
            Function space of a problem to which boundary conditions are applied.
        """

        # Construct bc value
        if isinstance(value, ufl.Coefficient):
            _value = value._cpp_object
        elif isinstance(value, cpp.fem.Function):
            _value = value
        elif isinstance(value, Function):
            _value = value._cpp_object
        else:
            raise NotImplementedError

        if V is not None:
            # Extract cpp function space
            try:
                _V = V._cpp_object
            except AttributeError:
                _V = V
            super().__init__(_value, dofs, _V)
        else:
            super().__init__(_value, dofs)
