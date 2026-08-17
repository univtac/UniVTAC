# Ref: https://github.com/spiriMirror/libuipc-samples/blob/main/python/20_contact_system_feature/main.py

import numpy as np
import polyscope as ps
from polyscope import imgui

from uipc import view
from uipc import Logger, Timer, Animation
from uipc import Vector3, Transform, Quaternion, AngleAxis
from uipc import builtin
from uipc.core import Engine, World, Scene, ContactSystemFeature
from uipc.geometry import (
    GeometrySlot,
    SimplicialComplex,
    SimplicialComplexIO,
    Geometry,
    label_surface,
    label_triangle_orient,
    flip_inward_triangles,
    ground,
)
from uipc.constitution import AffineBodyConstitution, StableNeoHookean
from uipc.gui import SceneGUI
from uipc.unit import MPa, GPa


class ContactInfo:
    def __init__(self, name, csf: ContactSystemFeature):
        self.name = name
        self.csf: ContactSystemFeature = csf
        # Normal Contact
        self.NE = Geometry()  # energy
        self.NG = Geometry()  # gradient
        self.NH = Geometry()  # hessian
        # Frictional Contact
        self.FE = Geometry()  # energy
        self.FG = Geometry()  # gradient
        self.FH = Geometry()  # hessian

    def retrieve(self):
        # Normal Contact
        self.csf.contact_energy(self.name + "+N", self.NE)
        self.csf.contact_gradient(self.name + "+N", self.NG)
        self.csf.contact_hessian(self.name + "+N", self.NH)
        # Frictional Contact
        self.csf.contact_energy(self.name + "+F", self.FE)
        self.csf.contact_gradient(self.name + "+F", self.FG)
        self.csf.contact_hessian(self.name + "+F", self.FH)

    def display_energy(self, t: str, geo: Geometry):
        topo = geo.instances().find("topo")
        if topo is None:
            return
        topo_view = topo.view()
        # imgui.Text(f"[{self.name}+{t}] Contact Topo: {topo_view.reshape(-1, topo_view.shape[1])}")
        energy = geo.instances().find("energy")
        energy_view = energy.view()
        # imgui.Text(f"[{self.name}+{t}] Contact Energy: {energy.view()}")
        # print(f"[{self.name}+{t}] Contact Energy: {energy.view().shape}")

        return topo_view, energy_view
