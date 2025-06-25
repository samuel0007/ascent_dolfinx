from basix.ufl import element
from ufl import (
    Coefficient,
    Constant,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    ds,
    dx,
    grad,
    inner,
)

# e = element("Lagrange", "triangle", 2)
# coord_element = element("Lagrange", "triangle", 1, shape=(2,))

e = element("Lagrange", "tetrahedron", 3)
coord_element = element("Lagrange", "tetrahedron", 1, shape=(3,))

mesh = Mesh(coord_element)

V = FunctionSpace(mesh, e)

u = TrialFunction(V)
v = TestFunction(V)
f = Coefficient(V)
g = Coefficient(V)
kappa = Constant(mesh)

a = kappa * inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds

