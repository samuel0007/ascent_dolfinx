
from basix.finite_element import create_element
from basix import CellType, ElementFamily, LagrangeVariant
from scipy.spatial import Delaunay




P = 4

lagrange = create_element(ElementFamily.P, CellType.triangle, P, LagrangeVariant.equispaced)
print(lagrange.points)
tri = Delaunay(lagrange.points, qhull_options="d")
arr = tri.simplices.flatten()
cpp_str = "{" + ", ".join(str(x) for x in arr) + "}"
print("Delaunay simplices:")
print(cpp_str)

assert len(arr) == 3 * P * P