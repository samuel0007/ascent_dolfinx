#include <ascent.hpp>
#include <conduit_blueprint.hpp>
#include <dolfinx.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/petsc.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <thrust/device_vector.h>

#include "poisson.h"

using namespace ascent;
using namespace conduit;
using namespace dolfinx;

static const std::unordered_map<mesh::CellType, std::string> dolfinx_celltype_to_blueprint = {
    {mesh::CellType::point, "point"},
    {mesh::CellType::interval, "line"},
    {mesh::CellType::triangle, "tri"},
    {mesh::CellType::quadrilateral, "quad"},
    {mesh::CellType::tetrahedron, "tet"},
    {mesh::CellType::hexahedron, "hex"},
    {mesh::CellType::prism, "prism"},
    {mesh::CellType::pyramid, "pyramid"}};


/// @brief Mesh the reference triangle assuming a lagrangian space structure
/// @param P 
/// @return dof connectivity
std::vector<int> MeshRefTriangle(const int P) {
    switch(P) {
        case 1: return {0, 1, 2};
        case 2: return {0, 5, 4, 5, 1, 3, 3, 4, 5, 4, 3, 2};
        case 3: return {0, 7, 5, 7, 8, 9, 8, 1, 3, 9, 5, 7, 3, 9, 8, 5, 9, 6, 9, 3, 4, 4, 6, 9, 6, 4, 2};
        case 4: return {3, 11, 1, 4, 5, 14, 8, 5, 2, 4, 13, 3, 6, 12, 7, 7, 12, 14, 9, 12, 0, 12, 6, 0, 5, 8, 7, 5, 7, 14, 11, 13, 10, 13, 11, 3, 12, 13, 14, 13, 4, 14, 13, 9, 10, 13, 12, 9};
        case 5: return {17, 3, 4, 3, 14, 1, 6, 20, 5, 10, 6, 2, 5, 19, 4, 14, 17, 13, 17, 14, 3, 13, 16, 12, 17, 16, 13, 18, 9, 8, 9, 18, 20, 10, 9, 20, 10, 20, 6, 15, 11, 12, 16, 15, 12, 15, 7, 0, 11, 15, 0, 7, 15, 8, 15, 18, 8, 15, 19, 18, 19, 15, 16, 19, 16, 17, 19, 17, 4, 18, 19, 20, 20, 19, 5};
        default:
            throw std::invalid_argument("MeshRefTriangle: unsupported P = " + std::to_string(P));
    }
}

template <typename T>
void MeshToBlueprintMesh(std::shared_ptr<fem::FunctionSpace<T>> V, const int P, conduit::Node &out)
{
    // Shape: (num_dofs, 3)
    std::vector<T> coords = V->tabulate_dof_coordinates(false);
    const int n_coords = coords.size() / 3;
    std::vector<T> X(n_coords), Y(n_coords), Z(n_coords);
    for (int i = 0; i < n_coords; ++i)
    {
        X[i] = coords[3 * i];
        Y[i] = coords[3 * i + 1];
        Z[i] = coords[3 * i + 2];
    }

    // Fill Conduit node for Blueprint mesh
    out["coordsets/coords/type"] = "explicit";
    out["coordsets/coords/values/x"].set(X.data(), n_coords);
    out["coordsets/coords/values/y"].set(Y.data(), n_coords);
    out["coordsets/coords/values/z"].set(Z.data(), n_coords);

    out["topologies/mesh/type"] = "unstructured";
    out["topologies/mesh/coordset"] = "coords";

    std::shared_ptr<const mesh::Topology> topology = V->mesh()->topology();
    auto it = dolfinx_celltype_to_blueprint.find(topology->cell_type());
    if (it == dolfinx_celltype_to_blueprint.end())
        throw std::runtime_error("Unknown cell type in dolfinx_celltype_to_blueprint mapping");
    out["topologies/mesh/elements/shape"] = it->second;


    // Connectivity
    const int tdim = topology->dim();
    const int num_local_cells = topology->index_map(tdim)->size_local();

    // Ref triangle connectivity
    std::vector<int> local_connectivity = MeshRefTriangle(P);
    const int P2 = P * P;
    assert(local_connectivity.size() == P2 * 3);
    // For triangles: one has a triangular mesh of P^2 triangles for each triangle
    std::vector<int> global_connectivity(P2 * 3 * num_local_cells);

    std::shared_ptr<const fem::DofMap> dofmap = V->dofmap();

    for(int i = 0; i < num_local_cells; ++i) {
        std::span<const std::int32_t> global_dofs = dofmap->cell_dofs(i);
        for(int k = 0; k < P2 * 3; ++k) {
            global_connectivity[i * P2 * 3 + k] = global_dofs[local_connectivity[k]];
        }
    }
    
    out["topologies/mesh/elements/connectivity"].set(global_connectivity.data(), global_connectivity.size());
}

template <typename T>
void DG0FunctionToBlueprintField(std::shared_ptr<fem::Function<T>> f,
                                 conduit::Node &out,
                                 const std::string &field_name)
{
    std::span<T> values = f->x()->mutable_array();
    out["fields"][field_name]["association"] = "element"; // DG
    out["fields"][field_name]["topology"] = "mesh";
    out["fields"][field_name]["values"].set_external(values.data(), values.size());
}

template <typename T>
void FunctionToBlueprintField(std::shared_ptr<fem::Function<T>> f,
                                 conduit::Node &out,
                                 const std::string &field_name)
{
    std::span<T> values = f->x()->mutable_array();
    out["fields"][field_name]["association"] = "vertex"; // CG1
    out["fields"][field_name]["topology"] = "mesh";
    out["fields"][field_name]["values"].set_external(values.data(), values.size());
}

int main(int argc, char **argv)
{
    using T = double;
    const int nelements = 10;
    constexpr int polynomial_degree = 3;

    init_logging(argc, argv);
    MPI_Init(&argc, &argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    {
        MPI_Comm comm{MPI_COMM_WORLD};
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        std::cout << ascent::about() << std::endl;

        auto mesh_p = std::make_shared<mesh::Mesh<T>>(mesh::create_rectangle<T>(
            comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {nelements, nelements},
            mesh::CellType::triangle,
            mesh::create_cell_partitioner(mesh::GhostMode::none)));
        mesh_p->topology()->create_entities(1);
        auto [vismesh, opt_indices, opt_flags] = refinement::refine(*mesh_p, std::nullopt, nullptr);
        std::shared_ptr<mesh::Mesh<T>> vismesh_p = std::make_shared<mesh::Mesh<T>>(vismesh);
        basix::FiniteElement element = basix::create_element<T>(
            basix::element::family::P, basix::cell::type::triangle, polynomial_degree,
            basix::element::lagrange_variant::gll_warped,
            basix::element::dpc_variant::unset, false);
        auto element_p = std::make_shared<const fem::FiniteElement<T>>(element);

        auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
            mesh_p, element_p));

        auto kappa = std::make_shared<fem::Constant<T>>(2.0);
        auto f = std::make_shared<fem::Function<T>>(V);
        auto g = std::make_shared<fem::Function<T>>(V);

        // Define variational forms
        fem::Form<T> a = fem::create_form<T>(*form_poisson_a, {V, V}, {},
                                             {{"kappa", kappa}}, {}, {});
        fem::Form<T> L = fem::create_form<T>(*form_poisson_L, {V},
                                             {{"f", f}, {"g", g}}, {}, {}, {});

        std::vector facets = mesh::locate_entities_boundary(
            *mesh_p, 1,
            [](auto x)
            {
                using U = typename decltype(x)::value_type;
                constexpr U eps = 1.0e-8;
                std::vector<std::int8_t> marker(x.extent(1), false);
                for (std::size_t p = 0; p < x.extent(1); ++p)
                {
                    auto x0 = x(0, p);
                    if (std::abs(x0) < eps or std::abs(x0 - 2) < eps)
                        marker[p] = true;
                }
                return marker;
            });
        std::vector bdofs = fem::locate_dofs_topological(
            *V->mesh()->topology_mutable(), *V->dofmap(), 1, facets);
        fem::DirichletBC<T> bc(0.0, bdofs, V);

        f->interpolate(
            [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
            {
                std::vector<T> f;
                for (std::size_t p = 0; p < x.extent(1); ++p)
                {
                    auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
                    auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
                    f.push_back(10 * std::exp(-(dx + dy) / 0.02));
                }

                return {f, {f.size()}};
            });

        g->interpolate(
            [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
            {
                std::vector<T> f;
                for (std::size_t p = 0; p < x.extent(1); ++p)
                    f.push_back(std::sin(5 * x(0, p)));
                return {f, {f.size()}};
            });

        auto u = std::make_shared<fem::Function<T>>(V);

        la::petsc::Matrix A(fem::petsc::create_matrix(a), false);
        la::Vector<T> b(L.function_spaces()[0]->dofmap()->index_map,
                        L.function_spaces()[0]->dofmap()->index_map_bs());

        MatZeroEntries(A.mat());
        fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                             a, {bc});
        MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
        fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V,
                             {bc});
        MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

        b.set(0.0);
        fem::assemble_vector(b.mutable_array(), L);
        fem::apply_lifting<T, T>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
        b.scatter_rev(std::plus<T>());
        bc.set(b.mutable_array(), std::nullopt);

        la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
        la::petsc::options::set("ksp_type", "preonly");
        la::petsc::options::set("pc_type", "lu");
        lu.set_from_options();

        lu.set_operator(A.mat());
        la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
        la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
        lu.solve(_u.vec(), _b.vec());

        // Update ghost values before output
        u->x()->scatter_fwd();

        const std::string output_field_name = "u";
        Node conduit_mesh;
        MeshToBlueprintMesh(V, polynomial_degree, conduit_mesh);
        FunctionToBlueprintField(u, conduit_mesh, output_field_name);


        // ---- ASCENT ----
        Ascent ascent_runner;
        ascent_runner.open();
        ascent_runner.publish(conduit_mesh);

        // declare a scene to render the dataset
        Node scenes;
        scenes["s1/plots/p1/type"] = "pseudocolor";
        scenes["s1/plots/p1/field"] = output_field_name;
        scenes["s1/plots/p2/type"] = "mesh";
        scenes["s1/image_prefix"] = output_field_name;

        Node actions;
        Node &add_act = actions.append();
        add_act["action"] = "add_scenes";
        add_act["scenes"] = scenes;

        std::cout << actions.to_yaml() << std::endl;

        ascent_runner.execute(actions);

        ascent_runner.close();
    }
    PetscFinalize();
    MPI_Finalize();
}
