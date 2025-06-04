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

template <typename T>
void MeshToBlueprintMesh(mesh::Mesh<T> &mesh, conduit::Node &out)
{
    // Topology: get connectivity array
    std::shared_ptr<mesh::Topology> topology = mesh.topology();
    const int tdim = topology->dim();
    topology->create_connectivity(tdim, 0);
    std::vector<int> conn = topology->connectivity(tdim, 0)->array();

    // Geometry: get coordinates
    std::span<const T> coords = mesh.geometry().x();
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
    auto it = dolfinx_celltype_to_blueprint.find(topology->cell_type());
    if (it == dolfinx_celltype_to_blueprint.end())
        throw std::runtime_error("Unknown cell type in dolfinx_celltype_to_blueprint mapping");
    out["topologies/mesh/elements/shape"] = it->second;
    out["topologies/mesh/elements/connectivity"].set(conn.data(), conn.size());
}

template <typename T>
void DG0FunctionToBlueprintField(std::shared_ptr<fem::Function<T>> f,
                                 conduit::Node &out,
                                 const std::string &field_name)
{
    std::span<T> values = f->x()->mutable_array();
    out["fields"][field_name]["association"] = "element"; // DG
    // out["fields"][field_name]["basis"] = "L2_2D_P0";
    out["fields"][field_name]["topology"] = "mesh";
    out["fields"][field_name]["values"].set_external(values.data(), values.size());
}

// H1_%dD_P%d
// Only works with GL basis?
template <typename T>
void CG1FunctionToBlueprintField(std::shared_ptr<fem::Function<T>> f,
                                 conduit::Node &out,
                                 const std::string &field_name)
{
    std::span<T> values = f->x()->mutable_array();
    out["fields"][field_name]["association"] = "vertex"; // CG1
    out["fields"][field_name]["topology"] = "mesh";
    std::cout << "values size: " << values.size() << std::endl;
    out["fields"][field_name]["values"].set_external(values.data(), values.size());
}

int main(int argc, char **argv)
{
    using T = double;
    const int nelements = 10;
    constexpr int degreeOfBasis = 2;

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
            basix::element::family::P, basix::cell::type::triangle, degreeOfBasis,
            basix::element::lagrange_variant::gll_warped,
            basix::element::dpc_variant::unset, false);
        basix::FiniteElement element_DG = basix::create_element<T>(
            basix::element::family::P, basix::cell::type::triangle, 0,
            basix::element::lagrange_variant::gll_warped,
            basix::element::dpc_variant::unset, true);

        basix::FiniteElement element_vis = basix::create_element<T>(
            basix::element::family::P, basix::cell::type::triangle, 1,
            basix::element::lagrange_variant::equispaced,
            basix::element::dpc_variant::unset, false);

        auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
            mesh_p, std::make_shared<const fem::FiniteElement<T>>(element)));
        auto V_DG =
            std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
                mesh_p, std::make_shared<const fem::FiniteElement<T>>(element_DG)));
        auto V_vis =
            std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
                vismesh_p, std::make_shared<const fem::FiniteElement<T>>(element_vis)));

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

        auto u_out_DG = std::make_shared<fem::Function<T>>(V_DG);
        u_out_DG->interpolate(*u);

        
        auto u_out_CG = std::make_shared<fem::Function<T>>(V_vis);
        auto cell_map = vismesh_p->topology()->index_map(vismesh_p->topology()->dim());
        assert(cell_map);
        std::vector<std::int32_t> cells(
            cell_map->size_local() + cell_map->num_ghosts(), 0);
        std::iota(cells.begin(), cells.end(), 0);
        geometry::PointOwnershipData<T> interpolation_data = fem::create_interpolation_data(
            u_out_CG->function_space()->mesh()->geometry(),
            *u_out_CG->function_space()->element(),
            *u->function_space()->mesh(), std::span(cells), 1e-8);
        u_out_CG->interpolate(*u, cells, interpolation_data);

        const std::string output_field_name = "u";
        Node conduit_mesh;
        MeshToBlueprintMesh(*mesh_p, conduit_mesh);

        Node conduit_refined_mesh;
        MeshToBlueprintMesh(*vismesh_p, conduit_refined_mesh);

        // DG0FunctionToBlueprintField(u_out_DG, conduit_mesh, output_field_name);
        CG1FunctionToBlueprintField(u_out_CG, conduit_refined_mesh, output_field_name);

        // ---- From GPU pointer ----
        // std::span<const T> u_out_span = u_out->x()->array();
        // thrust::device_vector<T> u_out_d(u_out_span.size());
        // std::span<const T> u_out_d(u_out_span.size()) = std::span<const T>(
        //   thrust::raw_pointer_cast(phi_d.data()), phi_d.size());

        // ---- ASCENT ----
        Ascent ascent_runner;
        ascent_runner.open();
        // ascent_runner.publish(conduit_mesh);
        ascent_runner.publish(conduit_refined_mesh);

        // declare a scene to render the dataset
        Node scenes;
        scenes["s1/plots/p1/type"] = "pseudocolor";
        scenes["s1/plots/p1/field"] = output_field_name;
        scenes["s1/image_prefix"] = output_field_name;
        scenes["s1/plots/p2/type"] = "mesh";

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
