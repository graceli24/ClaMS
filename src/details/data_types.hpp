// Copyright 2023-2026 Lawrence Livermore National Security, LLC and other ClaMS
// Project Developers. See the top-level COPYRIGHT file for details.

#pragma once

#include <cstddef>
#include <cstdint>

#include <metall/metall.hpp>

// CLAMS_USE_SALTATLAS is defined in cpp files
#ifdef CLAMS_USE_SALTATLAS
#include <saltatlas/dnnd/dnnd_adv.hpp>
#endif

#include "edge_list.hpp"

namespace clams {

// Point/vertex ID type.
using id_t = uint32_t;
// Distance type.
using distance_t = double;
// Feature element type.
using fe_t = float;

using weighted_edge_list_t = clams::weighted_edge_list<
    id_t, distance_t,
    metall::manager::scoped_fallback_allocator_type<std::byte>>;
using weighted_edge_t = weighted_edge_list_t::value_type;

#ifdef CLAMS_USE_SALTATLAS
using dnnd_t =
    saltatlas::dnnd_adv<id_t, saltatlas::pm_feature_vector<fe_t>, distance_t>;
// KNNG type supports Metall.
using pm_knng_t  = typename dnnd_t::knn_index_type;
using neighbor_t = typename dnnd_t::neighbor_type;
using point_t    = typename dnnd_t::point_type;
#endif
}  // namespace clams