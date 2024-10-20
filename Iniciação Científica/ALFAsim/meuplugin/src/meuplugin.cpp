#include <alfasim_sdk_api/alfasim_sdk.h>
#include "hook_specs.h"


HOOK_INITIALIZE(ctx){
    return 0;
}
HOOK_FINALIZE(ctx){
    return 0;
}
// HOOK_INITIALIZE(ctx){}
// HOOK_FINALIZE(ctx){}
// HOOK_UPDATE_PLUGINS_SECONDARY_VARIABLES_ON_FIRST_TIMESTEP(ctx){}
// HOOK_UPDATE_PLUGINS_SECONDARY_VARIABLES_TIME_EXPLICIT(ctx){}
// HOOK_UPDATE_PLUGINS_SECONDARY_VARIABLES(ctx){}
// HOOK_UPDATE_PLUGINS_SECONDARY_VARIABLES_ON_TRACER_SOLVER(ctx){}
// HOOK_CALCULATE_MASS_SOURCE_TERM(ctx, mass_source, n_fields, n_control_volumes){}
// HOOK_CALCULATE_MOMENTUM_SOURCE_TERM(ctx, momentum_source, n_layers, n_faces){}
// HOOK_CALCULATE_ENERGY_SOURCE_TERM(ctx, energy_source, n_energy_equation, n_control_volumes){}
// HOOK_CALCULATE_TRACER_SOURCE_TERM(ctx, phi_source, n_tracers, n_control_volumes){}
// HOOK_INITIALIZE_STATE_VARIABLES_CALCULATOR(ctx, P, T, T_mix, phi, n_control_volumes, n_layers, n_tracers){}
// HOOK_FINALIZE_STATE_VARIABLES_CALCULATOR(ctx){}
// HOOK_CALCULATE_STATE_VARIABLE(ctx, P, T, n_control_volumes, phase_id, field_id, property_id, output){}
// HOOK_CALCULATE_PHASE_PAIR_STATE_VARIABLE(ctx, P, T_mix, n_control_volumes, phase1_id, phase2_id, phase_pair_id, property_id, output){}
// HOOK_INITIALIZE_PARTICLE_DIAMETER_OF_SOLIDS_FIELDS(ctx, particle_diameter, n_control_volumes, solid_field_id){}
// HOOK_UPDATE_PARTICLE_DIAMETER_OF_SOLIDS_FIELDS(ctx, particle_diameter, n_control_volumes, solid_field_id){}
// HOOK_CALCULATE_SLIP_VELOCITY(ctx, U_slip, solid_field_index, layer_index, n_faces){}
// HOOK_CALCULATE_RELATIVE_SLURRY_VISCOSITY(ctx, mu_r, solid_field_index, layer_index, n_faces){}
// HOOK_INITIALIZE_MASS_FRACTION_OF_TRACER(ctx, phi_initial, tracer_index){}
// HOOK_CALCULATE_MASS_FRACTION_OF_TRACER_IN_PHASE(ctx, phi, phi_phase, tracer_index, phase_index, n_control_volumes){}
// HOOK_CALCULATE_MASS_FRACTION_OF_TRACER_IN_FIELD(ctx, phi_phase, phi_field, tracer_index, field_index, phase_index_of_field, n_control_volumes){}
// HOOK_SET_PRESCRIBED_BOUNDARY_CONDITION_OF_MASS_FRACTION_OF_TRACER(ctx, phi_presc, tracer_index){}
// HOOK_UPDATE_BOUNDARY_CONDITION_OF_MASS_FRACTION_OF_TRACER(ctx, phi_presc, tracer_index, vol_frac_bound, n_fields){}
// HOOK_CALCULATE_UCM_FRICTION_FACTOR_STRATIFIED(ctx, ff_wG, ff_wL, ff_i){}
// HOOK_CALCULATE_UCM_FRICTION_FACTOR_ANNULAR(ctx, ff_wG, ff_wL, ff_i){}
// HOOK_CALCULATE_LIQ_LIQ_FLOW_PATTERN(ctx, ll_fp, water_vol_frac){}
// HOOK_CALCULATE_LIQUID_EFFECTIVE_VISCOSITY(ctx, mu_l_eff, ll_fp){}
// HOOK_CALCULATE_GAS_LIQ_SURFACE_TENSION(ctx, sigma_gl, ll_fp){}
// HOOK_CALCULATE_LIQ_LIQ_SHEAR_FORCE_PER_VOLUME(ctx, shear_w, shear_i, u_fields, vol_frac_fields, ll_fp){}
// HOOK_CALCULATE_RELATIVE_EMULSION_VISCOSITY(ctx, mu_r, mu_disp, mu_cont, alpha_disp_in_layer, T, water_in_oil){}
// HOOK_FRICTION_FACTOR(v1, v2){}
// HOOK_ENV_TEMPERATURE(v3, v4){}
// HOOK_CALCULATE_ENTRAINED_LIQUID_FRACTION(U_S, rho, mu, sigma, D, theta){}
// HOOK_UPDATE_INTERNAL_DEPOSITION_LAYER(ctx, phase_id, thickness_variation_rate, density, heat_capacity, thermal_conductivity, n_control_volumes){}
