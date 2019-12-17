"""
Functions to implement TMA on a cobrapy mode.

1. Create activity variables.
1. Create thermodynamics reation-based constraints.
2. Create metabolomics constraints.
3. Perform analysis
"""

import equilibrator_api
import equilibrator_api.compatibility as compat
import cobra  # just for the test!

from equilibrator_cache import create_compound_cache_from_quilt
from time import time  # just form test!

# Avoid management optlang solvers
Variable = None
Constraint = None


def _create_activity_var(met):
    return Variable(f"activity_{met.id}", lb=0, ub=0)


def formulate_activities(model):
    """Create an activity variable for every metabolite on `model`.

    Every activity's bounds are set to 0,0. These bounds will be changed during
    the metabolomics application, if any.
    """
    model.add_cons_vars(
        [_create_activity_var(met) for met in model.metabolites]
    )


def _formulate_deltas(phased_reaction, cc):
    big_number = 10e3
    dG0_prime, dG0_uncertainty = cc.dG0_prime(phased_reaction)
    delta_G0 = Variable(
        f"deltag_{phased_reaction.id}",
        lb=dG0_prime - dG0_uncertainty,
        ub=dG0_prime + dG0_uncertainty,
    )
    delta_G = Variable(
        f"deltastd_{phased_reaction.id}", ub=big_number, lb=big_number
    )
    return delta_G, delta_G0


def _formulate_mets(metabolites, model, RT):
    return sum(
        [
            RT * model.solver.variables.get(f"activity_{metabolite.id}")
            for metabolite in metabolites
        ]
    )


def create_thermo_constrain(phased_reaction, model, cc, RT):
    """Add variables and create a thermo constraint for a `reaction`.

    ΔG = ΔGº + ∑_i RTln(activity_i)
    """
    metabolites = model.reactions.get_by_id(phased_reaction.id).metabolites

    delta_G, delta_G0 = _formulate_deltas(phased_reaction, cc)
    log_conc = _formulate_mets(metabolites, model, RT)

    return Constraint(delta_G0 - delta_G + log_conc, lb=0, ub=0)


def formulate_reactions(model, phased_reactions, RT, cc):
    """Apply thermodynamic constraints for the reactions.

    Parameters
    ----------
    model: cobra.Model
    phased_map: dict
        reaction id: equilibrator_api.Phased_Reaction
    RT: float
        gas constant on kJ/mol times the T in Kelvin

    """
    model.add_cons_vars(
        [
            create_thermo_constrain(reac, model, cc, RT)
            for reac in phased_reactions
        ]
    )


def is_trans(reaction):
    """Check if a reaction is a transporter."""
    return len(reaction.metabolites) == 1


def build_tma_problem(model, T=298.15):
    """Call conversion, if any and returns solution."""
    global Variable, Constraint

    Variable = model.problem.Variable
    Constraint = model.problem.Constraint
    thermo = model.copy()
    RT = T * 8.314472 / 1000
    # It will be interesting in the future to have the dictionary to show the
    # reactions that couldn't be computed
    phased_reactions = list(
        compat.map_cobra_reactions(
            create_compound_cache_from_quilt(),
            filter(lambda x: not is_trans(x), thermo.reactions),
        ).values()
    )
    cc = equilibrator_api.ComponentContribution(temperature=f"{T}K")
    formulate_activities(thermo)
    formulate_reactions(thermo, phased_reactions, cc, RT)
    return thermo


def tma(model, T=298.15):
    """Formulate and optimize TMA LP problem."""
    return build_tma_problem(model).optimize()


if __name__ == "__main__":
    model = cobra.io.read_sbml_model("iAB_RBC_283.xml")
    model.solver = "glpk"
    T = 298.15
    RT = T * 8.314472 / 1000

    # It will be interesting in the future to have the dictionary to show the
    # reactions that couldn't be computed
    start = time()
    phased_reactions = list(
        compat.map_cobra_reactions(
            create_compound_cache_from_quilt(), model.reactions
        ).values()
    )
    print(f"CLOCK map_cobra_reactions: {start - time()}")

    with model as thermo:
        cc = equilibrator_api.ComponentContribution(temperature=f"{T}K")
        start = time()
        formulate_activities(thermo)
        print(f"CLOCK formulate_activities: {start - time()}")

        start = time()
        formulate_reactions(thermo, phased_reactions, cc, RT)
        print(f"CLOCK formulate_reactions: {start - time()}")

        start = time()
        solution = thermo.optimize()
        print(f"CLOCK optimize thermo: {start - time()}")

    start = time()
    normal_solution = model.optimize()
    print(f"CLOCK optimize normal: {start - time()}")

    print(
        f"Solution of model -> {normal_solution.objective_value}"
        f"Sum of thermo fluxes -> {normal_solution.fluxes.sum()}",
        f"Solution of thermo model -> {solution.objective_value}",
        f"Sum of thermo fluxes -> {solution.fluxes.sum()}",
    )
