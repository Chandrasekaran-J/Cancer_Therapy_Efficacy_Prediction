"""
precompute_rxn_list.py
Run ONCE before the FBA pipeline.
Loads Recon3D.xml and saves a canonical ordered list of all reaction IDs
and a reaction-info table → rxn_list.pkl

Usage:
    python precompute_rxn_list.py
"""

import os, sys
os.environ["OPTLANG_DEFAULT_INTERFACE"] = "optlang.gurobi_interface"

import pickle
import logging
import cobra

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = "/data/chandrasekaran/algo/Biomarkers"
XML_PATH   = "/data/chandrasekaran/algo/Recon3D.xml"
OUT_PKL    = os.path.join(BASE_DIR, "rxn_list.pkl")

def main():
    log.info(f"Loading {XML_PATH} ...")
    model = cobra.io.read_sbml_model(XML_PATH)
    log.info(f"Model loaded: {len(model.reactions)} reactions, {len(model.genes)} genes, {len(model.metabolites)} metabolites")

    # ── Canonical reaction list (ordered) ──────────────────────────────────
    rxn_ids   = [r.id   for r in model.reactions]
    rxn_names = [r.name for r in model.reactions]

    # ── Per-reaction metadata for downstream use ───────────────────────────
    rxn_info = {}
    for rxn in model.reactions:
        rxn_info[rxn.id] = {
            "name"     : rxn.name,
            "subsystem": rxn.subsystem,
            "gpr"      : rxn.gene_reaction_rule,
            # gene ids (frozenset strings)
            "gene_ids" : [g.id   for g in rxn.genes],
            "gene_names": [g.name for g in rxn.genes],
            # metabolite ids
            "metabolite_ids"  : [m.id   for m in rxn.metabolites],
            "metabolite_names": [m.name for m in rxn.metabolites],
            "reactant_ids"    : [m.id   for m in rxn.reactants],
            "product_ids"     : [m.id   for m in rxn.products],
            # bounds
            "lb": rxn.lower_bound,
            "ub": rxn.upper_bound,
            # is exchange / demand / sink  (no GPR = structural)
            "is_structural": len(rxn.gene_reaction_rule.strip()) == 0,
        }

    payload = {
        "rxn_ids"  : rxn_ids,
        "rxn_names": rxn_names,
        "rxn_info" : rxn_info,
        "n_rxns"   : len(rxn_ids),
    }

    with open(OUT_PKL, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    log.info(f"Saved {len(rxn_ids)} reactions → {OUT_PKL}")
    log.info("Done. You can now run step1_fba_chunk.py")


if __name__ == "__main__":
    main()
