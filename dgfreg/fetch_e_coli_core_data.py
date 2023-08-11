from pathlib import Path

from cobra.io import load_model
from cobra.core import Reaction
import pandas as pd

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

def main():
    ecc = load_model("e_coli_core")

    def not_transport(reaction: Reaction) -> bool:
        return not all(
            r.name == p.name
            for r, p in zip(reaction.reactants, reaction.products)
        )

    non_transport_reactions = list(filter(not_transport, ecc.reactions))

    ecc_metabolites = pd.DataFrame(
        {"id": m.id, "name": m.name, "inchi_key": m.annotation["inchi_key"]}
        for r in non_transport_reactions for m in r.metabolites.keys()
    ).drop_duplicates(subset=["name"])

    ecc_reactions = pd.DataFrame(
        {"id": r.id, **r.annotation, "formula": r.reaction}
        for r in non_transport_reactions
    )

    ecc_metabolites.to_csv(RAW_DATA_DIR / "e_coli_ccm_metabolites.csv")
    ecc_reactions.to_csv(RAW_DATA_DIR / "e_coli_ccm_reactions.csv")


if __name__ == "__main__":
    main()
