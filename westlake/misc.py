from .reaction_matrices import ReactionMatrix
from .reaction_terms import ConstantRateTerm


def create_constant_rate_model(df_reac, df_spec=None, den_norm=None):
    reaction_matrix = ReactionMatrix(df_reac, df_spec)
    rmat_1st, rmat_2nd = reaction_matrix.create_index_matrices()
    return ConstantRateTerm(rmat_1st, rmat_2nd, df_reac["rate"].values, den_norm)