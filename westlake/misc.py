from .reaction_matrices import ReactionMatrix
from .reaction_modules import ConstantReactionRate
from .reaction_terms import ConstantRateTerm


def create_constant_rate_model(df_reac, df_spec=None, den_norm=None):
    reaction_matrix = ReactionMatrix(df_reac, df_spec)
    rmat_1st, rmat_2nd = reaction_matrix.create_index_matrices()
    rate_1st = ConstantReactionRate(rmat_1st, df_reac["rate"].values)
    rate_2nd = ConstantReactionRate(rmat_2nd, df_reac["rate"].values)
    # The rate signs are included in the rates, and therefore, they are
    # unnecessary.
    rmat_1st.rate_sign = None
    rmat_2nd.rate_sign = None
    return ConstantRateTerm(rmat_1st, rate_1st, rmat_2nd, rate_2nd, den_norm)