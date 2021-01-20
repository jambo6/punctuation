class LeadLag:
    """ Applies the leadlag transformation to each path.

    Example:
        This is a string man
            [1, 2, 3] -> [[1, 1], [2, 1], [2, 2], [3, 2], [3, 3]]
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Interleave
        X_repeat = X.repeat_interleave(2, dim=1)

        # Split out lead and lag
        lead = X_repeat[:, 1:, :]
        lag = X_repeat[:, :-1, :]

        # Combine
        X_leadlag = torch.cat((lead, lag), 2)

        return X_leadlag


