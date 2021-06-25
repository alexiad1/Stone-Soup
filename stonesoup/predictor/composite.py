# -*- coding: utf-8 -*-
from typing import Sequence

from ..base import Property
from ..predictor import Predictor
from ..predictor._utils import predict_lru_cache
from ..types.prediction import CompositePrediction
from ..types.state import CompositeState


class CompositePredictor(Predictor):
    """A composition of multiple sub-predictors.

    Predicts forward a :class:`CompositeState` composed of a sequence of states using a sequence
    of sub-predictors
    """
    sub_predictors: Sequence[Predictor] = Property(doc="A sequence of sub-predictors")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.sub_predictors, Sequence):
            raise ValueError("sub-predictors must be defined as an ordered list")

        if any(not isinstance(sub_predictor, Predictor) for sub_predictor in self.sub_predictors):
            raise ValueError("all sub-predictors must be a Predictor type")

    @property
    def transition_model(self):
        raise NotImplementedError("A composition of predictors have no defined transition model")

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        r"""The predict function

        Parameters
        ----------
        prior : :class:`~.CompositeState`
            The state of an object existing in a composite state space
        timestamp : :class:`datetime.datetime`, optional
            :math:`k`
        **kwargs :
            These are passed to each sub-predictor's prediction method

        Returns
        -------
        : :class:`~.CompositeState`
            The predicted composite state
        """

        if not isinstance(prior, CompositeState):
            raise ValueError("CompositePredictor can only be used with CompositeState types")
        if len(prior.sub_states) != len(self.sub_predictors):
            raise ValueError(
                "CompositeState must be composed of same number of sub-states as sub-predictors")

        prediction_sub_states = []

        for sub_predictor, sub_state in zip(self.sub_predictors, prior.sub_states):
            sub_prediction = sub_predictor.predict(prior=sub_state, timestamp=timestamp, **kwargs)
            prediction_sub_states.append(sub_prediction)

        return CompositePrediction(sub_states=prediction_sub_states)

    def __getitem__(self, index):
        """Can be indexed as a list, or sliced, in which case a new composite predictor will be
        created from the sub-list of sub-predictors."""
        if isinstance(index, slice):
            return self.__class__(self.sub_predictors.__getitem__(index))
        return self.sub_predictors.__getitem__(index)

    def __iter__(self):
        return iter(self.sub_predictors)

    def __len__(self):
        return self.sub_predictors.__len__()

    def __contains__(self, item):
        return self.sub_predictors.__contains__(item)

    def append(self, value):
        """Add value at end of :attr:`sub_predictors`.

        Parameters
        ----------
        value: :class:`~.Predictor`
            A Predictor to be added at the end of :attr:`sub_predictors`.
        """
        self.sub_predictors.append(value)
