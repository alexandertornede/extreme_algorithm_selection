package de.upb.isml.tornede.ecai2020.experiments.rankers;

public abstract class NonRandomIdBasedRanker implements IdBasedRanker {

	@Override
	public void initialize(long randomSeed) {
		// nothing to do here as this is a non ramdom ranker
	}

}
