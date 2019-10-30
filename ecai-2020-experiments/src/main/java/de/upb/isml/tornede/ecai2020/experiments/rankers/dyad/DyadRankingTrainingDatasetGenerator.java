package de.upb.isml.tornede.ecai2020.experiments.rankers.dyad;

import java.util.List;

import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingDataset;

public interface DyadRankingTrainingDatasetGenerator {

	public DyadRankingDataset generateTrainingDataset(List<Integer> trainingDatasetIds);

	public void initialize(long randomSeed);

	public String getName();
}
