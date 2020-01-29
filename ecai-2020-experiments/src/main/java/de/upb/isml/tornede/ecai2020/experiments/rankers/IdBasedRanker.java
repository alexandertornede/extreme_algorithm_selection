package de.upb.isml.tornede.ecai2020.experiments.rankers;

import java.util.List;

import ai.libs.jaicore.basic.sets.Pair;

public interface IdBasedRanker {

	public void initialize(long randomSeed);

	public void train(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds);

	public List<Pair<Integer, Double>> getRankingOfPipelinesOnDataset(List<Integer> pipelineIdsToRank, int datasetId);

	public String getName();
}
