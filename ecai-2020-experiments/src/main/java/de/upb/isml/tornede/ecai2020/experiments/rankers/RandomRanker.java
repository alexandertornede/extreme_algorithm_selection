package de.upb.isml.tornede.ecai2020.experiments.rankers;

import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;

public class RandomRanker implements IdBasedRanker {

	private Random random;

	@Override
	public void train(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		// nothing to do here
	}

	@Override
	public List<Pair<Integer, Double>> getRankingOfPipelinesOnDataset(List<Integer> pipelineIdsToRank, int datasetId) {
		return pipelineIdsToRank.stream().map(id -> new Pair<>(id, random.nextDouble())).sorted(Comparator.comparingDouble(p -> ((Pair<Integer, Double>) p).getY()).reversed()).collect(Collectors.toList());
	}

	@Override
	public void initialize(long randomSeed) {
		this.random = new Random(randomSeed);
	}

	@Override
	public String getName() {
		return "random";
	}

}
