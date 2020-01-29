package de.upb.isml.tornede.ecai2020.experiments.rankers;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;

public class OracleRanker extends NonRandomIdBasedRanker {

	private PipelinePerformanceStorage pipelinePerformanceStorage;

	public OracleRanker(PipelinePerformanceStorage pipelinePerformanceStorage) {
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
	}

	@Override
	public void train(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		// nothing to do here
	}

	@Override
	public List<Pair<Integer, Double>> getRankingOfPipelinesOnDataset(List<Integer> pipelineIdsToRank, int datasetId) {
		// System.out.println("oracle: " + pipelineIdsToRank.stream().map(p -> new Pair<>(p, pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(p, datasetId)))
		// .sorted(Comparator.comparingDouble(p -> ((Pair<Integer, Double>) p).getY()).reversed()).collect(Collectors.toList()));

		return pipelineIdsToRank.stream().map(p -> new Pair<>(p, pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(p, datasetId)))
				.sorted(Comparator.comparingDouble(p -> ((Pair<Integer, Double>) p).getY()).reversed()).collect(Collectors.toList());
	}

	@Override
	public String getName() {
		return "oracle";
	}

}
