package de.upb.isml.tornede.ecai2020.experiments.rankers;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;

public class AveragePerformanceRanker extends NonRandomIdBasedRanker {

	private PipelinePerformanceStorage pipelinePerformanceStorage;

	// Stores the average performance as a sorted list (in descending order) of pipeline ids and their respective average performance across training datasets
	private List<Pair<Integer, Double>> averagePerformanceOfPipelines;

	public AveragePerformanceRanker(PipelinePerformanceStorage pipelinePerformanceStorage) {
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
	}

	@Override
	public void train(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		List<Integer> pipelineIds = pipelinePerformanceStorage.getPipelineIds();
		Map<Integer, DescriptiveStatistics> pipelineToStatisticsMap = new HashMap<>();
		for (int pipelineId : pipelineIds) {
			if (!pipelineToStatisticsMap.containsKey(pipelineId)) {
				pipelineToStatisticsMap.put(pipelineId, new DescriptiveStatistics());
			}
			for (int datasetId : trainingDatasetIds) {
				double performanceOfPipelineOnDataset = pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(pipelineId, datasetId);
				pipelineToStatisticsMap.get(pipelineId).addValue(performanceOfPipelineOnDataset);
			}
		}
		averagePerformanceOfPipelines = pipelineIds.stream().map(id -> new Pair<>(id, pipelineToStatisticsMap.get(id).getMean())).sorted(Comparator.comparingDouble(p -> ((Pair<Integer, Double>) p).getY()).reversed())
				.collect(Collectors.toList());
	}

	@Override
	public List<Pair<Integer, Double>> getRankingOfPipelinesOnDataset(List<Integer> pipelineIdsToRank, int datasetId) {
		// System.out.println("average performance: " + averagePerformanceOfPipelines.stream().filter(p -> pipelineIdsToRank.contains(p.getX())).collect(Collectors.toList()));
		return averagePerformanceOfPipelines.stream().filter(p -> pipelineIdsToRank.contains(p.getX())).collect(Collectors.toList());
	}

	@Override
	public String getName() {
		return "average_performance";
	}

}
