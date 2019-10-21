package de.upb.isml.tornede.ecai2020.experiments.rankers;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;

public class AverageRankBasedRanker implements IdBasedRanker {

	private PipelinePerformanceStorage pipelinePerformanceStorage;
	private List<Integer> pipelineIds;

	// Stores the average performance as a sorted list (in descending order) of pipeline ids and their respective average performance across training datasets
	private List<Pair<Integer, Double>> averageRankOfPipelines;

	public AverageRankBasedRanker(PipelinePerformanceStorage pipelinePerformanceStorage) {
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
		this.pipelineIds = pipelinePerformanceStorage.getPipelineIds();
	}

	@Override
	public void train(List<Integer> trainingDatasetIds) {
		Map<Integer, DescriptiveStatistics> pipelineToStatisticsMap = new HashMap<>();
		for (int datasetId : trainingDatasetIds) {
			List<Integer> pipelineSortedAccordingToPerformanceInDecreasingOrder = pipelineIds.stream().map(p -> new Pair<>(p, pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(p, datasetId)))
					.sorted(Comparator.comparingDouble(p -> ((Pair<Integer, Double>) p).getY()).reversed()).map(p -> p.getX()).collect(Collectors.toList());
			for (int pipelineId : pipelineIds) {
				if (!pipelineToStatisticsMap.containsKey(pipelineId)) {
					pipelineToStatisticsMap.put(pipelineId, new DescriptiveStatistics());
				}
				double rankOfPipelineOnDataset = (pipelineSortedAccordingToPerformanceInDecreasingOrder.indexOf(pipelineId) + 1);
				pipelineToStatisticsMap.get(pipelineId).addValue(rankOfPipelineOnDataset);
			}
		}
		averageRankOfPipelines = pipelineIds.stream().map(id -> new Pair<>(id, -pipelineToStatisticsMap.get(id).getMean())).sorted(Comparator.comparingDouble(p -> ((Pair<Integer, Double>) p).getY()).reversed()).collect(Collectors.toList());
	}

	@Override
	public List<Pair<Integer, Double>> getRankingOfPipelinesOnDataset(List<Integer> pipelineIdsToRank, int datasetId) {
		// System.out.println("average rank: " + averageRankOfPipelines.stream().filter(p -> pipelineIdsToRank.contains(p.getX())).collect(Collectors.toList()));
		return averageRankOfPipelines.stream().filter(p -> pipelineIdsToRank.contains(p.getX())).collect(Collectors.toList());
	}

	@Override
	public String getName() {
		return "average_rank";
	}
}
