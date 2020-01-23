package de.upb.isml.tornede.ecai2020.experiments.rankers;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.rankers.regression.RegressionDatasetGenerator;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;

public class AveragePerformanceRanker implements IdBasedRanker {

	private PipelinePerformanceStorage pipelinePerformanceStorage;
	private RegressionDatasetGenerator regressionDatasetGenerator;
	private List<Pair<Integer, Integer>> availableDatasetAndAlgorithmTrainingPairs;

	private boolean useBayesianAveraging;

	// Stores the average performance as a sorted list (in descending order) of pipeline ids and their respective average performance across training datasets
	private List<Pair<Integer, Double>> averagePerformanceOfPipelines;

	public AveragePerformanceRanker(PipelinePerformanceStorage pipelinePerformanceStorage, RegressionDatasetGenerator regressionDatasetGenerator, boolean useBayesianAveraging) {
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
		this.regressionDatasetGenerator = regressionDatasetGenerator;
		this.useBayesianAveraging = useBayesianAveraging;
	}

	public AveragePerformanceRanker(PipelinePerformanceStorage pipelinePerformanceStorage, List<Pair<Integer, Integer>> availableDatasetAndAlgorithmTrainingPairs, boolean useBayesianAveraging) {
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
		this.availableDatasetAndAlgorithmTrainingPairs = availableDatasetAndAlgorithmTrainingPairs;
		this.useBayesianAveraging = useBayesianAveraging;
	}

	@Override
	public void train(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		if (useBayesianAveraging) {
			trainWithBayesianAveraging(trainingDatasetIds, trainingPipelineIds);
		} else {
			trainWithNormalAveraging(trainingDatasetIds, trainingPipelineIds);
		}
	}

	private void trainWithBayesianAveraging(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		Map<Integer, DescriptiveStatistics> sparsePipelineToStatisticsMap = computeSparsePipelineToStatisticsMap(trainingDatasetIds, trainingPipelineIds);
		Map<Integer, DescriptiveStatistics> densePipelineToStatisticsMap = computeDensePipelineToStatisticsMap(trainingDatasetIds, trainingPipelineIds);

		for (Entry<Integer, DescriptiveStatistics> entry : sparsePipelineToStatisticsMap.entrySet()) {
			for (double performanceValue : entry.getValue().getValues()) {
				densePipelineToStatisticsMap.get(entry.getKey()).addValue(performanceValue);
			}
		}

		List<Integer> pipelineIds = pipelinePerformanceStorage.getPipelineIds();

		averagePerformanceOfPipelines = pipelineIds.stream().map(id -> new Pair<>(id, getAveragePerformanceOfPipeline(id, densePipelineToStatisticsMap))).sorted(Comparator.comparingDouble(p -> ((Pair<Integer, Double>) p).getY()).reversed())
				.collect(Collectors.toList());
	}

	private Map<Integer, DescriptiveStatistics> computeDensePipelineToStatisticsMap(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		Map<Integer, DescriptiveStatistics> pipelineToStatisticsMap = new HashMap<>();
		for (int datasetId : trainingDatasetIds) {
			for (int pipelineId : trainingPipelineIds) {
				if (!pipelineToStatisticsMap.containsKey(pipelineId)) {
					pipelineToStatisticsMap.put(pipelineId, new DescriptiveStatistics());
				}
				double performanceOfPipelineOnDataset = pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(pipelineId, datasetId);
				if (performanceOfPipelineOnDataset > 0) {
					pipelineToStatisticsMap.get(pipelineId).addValue(performanceOfPipelineOnDataset);
				}
			}
		}
		return pipelineToStatisticsMap;
	}

	private void trainWithNormalAveraging(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		Map<Integer, DescriptiveStatistics> pipelineToStatisticsMap = computeSparsePipelineToStatisticsMap(trainingDatasetIds, trainingPipelineIds);
		List<Integer> pipelineIds = pipelinePerformanceStorage.getPipelineIds();
		averagePerformanceOfPipelines = pipelineIds.stream().map(id -> new Pair<>(id, getAveragePerformanceOfPipeline(id, pipelineToStatisticsMap))).sorted(Comparator.comparingDouble(p -> ((Pair<Integer, Double>) p).getY()).reversed())
				.collect(Collectors.toList());
	}

	private Map<Integer, DescriptiveStatistics> computeSparsePipelineToStatisticsMap(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		List<Pair<Integer, Integer>> datasetAndAlgorithmTrainingPairs = null;
		if (regressionDatasetGenerator != null) {
			datasetAndAlgorithmTrainingPairs = regressionDatasetGenerator.generateTrainingDataset(trainingDatasetIds, trainingPipelineIds).getY();
		} else {
			datasetAndAlgorithmTrainingPairs = availableDatasetAndAlgorithmTrainingPairs;
		}

		Map<Integer, DescriptiveStatistics> pipelineToStatisticsMap = new HashMap<>();
		for (Pair<Integer, Integer> datasetAndAlgorithmPair : datasetAndAlgorithmTrainingPairs) {
			int datasetId = datasetAndAlgorithmPair.getX();
			if (trainingDatasetIds.contains(datasetId)) {
				int pipelineId = datasetAndAlgorithmPair.getY();
				if (!pipelineToStatisticsMap.containsKey(pipelineId)) {
					pipelineToStatisticsMap.put(pipelineId, new DescriptiveStatistics());
				}
				double performanceOfPipelineOnDataset = pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(pipelineId, datasetId);
				pipelineToStatisticsMap.get(pipelineId).addValue(performanceOfPipelineOnDataset);
			}
		}
		return pipelineToStatisticsMap;
	}

	private double getAveragePerformanceOfPipeline(int pipelineId, Map<Integer, DescriptiveStatistics> pipelineToStatisticsMap) {
		if (!pipelineToStatisticsMap.containsKey(pipelineId)) {
			return 0;
		}
		return pipelineToStatisticsMap.get(pipelineId).getMean();
	}

	@Override
	public List<Pair<Integer, Double>> getRankingOfPipelinesOnDataset(List<Integer> pipelineIdsToRank, int datasetId) {
		// System.out.println("average performance: " + averagePerformanceOfPipelines.stream().filter(p -> pipelineIdsToRank.contains(p.getX())).collect(Collectors.toList()));
		return averagePerformanceOfPipelines.stream().filter(p -> pipelineIdsToRank.contains(p.getX())).collect(Collectors.toList());
	}

	@Override
	public String getName() {
		return "average_performance_" + regressionDatasetGenerator.getName() + (useBayesianAveraging ? "_bayesianAveraging" : "");
	}

	@Override
	public void initialize(long randomSeed) {
		regressionDatasetGenerator.initialize(randomSeed);
	}

}
