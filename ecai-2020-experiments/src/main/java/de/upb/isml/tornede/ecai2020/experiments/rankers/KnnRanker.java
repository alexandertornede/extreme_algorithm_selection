package de.upb.isml.tornede.ecai2020.experiments.rankers;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.ml.tsc.distances.ITimeSeriesDistance;
import de.upb.isml.tornede.ecai2020.experiments.rankers.regression.RegressionDatasetGenerator;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;

public class KnnRanker implements IdBasedRanker {

	private PipelinePerformanceStorage pipelinePerformanceStorage;
	private DatasetFeatureRepresentationMap datasetFeatureRepresentationMap;
	private List<Integer> trainingDatasetIds;
	private List<Integer> trainingPipelineIds;

	private ITimeSeriesDistance distanceFunction;
	private int k;
	private boolean useBayesianAveraging;

	private RegressionDatasetGenerator regressionDatasetGenerator;
	private List<Pair<Integer, Integer>> datasetAndAlgorithmTrainingPairs;

	public KnnRanker(PipelinePerformanceStorage pipelinePerformanceStorage, DatasetFeatureRepresentationMap datasetFeatureRepresentationMap, ITimeSeriesDistance distanceFunction, int k, RegressionDatasetGenerator regressionDatasetGenerator,
			boolean useBayesianAveraging) {
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
		this.datasetFeatureRepresentationMap = datasetFeatureRepresentationMap;
		this.distanceFunction = distanceFunction;
		this.k = k;
		this.regressionDatasetGenerator = regressionDatasetGenerator;
		this.useBayesianAveraging = useBayesianAveraging;
	}

	@Override
	public void train(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		this.trainingDatasetIds = new ArrayList<>(trainingDatasetIds);
		this.trainingPipelineIds = new ArrayList<>(trainingPipelineIds);
		this.datasetAndAlgorithmTrainingPairs = regressionDatasetGenerator.generateTrainingDataset(trainingDatasetIds, trainingPipelineIds).getY();
	}

	@Override
	public List<Pair<Integer, Double>> getRankingOfPipelinesOnDataset(List<Integer> pipelineIdsToRank, int datasetId) {

		List<Integer> kNearestNeighborDatasets = findKNearestDatasets(datasetId, k);

		AveragePerformanceRanker averageRankRanker = new AveragePerformanceRanker(pipelinePerformanceStorage, datasetAndAlgorithmTrainingPairs, useBayesianAveraging);
		averageRankRanker.train(kNearestNeighborDatasets, trainingPipelineIds);

		return averageRankRanker.getRankingOfPipelinesOnDataset(pipelineIdsToRank, datasetId);
	}

	private List<Integer> findKNearestDatasets(int referenceDatasetId, int k) {
		// System.out.println("knn: " + trainingDatasetIds.stream()
		// .map(id -> new Pair<>(id, distanceFunction.distance(datasetFeatureRepresentationMap.getFeatureRepresentationForDataset(id), datasetFeatureRepresentationMap.getFeatureRepresentationForDataset(referenceDatasetId))))
		// .sorted(Comparator.comparingDouble(p -> p.getY())).limit(k).map(p -> p.getX()).collect(Collectors.toList()));

		return trainingDatasetIds.stream()
				.map(id -> new Pair<>(id, distanceFunction.distance(datasetFeatureRepresentationMap.getFeatureRepresentationForDataset(id), datasetFeatureRepresentationMap.getFeatureRepresentationForDataset(referenceDatasetId))))
				.sorted(Comparator.comparingDouble(p -> p.getY())).limit(k).map(p -> p.getX()).collect(Collectors.toList());
	}

	@Override
	public String getName() {
		return k + "_nn_" + distanceFunction.getClass().getSimpleName().toLowerCase() + "_" + regressionDatasetGenerator.getName() + (useBayesianAveraging ? "_bayesianAveraging" : "");
	}

	@Override
	public void initialize(long randomSeed) {
		regressionDatasetGenerator.initialize(randomSeed);
	}
}
