package de.upb.isml.tornede.ecai2020.experiments.rankers;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.ml.tsc.distances.ITimeSeriesDistance;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;

public class KnnRanker extends NonRandomIdBasedRanker {

	private PipelinePerformanceStorage pipelinePerformanceStorage;
	private DatasetFeatureRepresentationMap datasetFeatureRepresentationMap;
	private List<Integer> trainingDatasetIds;

	private ITimeSeriesDistance distanceFunction;
	private int k;

	public KnnRanker(PipelinePerformanceStorage pipelinePerformanceStorage, DatasetFeatureRepresentationMap datasetFeatureRepresentationMap, ITimeSeriesDistance distanceFunction, int k) {
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
		this.datasetFeatureRepresentationMap = datasetFeatureRepresentationMap;
		this.distanceFunction = distanceFunction;
		this.k = k;
	}

	@Override
	public void train(List<Integer> trainingDatasetIds) {
		this.trainingDatasetIds = new ArrayList<>(trainingDatasetIds);
	}

	@Override
	public List<Pair<Integer, Double>> getRankingOfPipelinesOnDataset(List<Integer> pipelineIdsToRank, int datasetId) {
		AveragePerformanceRanker averageRankRanker = new AveragePerformanceRanker(pipelinePerformanceStorage);
		List<Integer> kNearestNeighborDatasets = findKNearestDatasets(datasetId, k);
		averageRankRanker.train(kNearestNeighborDatasets);
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
		return k + "_nn_" + distanceFunction.getClass().getSimpleName().toLowerCase();
	}
}
