package de.upb.isml.tornede.ecai2020.experiments.rankers.dyad;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.math.linearalgebra.DenseDoubleVector;
import ai.libs.jaicore.ml.dyadranking.Dyad;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;

public class RandomDyadRankingDatasetGenerator implements DyadRankingTrainingDatasetGenerator {

	private PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap;
	private DatasetFeatureRepresentationMap datasetFeatureRepresentationMap;
	private PipelinePerformanceStorage pipelinePerformanceStorage;
	private Random random;

	private int lengthOfRanking;
	private int numberOfRankingsPerTrainingDataset;

	public RandomDyadRankingDatasetGenerator(PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap, DatasetFeatureRepresentationMap datasetFeatureRepresentationMap, PipelinePerformanceStorage pipelinePerformanceStorage,
			int lengthOfRankings, int numberOfRankingsPerTrainingDataset) {
		this.pipelineFeatureRepresentationMap = pipelineFeatureRepresentationMap;
		this.datasetFeatureRepresentationMap = datasetFeatureRepresentationMap;
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
		this.lengthOfRanking = lengthOfRankings;
		this.numberOfRankingsPerTrainingDataset = numberOfRankingsPerTrainingDataset;
	}

	@Override
	public DyadRankingDataset generateTrainingDataset(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		DyadRankingDataset dataset = new DyadRankingDataset();
		for (int datasetId : trainingDatasetIds) {
			for (int i = 0; i < numberOfRankingsPerTrainingDataset; i++) {
				List<Integer> pipelineIdsToUse = new ArrayList<>(lengthOfRanking);
				Set<Double> performancesSeen = new HashSet<>(lengthOfRanking);
				List<Pair<Dyad, Double>> dyadPerformancePairsOfRanking = new ArrayList<>(lengthOfRanking);
				while (pipelineIdsToUse.size() < lengthOfRanking) {
					int randomPipelineId = trainingPipelineIds.get(random.nextInt(trainingPipelineIds.size()));
					if (!pipelineIdsToUse.contains(randomPipelineId)) {
						double performanceOfId = pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(randomPipelineId, datasetId);
						if (performanceOfId > 0 && !performancesSeen.contains(performanceOfId)) {
							pipelineIdsToUse.add(randomPipelineId);
							performancesSeen.add(performanceOfId);
							Dyad dyad = constructDyadForDatasetAndPipeline(datasetId, randomPipelineId);
							dyadPerformancePairsOfRanking.add(new Pair<>(dyad, performanceOfId));
						}
					}
				}
				List<Dyad> rankingOfDyadsAccordingToPerformance = dyadPerformancePairsOfRanking.stream().sorted(Comparator.comparingDouble(p -> ((Pair<Dyad, Double>) p).getY()).reversed()).map(d -> d.getX()).collect(Collectors.toList());
				dataset.add(new DyadRankingInstance(rankingOfDyadsAccordingToPerformance));
			}
		}
		return dataset;
	}

	private Dyad constructDyadForDatasetAndPipeline(int datasetId, int pipelineId) {
		return new Dyad(new DenseDoubleVector(datasetFeatureRepresentationMap.getFeatureRepresentationForDataset(datasetId)), new DenseDoubleVector(pipelineFeatureRepresentationMap.getFeatureRepresentationForPipeline(pipelineId)));
	}

	@Override
	public void initialize(long randomSeed) {
		this.random = new Random(randomSeed);
	}

	@Override
	public String getName() {
		return "random_" + lengthOfRanking + "_" + numberOfRankingsPerTrainingDataset;
	}

}
