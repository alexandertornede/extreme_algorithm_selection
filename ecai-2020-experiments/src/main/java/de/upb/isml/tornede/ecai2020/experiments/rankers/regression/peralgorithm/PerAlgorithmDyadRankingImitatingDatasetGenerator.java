package de.upb.isml.tornede.ecai2020.experiments.rankers.regression.peralgorithm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class PerAlgorithmDyadRankingImitatingDatasetGenerator extends AbstractPerAlgorithmRegressionDatasetGenerator {

	private int lengthOfRanking;
	private int numberOfRankingsPerTrainingDataset;

	public PerAlgorithmDyadRankingImitatingDatasetGenerator(DatasetFeatureRepresentationMap datasetFeatureRepresentationMap, PipelinePerformanceStorage pipelinePerformanceStorage, int lengthOfRanking,
			int numberOfRankingsPerTrainingDataset) {
		super(datasetFeatureRepresentationMap, pipelinePerformanceStorage);
		this.lengthOfRanking = lengthOfRanking;
		this.numberOfRankingsPerTrainingDataset = numberOfRankingsPerTrainingDataset;
	}

	@Override
	public List<Pair<Integer, Instances>> generateTrainingDataset(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {

		Map<Integer, Instances> algorithmIdToTrainingDataset = new HashMap<>();
		for (int pipelineId : trainingPipelineIds) {
			ArrayList<Attribute> datasetFeatureAttributes = getAttributeInfo();
			Attribute targetAttribute = new Attribute("performance");
			datasetFeatureAttributes.add(targetAttribute);

			Instances instances = new Instances("dataset", datasetFeatureAttributes, 0);
			instances.setClassIndex(instances.numAttributes() - 1);
			algorithmIdToTrainingDataset.put(pipelineId, instances);
		}

		for (int datasetId : trainingDatasetIds) {
			for (int i = 0; i < numberOfRankingsPerTrainingDataset; i++) {
				List<Integer> pipelineIdsToUse = new ArrayList<>(lengthOfRanking);
				Set<Double> performancesSeen = new HashSet<>(lengthOfRanking);

				while (pipelineIdsToUse.size() < lengthOfRanking) {
					int randomPipelineId = trainingPipelineIds.get(random.nextInt(trainingPipelineIds.size()));
					if (!pipelineIdsToUse.contains(randomPipelineId)) {
						double performanceOfId = pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(randomPipelineId, datasetId);
						if (performanceOfId > 0 && !performancesSeen.contains(performanceOfId)) {
							pipelineIdsToUse.add(randomPipelineId);
							performancesSeen.add(performanceOfId);

							Instances instances = algorithmIdToTrainingDataset.get(randomPipelineId);

							Instance instance = createInstanceForPipelineAndDataset(randomPipelineId, datasetId);
							instance.setDataset(instances);
							instances.add(instance);
						}
					}
				}
			}
		}

		return algorithmIdToTrainingDataset.entrySet().stream().map(e -> new Pair<>(e.getKey(), e.getValue())).collect(Collectors.toList());
	}

	@Override
	public String getName() {
		return "dyad_ranking_imitating_" + lengthOfRanking + "_" + numberOfRankingsPerTrainingDataset;
	}

}
