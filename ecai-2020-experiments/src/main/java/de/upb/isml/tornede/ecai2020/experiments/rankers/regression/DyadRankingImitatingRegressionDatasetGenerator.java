package de.upb.isml.tornede.ecai2020.experiments.rankers.regression;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class DyadRankingImitatingRegressionDatasetGenerator extends AbstractRegressionDatasetGenerator {

	private Random random;

	private int lengthOfRanking;
	private int numberOfRankingsPerTrainingDataset;

	private ArrayList<Attribute> attributeInfo;

	public DyadRankingImitatingRegressionDatasetGenerator(boolean oldDataset, PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap, DatasetFeatureRepresentationMap datasetFeatureRepresentationMap,
			PipelinePerformanceStorage pipelinePerformanceStorage, int lengthOfRankings, int numberOfRankingsPerTrainingDataset) {
		super(oldDataset, pipelineFeatureRepresentationMap, datasetFeatureRepresentationMap, pipelinePerformanceStorage);
		this.lengthOfRanking = lengthOfRankings;
		this.numberOfRankingsPerTrainingDataset = numberOfRankingsPerTrainingDataset;
	}

	@Override
	public Pair<Instances, List<Pair<Integer, Integer>>> generateTrainingDataset(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {

		List<Attribute> datasetFeatureAttributes = createDatasetAttributeList();
		List<Attribute> pipelineFeatureAttributes = createAlgorithmAttributeList();
		Attribute targetAttribute = new Attribute("performance");
		attributeInfo = new ArrayList<>();
		attributeInfo.addAll(datasetFeatureAttributes);
		attributeInfo.addAll(pipelineFeatureAttributes);
		attributeInfo.add(targetAttribute);

		List<Pair<Integer, Integer>> datasetAndAlgorithmPairs = new ArrayList<>();

		Instances instances = new Instances("dataset", attributeInfo, 0);
		instances.setClassIndex(instances.numAttributes() - 1);

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

							Instance instance = createInstanceForPipelineAndDataset(randomPipelineId, datasetId);
							instance.setDataset(instances);
							instances.add(instance);

							datasetAndAlgorithmPairs.add(new Pair<>(datasetId, randomPipelineId));
						}
					}
				}
			}
		}

		System.out.println("Generated dataset with " + instances.size() + " instances.");

		return new Pair<>(instances, datasetAndAlgorithmPairs);
	}

	@Override
	public void initialize(long randomSeed) {
		this.random = new Random(randomSeed);
	}

	@Override
	public String getName() {
		return "dyad_ranking_imitating_" + lengthOfRanking + "_" + numberOfRankingsPerTrainingDataset;
	}

	@Override
	public ArrayList<Attribute> getAttributeInfo() {
		return attributeInfo;
	}

}
