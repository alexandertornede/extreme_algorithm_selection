package de.upb.isml.tornede.ecai2020.experiments.rankers.regression.peralgorithm;

import java.util.ArrayList;
import java.util.List;

import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class RandomPerAlgorithmRegressionDatasetGenerator extends AbstractPerAlgorithmRegressionDatasetGenerator {

	private double percentageFilled;

	public RandomPerAlgorithmRegressionDatasetGenerator(DatasetFeatureRepresentationMap datasetFeatureRepresentationMap, PipelinePerformanceStorage pipelinePerformanceStorage, double percentageFilled) {
		super(datasetFeatureRepresentationMap, pipelinePerformanceStorage);
		this.percentageFilled = percentageFilled;
	}

	@Override
	public String getName() {
		return "random_" + percentageFilled;
	}

	@Override
	public List<Pair<Integer, Instances>> generateTrainingDataset(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		List<Pair<Integer, Instances>> algorithmIdTrainingDatasetPairs = new ArrayList<>();

		for (int pipelineId : trainingPipelineIds) {
			ArrayList<Attribute> datasetFeatureAttributes = getAttributeInfo();
			Attribute targetAttribute = new Attribute("performance");
			datasetFeatureAttributes.add(targetAttribute);

			Instances instances = new Instances("dataset", datasetFeatureAttributes, 0);
			instances.setClassIndex(instances.numAttributes() - 1);
			for (int trainingDatasetId : trainingDatasetIds) {
				double randomValue = random.nextDouble();
				double targetValue = pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(pipelineId, trainingDatasetId);
				if (targetValue > 0 && randomValue < percentageFilled) {
					Instance instance = createInstanceForPipelineAndDataset(pipelineId, trainingDatasetId);
					instance.setDataset(instances);
					instances.add(instance);
				}
			}
			algorithmIdTrainingDatasetPairs.add(new Pair<>(pipelineId, instances));
			System.out.println("Generated dataset with " + instances.size() + " instances for algorithm " + pipelineId);
		}
		return algorithmIdTrainingDatasetPairs;
	}

}
