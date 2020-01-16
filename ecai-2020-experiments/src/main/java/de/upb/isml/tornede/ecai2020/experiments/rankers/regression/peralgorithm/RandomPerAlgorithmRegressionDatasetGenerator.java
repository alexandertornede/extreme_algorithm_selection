package de.upb.isml.tornede.ecai2020.experiments.rankers.regression.peralgorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class RandomPerAlgorithmRegressionDatasetGenerator implements PerAlgorithmRegressionDatasetGenerator {

	private Random random;
	private double percentageFilled;

	private PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap;
	private DatasetFeatureRepresentationMap datasetFeatureRepresentationMap;
	private PipelinePerformanceStorage pipelinePerformanceStorage;

	public RandomPerAlgorithmRegressionDatasetGenerator(DatasetFeatureRepresentationMap datasetFeatureRepresentationMap, PipelinePerformanceStorage pipelinePerformanceStorage, double percentageFilled) {
		this.datasetFeatureRepresentationMap = datasetFeatureRepresentationMap;
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
		this.percentageFilled = percentageFilled;
	}

	@Override
	public void initialize(long randomSeed) {
		this.random = new Random(randomSeed);
	}

	@Override
	public String getName() {
		return "random_" + percentageFilled;
	}

	@Override
	public ArrayList<Attribute> getAttributeInfo() {
		ArrayList<Attribute> attributes = new ArrayList<>();
		for (int i = 0; i < 45; i++) {
			attributes.add(new Attribute("d" + i));
		}
		return attributes;

	}

	@Override
	public List<Pair<Integer, Instances>> generateTrainingDataset(List<Integer> trainingDatasetIds) {
		List<Pair<Integer, Instances>> algorithmIdTrainingDatasetPairs = new ArrayList<>();

		for (int pipelineId : pipelinePerformanceStorage.getPipelineIds()) {
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

	private Instance createInstanceForPipelineAndDataset(int pipelineId, int trainingDatasetId) {
		double[] datasetFeatures = datasetFeatureRepresentationMap.getFeatureRepresentationForDataset(trainingDatasetId);
		double targetValue = pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(pipelineId, trainingDatasetId);

		int numberOfFeatures = datasetFeatures.length + 1;

		Instance instance = new DenseInstance(numberOfFeatures);
		int counter = 0;
		while (counter < datasetFeatures.length) {
			instance.setValue(counter, datasetFeatures[counter]);
			counter++;
		}
		instance.setValue(numberOfFeatures - 1, targetValue);
		return instance;
	}
}
