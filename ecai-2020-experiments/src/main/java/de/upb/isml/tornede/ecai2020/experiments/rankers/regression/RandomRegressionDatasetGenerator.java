package de.upb.isml.tornede.ecai2020.experiments.rankers.regression;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class RandomRegressionDatasetGenerator extends AbstractRegressionDatasetGenerator {

	private Random random;

	private ArrayList<Attribute> attributeInfo;

	private int datasetSize;

	public RandomRegressionDatasetGenerator(boolean oldDataset, PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap, DatasetFeatureRepresentationMap datasetFeatureRepresentationMap,
			PipelinePerformanceStorage pipelinePerformanceStorage, int datasetSize) {
		super(oldDataset, pipelineFeatureRepresentationMap, datasetFeatureRepresentationMap, pipelinePerformanceStorage);
		this.datasetSize = datasetSize;
	}

	@Override
	public Instances generateTrainingDataset(List<Integer> trainingDatasetIds) {
		List<Attribute> datasetFeatureAttributes = createDatasetAttributeList();
		List<Attribute> pipelineFeatureAttributes = createAlgorithmAttributeList();
		Attribute targetAttribute = new Attribute("performance");
		attributeInfo = new ArrayList<>();
		attributeInfo.addAll(datasetFeatureAttributes);
		attributeInfo.addAll(pipelineFeatureAttributes);
		attributeInfo.add(targetAttribute);

		Instances instances = new Instances("dataset", attributeInfo, 0);
		instances.setClassIndex(instances.numAttributes() - 1);

		for (int pipelineId : pipelinePerformanceStorage.getPipelineIds()) {
			for (int trainingDatasetId : trainingDatasetIds) {
				double targetValue = pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(pipelineId, trainingDatasetId);
				if (targetValue > 0) {
					Instance instance = createInstanceForPipelineAndDataset(pipelineId, trainingDatasetId);
					instance.setDataset(instances);
					instances.add(instance);
				}
			}
		}
		System.out.println("Generated dataset with " + instances.size() + " instances.");
		instances = sampleInstances(instances, datasetSize);
		System.out.println("Sampled dataset down to " + instances.size() + " instances.");

		return instances;
	}

	private Instances sampleInstances(Instances instances, int amount) {
		// if we have nothing to sample or not enough, we return the original dataset
		if (amount <= 0 || amount > instances.size()) {
			return instances;
		}
		Instances sampledInstances = new Instances(instances.relationName() + "_" + amount, attributeInfo, amount);
		sampledInstances.setClassIndex(sampledInstances.numAttributes() - 1);
		Set<Integer> sampledIndices = new HashSet<>();
		while (sampledInstances.size() < amount) {
			int randomIndex = random.nextInt(instances.size());
			if (!sampledIndices.contains(randomIndex)) {
				sampledIndices.add(randomIndex);
				Instance randomInstance = instances.get(randomIndex);
				randomInstance.setDataset(sampledInstances);
				sampledInstances.add(randomInstance);
			}
		}
		return sampledInstances;
	}

	@Override
	public void initialize(long randomSeed) {
		this.random = new Random(randomSeed);
	}

	@Override
	public String getName() {
		return "random";
	}

	@Override
	public ArrayList<Attribute> getAttributeInfo() {
		return attributeInfo;
	}

}
