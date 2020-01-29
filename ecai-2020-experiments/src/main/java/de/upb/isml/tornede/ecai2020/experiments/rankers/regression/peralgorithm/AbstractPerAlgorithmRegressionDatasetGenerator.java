package de.upb.isml.tornede.ecai2020.experiments.rankers.regression.peralgorithm;

import java.util.ArrayList;
import java.util.Random;

import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;

public abstract class AbstractPerAlgorithmRegressionDatasetGenerator implements PerAlgorithmRegressionDatasetGenerator {

	protected Random random;

	protected DatasetFeatureRepresentationMap datasetFeatureRepresentationMap;
	protected PipelinePerformanceStorage pipelinePerformanceStorage;

	public AbstractPerAlgorithmRegressionDatasetGenerator(DatasetFeatureRepresentationMap datasetFeatureRepresentationMap, PipelinePerformanceStorage pipelinePerformanceStorage) {
		this.datasetFeatureRepresentationMap = datasetFeatureRepresentationMap;
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
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
	public void initialize(long randomSeed) {
		this.random = new Random(randomSeed);
	}

	protected Instance createInstanceForPipelineAndDataset(int pipelineId, int trainingDatasetId) {
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
