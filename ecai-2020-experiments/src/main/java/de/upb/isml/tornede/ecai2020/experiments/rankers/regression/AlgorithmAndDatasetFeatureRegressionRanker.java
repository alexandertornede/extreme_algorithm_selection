package de.upb.isml.tornede.ecai2020.experiments.rankers.regression;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.rankers.IdBasedRanker;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class AlgorithmAndDatasetFeatureRegressionRanker implements IdBasedRanker {

	private PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap;
	private DatasetFeatureRepresentationMap datasetFeatureRepresentationMap;
	private PipelinePerformanceStorage pipelinePerformanceStorage;

	private Classifier nonTrainedRegressionFunction;
	private Classifier trainedRegressionFunction;

	private ArrayList<Attribute> attributeInfo;

	private RegressionDatasetGenerator regressionDatasetGenerator;

	public AlgorithmAndDatasetFeatureRegressionRanker(PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap, DatasetFeatureRepresentationMap datasetFeatureRepresentationMap, PipelinePerformanceStorage pipelinePerformanceStorage,
			Classifier regressionFunction, RegressionDatasetGenerator regressionDatasetGenerator) {
		this.pipelineFeatureRepresentationMap = pipelineFeatureRepresentationMap;
		this.datasetFeatureRepresentationMap = datasetFeatureRepresentationMap;
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
		this.nonTrainedRegressionFunction = regressionFunction;
		this.regressionDatasetGenerator = regressionDatasetGenerator;
	}

	@Override
	public void train(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {

		Instances instances = regressionDatasetGenerator.generateTrainingDataset(trainingDatasetIds, trainingPipelineIds).getX();
		attributeInfo = regressionDatasetGenerator.getAttributeInfo();

		try {
			trainedRegressionFunction = AbstractClassifier.makeCopy(nonTrainedRegressionFunction);
			trainedRegressionFunction.buildClassifier(instances);
			System.out.println("Trained regression function on dataset.");
		} catch (Exception e) {
			throw new RuntimeException("Could not train classifier!", e);
		}
	}

	@Override
	public List<Pair<Integer, Double>> getRankingOfPipelinesOnDataset(List<Integer> pipelineIdsToRank, int datasetId) {
		Instances dummyInstances = new Instances("dummy", attributeInfo, 0);
		dummyInstances.setClassIndex(dummyInstances.numAttributes() - 1);
		return pipelineIdsToRank.stream().map(id -> {
			try {
				Instance instance = createInstanceForPipelineAndDataset(id, datasetId);
				instance.setDataset(dummyInstances);
				dummyInstances.add(instance);
				return new Pair<>(id, trainedRegressionFunction.classifyInstance(instance));
			} catch (Exception e) {
				throw new RuntimeException("Could not apply regression function.", e);
			}
		}).sorted(Comparator.comparingDouble(p -> ((Pair<Integer, Double>) p).getY()).reversed()).collect(Collectors.toList());

	}

	private Instance createInstanceForPipelineAndDataset(int pipelineId, int trainingDatasetId) {
		double[] datasetFeatures = datasetFeatureRepresentationMap.getFeatureRepresentationForDataset(trainingDatasetId);
		double[] pipelineFeatures = pipelineFeatureRepresentationMap.getFeatureRepresentationForPipeline(pipelineId);
		double targetValue = pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(pipelineId, trainingDatasetId);

		int numberOfFeatures = datasetFeatures.length + pipelineFeatures.length + 1;

		Instance instance = new DenseInstance(numberOfFeatures);
		int counter = 0;
		while (counter < datasetFeatures.length) {
			instance.setValue(counter, datasetFeatures[counter]);
			counter++;
		}
		while (counter < datasetFeatures.length + pipelineFeatures.length) {
			instance.setValue(counter, pipelineFeatures[counter - datasetFeatures.length]);
			counter++;
		}
		instance.setValue(numberOfFeatures - 1, targetValue);
		return instance;
	}

	@Override
	public String getName() {
		return "2xfeature_regression_" + regressionDatasetGenerator.getName();
	}

	@Override
	public void initialize(long randomSeed) {
		this.regressionDatasetGenerator.initialize(randomSeed);
	}

}
