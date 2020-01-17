package de.upb.isml.tornede.ecai2020.experiments.rankers.regression.peralgorithm;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.rankers.IdBasedRanker;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PerAlgorithmRegressionRanker implements IdBasedRanker {

	private PipelinePerformanceStorage pipelinePerformanceStorage;
	private DatasetFeatureRepresentationMap datasetFeatureRepresentationMap;
	private PerAlgorithmRegressionDatasetGenerator datasetGenerator;
	private Classifier nonTrainedRegressionFunction;
	private Map<Integer, Classifier> algorithmToTrainedRegressionFunctionMap;

	private ArrayList<Attribute> attributeInfo;

	public PerAlgorithmRegressionRanker(PipelinePerformanceStorage pipelinePerformanceStorage, DatasetFeatureRepresentationMap datasetFeatureRepresentationMap, PerAlgorithmRegressionDatasetGenerator datasetGenerator,
			Classifier nonTrainedRegressionFunction) {
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
		this.datasetFeatureRepresentationMap = datasetFeatureRepresentationMap;
		this.datasetGenerator = datasetGenerator;
		this.nonTrainedRegressionFunction = nonTrainedRegressionFunction;
	}

	@Override
	public void initialize(long randomSeed) {
		this.datasetGenerator.initialize(randomSeed);
	}

	@Override
	public void train(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		algorithmToTrainedRegressionFunctionMap = new HashMap<>();
		List<Pair<Integer, Instances>> algorithmDatasetPairs = datasetGenerator.generateTrainingDataset(trainingDatasetIds, trainingPipelineIds);
		attributeInfo = datasetGenerator.getAttributeInfo();
		for (Integer algorithmId : pipelinePerformanceStorage.getPipelineIds()) {
			try {
				Instances instances = algorithmDatasetPairs.stream().filter(p -> p.getX().intValue() == algorithmId.intValue()).findFirst().get().getY();
				if (instances.size() > 0) {
					Classifier trainedRegressionFunction = AbstractClassifier.makeCopy(nonTrainedRegressionFunction);
					trainedRegressionFunction.buildClassifier(instances);
					algorithmToTrainedRegressionFunctionMap.put(algorithmId, trainedRegressionFunction);
					System.out.println("Trained regression function for algorithm " + algorithmId + " on dataset.");
				}
			} catch (Exception e) {
				throw new RuntimeException("Could not train classifier!", e);
			}
		}
		System.out.println("Trained all regression functions.");
	}

	@Override
	public List<Pair<Integer, Double>> getRankingOfPipelinesOnDataset(List<Integer> pipelineIdsToRank, int datasetId) {
		Instances dummyInstances = new Instances("dummy", attributeInfo, 0);
		dummyInstances.setClassIndex(dummyInstances.numAttributes() - 1);
		return pipelineIdsToRank.stream().map(id -> {
			try {
				Instance instance = createInstanceForDataset(datasetId);
				instance.setDataset(dummyInstances);
				dummyInstances.add(instance);
				return new Pair<>(id, getPredictedPerformanceForAlgorithmOnInstance(id, instance));
			} catch (Exception e) {
				throw new RuntimeException("Could not apply regression function.", e);
			}
		}).sorted(Comparator.comparingDouble(p -> ((Pair<Integer, Double>) p).getY()).reversed()).collect(Collectors.toList());
	}

	private double getPredictedPerformanceForAlgorithmOnInstance(Integer id, Instance instance) throws Exception {
		if (algorithmToTrainedRegressionFunctionMap.containsKey(id)) {
			return algorithmToTrainedRegressionFunctionMap.get(id).classifyInstance(instance);
		}
		return 0;
	}

	private Instance createInstanceForDataset(int trainingDatasetId) {
		double[] datasetFeatures = datasetFeatureRepresentationMap.getFeatureRepresentationForDataset(trainingDatasetId);

		int numberOfFeatures = datasetFeatures.length + 1;

		Instance instance = new DenseInstance(numberOfFeatures);
		int counter = 0;
		while (counter < datasetFeatures.length) {
			instance.setValue(counter, datasetFeatures[counter]);
			counter++;
		}
		// set dummy target value
		instance.setValue(numberOfFeatures - 1, 0);
		return instance;
	}

	@Override
	public String getName() {
		return "per_algorithm_regression_" + datasetGenerator.getName();
	}

}
