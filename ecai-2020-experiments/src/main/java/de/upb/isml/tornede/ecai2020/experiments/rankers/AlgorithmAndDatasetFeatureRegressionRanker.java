package de.upb.isml.tornede.ecai2020.experiments.rankers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class AlgorithmAndDatasetFeatureRegressionRanker extends NonRandomIdBasedRanker {

	private PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap;
	private DatasetFeatureRepresentationMap datasetFeatureRepresentationMap;
	private PipelinePerformanceStorage pipelinePerformanceStorage;

	private Classifier nonTrainedRegressionFunction;
	private Classifier trainedRegressionFunction;

	private ArrayList<Attribute> attributeInfo;

	public AlgorithmAndDatasetFeatureRegressionRanker(PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap, DatasetFeatureRepresentationMap datasetFeatureRepresentationMap, PipelinePerformanceStorage pipelinePerformanceStorage,
			Classifier regressionFunction) {
		this.pipelineFeatureRepresentationMap = pipelineFeatureRepresentationMap;
		this.datasetFeatureRepresentationMap = datasetFeatureRepresentationMap;
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
		this.nonTrainedRegressionFunction = regressionFunction;
	}

	@Override
	public void train(List<Integer> trainingDatasetIds) {
		List<Attribute> datasetFeatureAttributes = createDatasetAttributeList();
		List<Attribute> pipelineFeatureAttributes = createPipelineAttributeList();
		Attribute targetAttribute = new Attribute("performance");
		attributeInfo = new ArrayList<>();
		attributeInfo.addAll(datasetFeatureAttributes);
		attributeInfo.addAll(pipelineFeatureAttributes);
		attributeInfo.add(targetAttribute);

		Instances instances = new Instances("dataset", attributeInfo, 0);
		instances.setClassIndex(instances.numAttributes() - 1);

		for (int pipelineId : pipelinePerformanceStorage.getPipelineIds()) {
			for (int trainingDatasetId : trainingDatasetIds) {
				Instance instance = createInstanceForPipelineAndDataset(pipelineId, trainingDatasetId);
				instance.setDataset(instances);
				instances.add(instance);
			}
		}
		System.out.println("Generated dataset with " + instances.size() + " instances.");

		try {
			trainedRegressionFunction = AbstractClassifier.makeCopy(nonTrainedRegressionFunction);
			trainedRegressionFunction.buildClassifier(instances);
			System.out.println("Trained regression function on dataset.");
		} catch (Exception e) {
			throw new RuntimeException("Could not train classifier!", e);
		}
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

	private List<Attribute> createDatasetAttributeList() {
		List<Attribute> attributes = new ArrayList<>();
		for (int i = 0; i < 45; i++) {
			attributes.add(new Attribute("d" + i));
		}
		return attributes;
	}

	private List<Attribute> createPipelineAttributeList() {
		List<Attribute> attributes = new ArrayList<>();
		int counter;
		// 0-24: 0/1
		for (counter = 0; counter < 25; counter++) {
			attributes.add(new Attribute(counter + "", Arrays.asList("0.0", "1.0")));
		}
		// 25: numeric
		attributes.add(new Attribute("" + counter));
		counter++;
		// 26-36: 0/1
		while (counter < 37) {
			attributes.add(new Attribute(counter + "", Arrays.asList("0.0", "1.0")));
			counter++;
		}
		// 37: numeric
		attributes.add(new Attribute("" + counter));
		counter++;
		// 38-42: 0/1
		while (counter < 43) {
			attributes.add(new Attribute(counter + "", Arrays.asList("0.0", "1.0")));
			counter++;
		}
		// 43: numeric
		attributes.add(new Attribute("" + counter));
		counter++;
		// 44: numeric
		attributes.add(new Attribute("" + counter));
		counter++;
		// 45: numeric
		attributes.add(new Attribute("" + counter));
		counter++;
		// 46-58: 0/1
		while (counter < 59) {
			attributes.add(new Attribute(counter + "", Arrays.asList("0.0", "1.0")));
			counter++;
		}
		// 59: numeric
		attributes.add(new Attribute("" + counter));
		counter++;
		// 60-61: 0/1
		while (counter < 62) {
			attributes.add(new Attribute(counter + "", Arrays.asList("0.0", "1.0")));
			counter++;
		}
		// 62: numeric
		attributes.add(new Attribute("" + counter));
		counter++;
		// 63-70: 0/1
		while (counter < 71) {
			attributes.add(new Attribute(counter + "", Arrays.asList("0.0", "1.0")));
			counter++;
		}
		// 71: numeric
		attributes.add(new Attribute("" + counter));
		counter++;
		// 72:numeric
		attributes.add(new Attribute("" + counter));
		counter++;
		// 73-75: 0/1
		while (counter < 76) {
			attributes.add(new Attribute(counter + "", Arrays.asList("0.0", "1.0")));
			counter++;
		}
		// 76: numeric
		attributes.add(new Attribute("" + counter));
		counter++;
		// 77: 0/1
		attributes.add(new Attribute(counter + "", Arrays.asList("0.0", "1.0")));
		counter++;
		// 78: numeric
		attributes.add(new Attribute("" + counter));
		counter++;
		// 79-95: 0/1
		while (counter < 96) {
			attributes.add(new Attribute(counter + "", Arrays.asList("0.0", "1.0")));
			counter++;
		}
		return attributes;
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

	@Override
	public String getName() {
		return "2xfeature_regression";
	}

}
