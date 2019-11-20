package de.upb.isml.tornede.ecai2020.experiments.rankers.regression;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.libs.hasco.serialization.ComponentLoader;
import de.upb.isml.tornede.ecai2020.experiments.datasetgen.algorithm.AlgorithmMetafeatureGenerator;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;

public abstract class AbstractRegressionDatasetGenerator implements RegressionDatasetGenerator {

	protected boolean oldDataset;
	protected PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap;
	protected DatasetFeatureRepresentationMap datasetFeatureRepresentationMap;
	protected PipelinePerformanceStorage pipelinePerformanceStorage;

	public AbstractRegressionDatasetGenerator(boolean oldDataset, PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap, DatasetFeatureRepresentationMap datasetFeatureRepresentationMap,
			PipelinePerformanceStorage pipelinePerformanceStorage) {
		this.oldDataset = oldDataset;
		this.pipelineFeatureRepresentationMap = pipelineFeatureRepresentationMap;
		this.datasetFeatureRepresentationMap = datasetFeatureRepresentationMap;
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
	}

	protected Instance createInstanceForPipelineAndDataset(int pipelineId, int trainingDatasetId) {
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

	protected List<Attribute> createDatasetAttributeList() {
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

	protected List<Attribute> createAlgorithmAttributeList() {
		if (oldDataset) {
			return createPipelineAttributeList();
		}
		try {
			ComponentLoader componenLoader = new ComponentLoader(new File("components/weka-singlelabel-base.json"));
			AlgorithmMetafeatureGenerator featureGenerator = new AlgorithmMetafeatureGenerator(componenLoader.getComponents());
			return featureGenerator.getWekaAttributeList();
		} catch (IOException e) {
			throw new RuntimeException("Cannot create algorithm attribute list!", e);
		}
	}
}
