package de.upb.isml.tornede.ecai2020.experiments.datasetgen.algorithm;

import java.util.Map;
import java.util.Random;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.mlplan.multiclass.wekamlplan.weka.WekaPipelineFactory;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class AlgorithmEvaluator {

	private Random random;
	private Map<Integer, Instances> datasetIdToInstancesMap;

	public AlgorithmEvaluator(long randomSeed, Map<Integer, Instances> datasetIdToInstancesMap) {
		this.random = new Random(randomSeed);
		this.datasetIdToInstancesMap = datasetIdToInstancesMap;
	}

	public double evaluateAlgorithm(ComponentInstance componentInstance, int datasetId) throws Exception {
		Instances instances = datasetIdToInstancesMap.get(datasetId);

		WekaPipelineFactory factory = new WekaPipelineFactory();
		Classifier classifier = factory.getComponentInstantiation(componentInstance);

		Evaluation evaluation = new Evaluation(instances);
		evaluation.crossValidateModel(classifier, instances, 5, random);
		return evaluation.pctCorrect() / 100.0;
	}

}
