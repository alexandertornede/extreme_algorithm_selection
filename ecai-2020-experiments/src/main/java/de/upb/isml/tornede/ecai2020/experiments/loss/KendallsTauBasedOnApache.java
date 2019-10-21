package de.upb.isml.tornede.ecai2020.experiments.loss;

import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.correlation.KendallsCorrelation;

import ai.libs.jaicore.basic.sets.Pair;

public class KendallsTauBasedOnApache implements Metric {

	@Override
	public double evaluate(List<Pair<Integer, Double>> groundTruth, List<Pair<Integer, Double>> predicted) {
		KendallsCorrelation kC = new KendallsCorrelation();
		double[] groundTruthArray = new double[groundTruth.size()];
		double[] predictedArray = new double[predicted.size()];

		List<Integer> pipelineIds = groundTruth.stream().map(p -> p.getX()).collect(Collectors.toList());

		for (int i = 0; i < groundTruth.size(); i++) {
			Pair<Integer, Double> pair = groundTruth.get(i);
			groundTruthArray[pipelineIds.indexOf(pair.getX())] = pair.getY();
		}

		for (int i = 0; i < predicted.size(); i++) {
			Pair<Integer, Double> pair = predicted.get(i);
			predictedArray[pipelineIds.indexOf(pair.getX())] = pair.getY();
		}

		return kC.correlation(groundTruthArray, predictedArray);
	}

	@Override
	public String getName() {
		return "kendalls_apache";
	}

}
