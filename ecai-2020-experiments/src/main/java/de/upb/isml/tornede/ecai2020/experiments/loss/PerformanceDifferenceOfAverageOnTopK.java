package de.upb.isml.tornede.ecai2020.experiments.loss;

import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;

public class PerformanceDifferenceOfAverageOnTopK implements Metric {

	private int k;

	public PerformanceDifferenceOfAverageOnTopK(int k) {
		this.k = k;
	}

	@Override
	public double evaluate(List<Pair<Integer, Double>> groundTruth, List<Pair<Integer, Double>> predicted) {

		List<Pair<Integer, Double>> groundTruthTopK = groundTruth.stream().limit(k).collect(Collectors.toList());
		double averagePerformanceGroundTruth = groundTruthTopK.stream().mapToDouble(p -> p.getY().doubleValue()).average().getAsDouble();

		List<Pair<Integer, Double>> predictedTopKMappedToTruePerformances = predicted.stream().limit(k).map(p -> new Pair<>(p.getX(), extractPerformanceForPipelineFromGroundTruth(groundTruth, p.getX()))).collect(Collectors.toList());
		double averagePerformancePredicted = predictedTopKMappedToTruePerformances.stream().mapToDouble(p -> p.getY().doubleValue()).average().getAsDouble();

		return averagePerformanceGroundTruth - averagePerformancePredicted;
	}

	private double extractPerformanceForPipelineFromGroundTruth(List<Pair<Integer, Double>> groundTruth, int pipelineId) {
		return groundTruth.stream().filter(p -> p.getX().intValue() == pipelineId).findFirst().get().getY();
	}

	@Override
	public String getName() {
		return "avg_top_" + k;
	}

}
