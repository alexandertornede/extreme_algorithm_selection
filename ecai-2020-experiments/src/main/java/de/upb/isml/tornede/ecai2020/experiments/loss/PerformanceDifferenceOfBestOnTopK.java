package de.upb.isml.tornede.ecai2020.experiments.loss;

import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;

public class PerformanceDifferenceOfBestOnTopK implements Metric {

	private int k;

	public PerformanceDifferenceOfBestOnTopK(int k) {
		this.k = k;
	}

	@Override
	public double evaluate(List<Pair<Integer, Double>> groundTruth, List<Pair<Integer, Double>> predicted) {

		List<Pair<Integer, Double>> groundTruthTopK = groundTruth.stream().limit(k).collect(Collectors.toList());
		double bestPerformanceGroundTruth = groundTruthTopK.stream().mapToDouble(p -> p.getY().doubleValue()).max().getAsDouble();

		List<Pair<Integer, Double>> predictedTopKMappedToTruePerformances = predicted.stream().limit(k).map(p -> new Pair<>(p.getX(), extractPerformanceForPipelineFromGroundTruth(groundTruth, p.getX()))).collect(Collectors.toList());
		double bestPerformancePredicted = predictedTopKMappedToTruePerformances.stream().mapToDouble(p -> p.getY().doubleValue()).max().getAsDouble();

		return bestPerformanceGroundTruth - bestPerformancePredicted;
	}

	private double extractPerformanceForPipelineFromGroundTruth(List<Pair<Integer, Double>> groundTruth, int pipelineId) {
		return groundTruth.stream().filter(p -> p.getX().intValue() == pipelineId).findFirst().get().getY();
	}

	@Override
	public String getName() {
		return "best_top_" + k;
	}
}
