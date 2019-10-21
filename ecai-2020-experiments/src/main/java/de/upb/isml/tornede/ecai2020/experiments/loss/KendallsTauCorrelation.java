package de.upb.isml.tornede.ecai2020.experiments.loss;

import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;

public class KendallsTauCorrelation implements Metric {

	@Override
	public double evaluate(List<Pair<Integer, Double>> groundTruth, List<Pair<Integer, Double>> predicted) {
		int numberOfConcordantPairs = 0;
		int numberOfDiscordantPairs = 0;

		List<Integer> groundTruthRanking = groundTruth.stream().map(p -> p.getX()).collect(Collectors.toList());
		List<Integer> predictedRanking = predicted.stream().map(p -> p.getX()).collect(Collectors.toList());

		for (Integer pipeline1 : groundTruthRanking) {
			for (Integer pipeline2 : groundTruthRanking) {
				if (groundTruthRanking.indexOf(pipeline1) < groundTruthRanking.indexOf(pipeline2)) {
					if (predictedRanking.indexOf(pipeline1) < predictedRanking.indexOf(pipeline2)) {
						numberOfConcordantPairs++;
					} else {
						numberOfDiscordantPairs++;
					}
				} else if (groundTruthRanking.indexOf(pipeline1) > groundTruthRanking.indexOf(pipeline2)) {
					if (predictedRanking.indexOf(pipeline1) > predictedRanking.indexOf(pipeline2)) {
						numberOfConcordantPairs++;
					} else {
						numberOfDiscordantPairs++;
					}
				}
			}
		}
		return (numberOfConcordantPairs - numberOfDiscordantPairs) / (double) (groundTruthRanking.size() * groundTruthRanking.size());
	}

	@Override
	public String getName() {
		return "kendall";
	}

}
