package de.upb.isml.tornede.ecai2020.experiments.loss;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.util.FastMath;

import ai.libs.jaicore.basic.sets.Pair;

public class NormalizedDiscountedCumulativeGain implements Metric {

	// truncation threshold, if < 0 then unlimited
	private int k;

	public NormalizedDiscountedCumulativeGain(int k) {
		this.k = k;
	}

	@Override
	public double evaluate(List<Pair<Integer, Double>> groundTruth, List<Pair<Integer, Double>> predicted) {
		Map<Integer, Double> itemToYScoresMap = new HashMap<>();
		for (int i = 0; i < groundTruth.size(); i++) {
			Pair<Integer, Double> element = groundTruth.get(i);
			itemToYScoresMap.put(element.getX(), element.getY());
		}
		double result = computeDiscountedCumulativeGain(predicted, itemToYScoresMap) / computeDiscountedCumulativeGain(groundTruth, itemToYScoresMap);
		return result;
	}

	public double computeDiscountedCumulativeGain(List<Pair<Integer, Double>> ranking, Map<Integer, Double> itemToYScoresMap) {
		double dcgSum = 0;
		int upperBound = k;
		if (k <= 0) {
			upperBound = ranking.size();
		}
		for (int i = 0; i < upperBound; i++) {
			double numerator = Math.pow(2, itemToYScoresMap.get(ranking.get(i).getX())) - 1;
			double denominator = FastMath.log(2, (i + 1) + 2);
			dcgSum += numerator / denominator;
		}
		return dcgSum;
	}

	@Override
	public String getName() {
		return "NDCG@" + k;
	}

}
