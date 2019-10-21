package de.upb.isml.tornede.ecai2020.experiments.loss;

import java.util.List;

import ai.libs.jaicore.basic.sets.Pair;

public interface Metric {

	public double evaluate(List<Pair<Integer, Double>> groundTruth, List<Pair<Integer, Double>> predicted);

	public String getName();
}
