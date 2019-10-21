package de.upb.isml.tornede.ecai2020.experiments;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.math.linearalgebra.DenseDoubleVector;
import ai.libs.jaicore.ml.dyadranking.Dyad;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import ai.libs.jaicore.ml.dyadranking.loss.KendallsTauDyadRankingLoss;
import de.upb.isml.tornede.ecai2020.experiments.loss.KendallsTauBasedOnApache;
import de.upb.isml.tornede.ecai2020.experiments.loss.KendallsTauCorrelation;

public class PipelineFeatureRepresentationMapTest {

	public static void main(String[] args) throws SQLException {

		KendallsTauBasedOnApache kA = new KendallsTauBasedOnApache();

		KendallsTauCorrelation kC = new KendallsTauCorrelation();

		KendallsTauDyadRankingLoss rL = new KendallsTauDyadRankingLoss();

		while (true) {
			List<Integer> list = IntStream.range(0, 10).boxed().collect(Collectors.toList());

			List<Integer> list2 = new ArrayList<>(list);
			Collections.shuffle(list2);

			List<Pair<Integer, Double>> listPairs = list.stream().map(i -> new Pair<>(i, 0d)).collect(Collectors.toList());
			List<Pair<Integer, Double>> list2Pairs = list2.stream().map(i -> new Pair<>(i, 0d)).collect(Collectors.toList());

			DyadRankingInstance instance1 = new DyadRankingInstance(list.stream().map(i -> new Dyad(new DenseDoubleVector(1), new DenseDoubleVector(new double[] { i }))).collect(Collectors.toList()));
			DyadRankingInstance instance2 = new DyadRankingInstance(list2.stream().map(i -> new Dyad(new DenseDoubleVector(1), new DenseDoubleVector(new double[] { i }))).collect(Collectors.toList()));

			System.out.println("apache: " + kA.evaluate(listPairs, list2Pairs) + " - own: " + kC.evaluate(listPairs, list2Pairs) + " -pg: " + rL.loss(instance1, instance2));

		}

	}

}
