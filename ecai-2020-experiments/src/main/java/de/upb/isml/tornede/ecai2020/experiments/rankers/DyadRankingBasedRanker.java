package de.upb.isml.tornede.ecai2020.experiments.rankers;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.math.linearalgebra.DenseDoubleVector;
import ai.libs.jaicore.ml.core.exception.PredictionException;
import ai.libs.jaicore.ml.dyadranking.Dyad;
import ai.libs.jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import ai.libs.jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;

public class DyadRankingBasedRanker extends NonRandomIdBasedRanker {

	private int numberOfPairwiseComparisonsPerTrainingDataset;
	private PLNetDyadRanker dyadRanker;

	private PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap;
	private DatasetFeatureRepresentationMap datasetFeatureRepresentationMap;

	public DyadRankingBasedRanker(int numberOfPairwiseComparisonsPerTrainingDataset, PLNetDyadRanker dyadRanker, PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap,
			DatasetFeatureRepresentationMap datasetFeatureRepresentationMap) {
		this.numberOfPairwiseComparisonsPerTrainingDataset = numberOfPairwiseComparisonsPerTrainingDataset;
		this.dyadRanker = dyadRanker;
		this.pipelineFeatureRepresentationMap = pipelineFeatureRepresentationMap;
		this.datasetFeatureRepresentationMap = datasetFeatureRepresentationMap;
	}

	@Override
	public void train(List<Integer> trainingDatasetIds) {
		// nothing to do here as we assume the dyad ranker given in the constructor to be already trained
	}

	@Override
	public List<Pair<Integer, Double>> getRankingOfPipelinesOnDataset(List<Integer> pipelineIdsToRank, int datasetId) {

		double[] featureRepresentationForDataset = datasetFeatureRepresentationMap.getFeatureRepresentationForDataset(datasetId);
		List<Pair<Integer, Dyad>> pipelineIdDyadPairsToRank = pipelineIdsToRank.stream()
				.map(id -> new Pair<>(id, new Dyad(new DenseDoubleVector(featureRepresentationForDataset), new DenseDoubleVector(pipelineFeatureRepresentationMap.getFeatureRepresentationForPipeline(id))))).collect(Collectors.toList());
		List<Dyad> pipelineFeatureRepresentationsToRank = pipelineIdDyadPairsToRank.stream().map(p -> p.getY()).collect(Collectors.toList());

		IDyadRankingInstance instanceToPredictOn = new DyadRankingInstance(pipelineFeatureRepresentationsToRank);

		try {
			IDyadRankingInstance rankedInstance = dyadRanker.predict(instanceToPredictOn);
			List<Pair<Integer, Double>> dyadsInRankedOrder = new ArrayList<>(pipelineIdsToRank.size());
			for (Dyad dyad : rankedInstance) {
				int pipelineId = pipelineIdDyadPairsToRank.stream().filter(d -> d.getY().equals(dyad)).findFirst().get().getX();
				dyadsInRankedOrder.add(new Pair<>(pipelineId, dyadRanker.getSkillForDyad(dyad)));
			}
			// System.out.println("dyad_ranking: " + dyadsInRankedOrder);
			return dyadsInRankedOrder;
		} catch (PredictionException e) {
			throw new RuntimeException("Encountered an error during prediction using the dyad ranker.", e);
		}
	}

	@Override
	public String getName() {
		return "dyad_" + numberOfPairwiseComparisonsPerTrainingDataset;
	}

}
