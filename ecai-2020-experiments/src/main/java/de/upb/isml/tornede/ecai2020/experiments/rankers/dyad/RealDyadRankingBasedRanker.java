package de.upb.isml.tornede.ecai2020.experiments.rankers.dyad;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.math.linearalgebra.DenseDoubleVector;
import ai.libs.jaicore.ml.core.exception.PredictionException;
import ai.libs.jaicore.ml.core.exception.TrainingException;
import ai.libs.jaicore.ml.dyadranking.Dyad;
import ai.libs.jaicore.ml.dyadranking.algorithm.IPLNetDyadRankerConfiguration;
import ai.libs.jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import ai.libs.jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import de.upb.isml.tornede.ecai2020.experiments.rankers.IdBasedRanker;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;

public class RealDyadRankingBasedRanker implements IdBasedRanker {

	private PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap;
	private DatasetFeatureRepresentationMap datasetFeatureRepresentationMap;

	private DyadRankingTrainingDatasetGenerator trainingDatasetGenerator;

	private PLNetDyadRanker dyadRanker;

	public RealDyadRankingBasedRanker(PipelineFeatureRepresentationMap pipelineFeatureRepresentationMap, DatasetFeatureRepresentationMap datasetFeatureRepresentationMap, DyadRankingTrainingDatasetGenerator trainingDatasetGenerator) {
		this.pipelineFeatureRepresentationMap = pipelineFeatureRepresentationMap;
		this.datasetFeatureRepresentationMap = datasetFeatureRepresentationMap;
		this.trainingDatasetGenerator = trainingDatasetGenerator;
	}

	@Override
	public void initialize(long randomSeed) {
		trainingDatasetGenerator.initialize(randomSeed);
	}

	@Override
	public void train(List<Integer> trainingDatasetIds) {
		dyadRanker = new PLNetDyadRanker(ConfigFactory.create(IPLNetDyadRankerConfiguration.class));
		System.out.println("Training dyad ranker with config: " + dyadRanker.getConfiguration());

		DyadRankingDataset dataset = trainingDatasetGenerator.generateTrainingDataset(trainingDatasetIds);
		// System.out.println("Start training of PLNET with " + dataset.size() + " rankings.");
		try {
			dyadRanker.train(dataset);
		} catch (TrainingException e) {
			System.err.println("Could not train dyad ranker: " + e);
		}
		// System.out.println("Finished training of PLNET");

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
		return "real_dyad_ranker_" + trainingDatasetGenerator.getName();
	}

}
