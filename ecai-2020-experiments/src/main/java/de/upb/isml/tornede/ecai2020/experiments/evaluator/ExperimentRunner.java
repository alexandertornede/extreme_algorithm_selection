package de.upb.isml.tornede.ecai2020.experiments.evaluator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import ai.libs.jaicore.ml.tsc.distances.EuclideanDistance;
import de.upb.isml.tornede.ecai2020.experiments.loss.KendallsTauBasedOnApache;
import de.upb.isml.tornede.ecai2020.experiments.loss.KendallsTauBasedOnApacheAndRanks;
import de.upb.isml.tornede.ecai2020.experiments.loss.Metric;
import de.upb.isml.tornede.ecai2020.experiments.loss.PerformanceDifferenceOfAverageOnTopK;
import de.upb.isml.tornede.ecai2020.experiments.loss.PerformanceDifferenceOfBestOnTopK;
import de.upb.isml.tornede.ecai2020.experiments.rankers.AveragePerformanceRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.AverageRankBasedRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.DyadRankingBasedRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.IdBasedRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.KnnRanker;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import de.upb.isml.tornede.ecai2020.experiments.utils.Util;

public class ExperimentRunner {

	private static String databaseName = "conference_ecai2020";
	private static String tableName = "all_results_rank_all_with_values_with_smo";

	private static String pathToStoredRankingModels = "rankersMCCV";

	public static void main(String[] args) throws Exception {

		// int[] numberOfPairwiseSamplesPerDatasetSizes = { 100, 1000, 1900, 2750 };
		int[] numberOfPairwiseSamplesPerDatasetSizes = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2250, 2500, 2750 };

		int experimentNumber = 0;
		for (int datasetTestSplitId = 0; datasetTestSplitId < 10; datasetTestSplitId++) {
			Pair<List<Integer>, List<Integer>> trainTestSplit = Util.getTrainingAndTestDatasetSplitsForSplitId(datasetTestSplitId);

			List<Integer> trainindDatasetIds = trainTestSplit.getX();
			List<Integer> testDatasetIds = trainTestSplit.getY();

			SQLAdapter sqlAdapter = new SQLAdapter("isys-db.cs.upb.de:3306", "user", "password", databaseName, true);

			PipelineFeatureRepresentationMap pipelineFeatureMap = new PipelineFeatureRepresentationMap(sqlAdapter, "pipeline_feature_representations");
			DatasetFeatureRepresentationMap datasetFeatureMap = new DatasetFeatureRepresentationMap(sqlAdapter, "dataset_metafeatures");
			PipelinePerformanceStorage pipelinePerformanceStorage = new PipelinePerformanceStorage(sqlAdapter, "pipeline_evaluations");

			AveragePerformanceRanker averageRankRanker = new AveragePerformanceRanker(pipelinePerformanceStorage);
			KnnRanker onennRanker = new KnnRanker(pipelinePerformanceStorage, datasetFeatureMap, new EuclideanDistance(), 1);
			KnnRanker twonnRanker = new KnnRanker(pipelinePerformanceStorage, datasetFeatureMap, new EuclideanDistance(), 2);
			AverageRankBasedRanker averageRankBasedRanker = new AverageRankBasedRanker(pipelinePerformanceStorage);

			List<IdBasedRanker> rankers = new ArrayList<>(Arrays.asList(averageRankRanker, onennRanker, twonnRanker, averageRankBasedRanker));

			for (int numberOfPairwiseSamplesPerDataset : numberOfPairwiseSamplesPerDatasetSizes) {
				PLNetDyadRanker dyadRanker = getDyadRankerForNumberOfPairwiseSamples(numberOfPairwiseSamplesPerDataset, datasetTestSplitId);
				DyadRankingBasedRanker dyadRankingBasedRanker = new DyadRankingBasedRanker(numberOfPairwiseSamplesPerDataset, dyadRanker, pipelineFeatureMap, datasetFeatureMap);
				rankers.add(dyadRankingBasedRanker);
			}

			List<Metric> metrics = Arrays.asList(new KendallsTauBasedOnApache(), new KendallsTauBasedOnApacheAndRanks(), new PerformanceDifferenceOfAverageOnTopK(3), new PerformanceDifferenceOfBestOnTopK(3),
					new PerformanceDifferenceOfAverageOnTopK(5), new PerformanceDifferenceOfBestOnTopK(5));
			int numberOfTestPipelineSets = 1; // 1, since we rank all

			for (IdBasedRanker ranker : rankers) {
				Experiment experiment = new Experiment(pipelinePerformanceStorage, sqlAdapter, databaseName, tableName);
				experiment.runExperiment(datasetTestSplitId, trainindDatasetIds, testDatasetIds, ranker, metrics, numberOfTestPipelineSets);
				System.out.println("Experiment " + experimentNumber + " / " + (10 * rankers.size()) + " done.");
				experimentNumber++;
			}

			sqlAdapter.close();
		}

	}

	private static PLNetDyadRanker getDyadRankerForNumberOfPairwiseSamples(int numberOfPairwiseSamplesPerDataset, int datasetTestSplitId) throws IOException {
		PLNetDyadRanker dyadRanker = new PLNetDyadRanker();
		String filePath = pathToStoredRankingModels + "/ranker_" + numberOfPairwiseSamplesPerDataset + "_" + datasetTestSplitId + ".zip";
		// System.out.println("Loading: " + filePath);
		dyadRanker.loadModelFromFile(filePath);
		return dyadRanker;
	}

}
