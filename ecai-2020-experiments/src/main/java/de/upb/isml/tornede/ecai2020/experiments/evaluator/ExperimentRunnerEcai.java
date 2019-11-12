package de.upb.isml.tornede.ecai2020.experiments.evaluator;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.ml.tsc.distances.EuclideanDistance;
import de.upb.isml.tornede.ecai2020.experiments.loss.KendallsTauBasedOnApache;
import de.upb.isml.tornede.ecai2020.experiments.loss.KendallsTauBasedOnApacheAndRanks;
import de.upb.isml.tornede.ecai2020.experiments.loss.Metric;
import de.upb.isml.tornede.ecai2020.experiments.loss.NormalizedDiscountedCumulativeGain;
import de.upb.isml.tornede.ecai2020.experiments.loss.PerformanceDifferenceOfAverageOnTopK;
import de.upb.isml.tornede.ecai2020.experiments.loss.PerformanceDifferenceOfBestOnTopK;
import de.upb.isml.tornede.ecai2020.experiments.rankers.AveragePerformanceRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.AverageRankBasedRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.IdBasedRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.KnnRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.OracleRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.RandomRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.dyad.DyadRankingTrainingDatasetGenerator;
import de.upb.isml.tornede.ecai2020.experiments.rankers.dyad.RandomDyadRankingDatasetGenerator;
import de.upb.isml.tornede.ecai2020.experiments.rankers.dyad.RealDyadRankingBasedRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.regression.AlgorithmAndDatasetFeatureRegressionRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.regression.RandomRegressionDatasetGenerator;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import weka.classifiers.trees.RandomForest;

public class ExperimentRunnerEcai {

	public static void main(String[] args) throws Exception {
		// int[] datasetIds = new int[] { 44, 1504, 312, 1467, 50, 31, 1046, 1493, 151, 334, 1491, 1471, 1067, 1038, 1480, 1494, 1050, 1464, 1063, 333, 1510, 37, 1489, 335, 1462, 1487, 1049, 1068, 3 }; // old dataset

		ExperimentRunnerEcaiConfig config = ConfigFactory.create(ExperimentRunnerEcaiConfig.class);
		System.out.println(config);

		String databaseName = config.getDBDatabaseName();
		String tableName = config.getDBTableName();
		String dbUser = config.getDBUsername();
		String dbPassword = config.getDBPassword();
		String dbHost = config.getDBHost();

		// int[] datasetIds = new int[] { 3, 40966, 6, 11, 12, 14, 15, 40975, 16, 18, 40978, 40979, 40982, 22, 23, 40983, 151, 40984, 1049, 1050, 28, 29, 1053, 31, 32, 40994, 37, 38, 4134, 1063, 1067, 1068, 44, 300, 46, 50, 307, 40499,
		// 1461,
		// 1462, 54, 182, 4534, 1590, 1464, 4538, 188, 6332, 1468, 1475, 41027, 1478, 1480, 458, 1485, 1486, 1487, 1489, 23381, 469, 1494, 1497, 40668, 1501, 23517, 40670, 1510, 40701 }; // new dataset

		List<Integer> datasetIds = config.getDatasetIds();
		int fold = Integer.parseInt(args[0]);

		int experimentNumber = 0;
		Pair<List<Integer>, List<Integer>> trainTestSplit = getTrainTestSplitForFold(10, fold, datasetIds);

		System.out.println(getCurrentTimestep() + " :: Running experiments for fold " + fold);

		List<Integer> trainindDatasetIds = trainTestSplit.getX();
		System.out.println("Training on:" + trainindDatasetIds);
		List<Integer> testDatasetIds = trainTestSplit.getY();
		System.out.println("Testing on:" + testDatasetIds);

		SQLAdapter sqlAdapter = new SQLAdapter(dbHost, dbUser, dbPassword, databaseName, true);

		PipelineFeatureRepresentationMap pipelineFeatureMap = new PipelineFeatureRepresentationMap(sqlAdapter, "algorithm_metafeatures"); // pipeline_feature_representations
		DatasetFeatureRepresentationMap datasetFeatureMap = new DatasetFeatureRepresentationMap(sqlAdapter, "dataset_metafeatures_new"); // dataset_metafeatures
		PipelinePerformanceStorage pipelinePerformanceStorage = new PipelinePerformanceStorage(sqlAdapter, "algorithm_evaluations_with_timeouts"); // pipeline_evaluations

		List<IdBasedRanker> rankers = new ArrayList<>();

		// add own ranker variants
		int[] numberOfRankingsPerTrainingDataset = new int[] { 25, 50, 125, 250, 500, 1000 };
		int[] lengthOfRankingsForDyadRanker = new int[] { 2, 3 };
		int numberOfRankingsToTest = 100;
		int numberOfPipelinesPerTestRanking = 10;

		for (int l : lengthOfRankingsForDyadRanker) {
			for (int n : numberOfRankingsPerTrainingDataset) {
				// DyadRankingImitatingRegressionDatasetGenerator regressionDatasetGenerator = new DyadRankingImitatingRegressionDatasetGenerator(pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
				// AlgorithmAndDatasetFeatureRegressionRanker algorithmAndDatasetRegressionRanker = new AlgorithmAndDatasetFeatureRegressionRanker(pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, new RandomForest(),
				// regressionDatasetGenerator);
				// rankers.add(algorithmAndDatasetRegressionRanker);

				DyadRankingTrainingDatasetGenerator datasetGenerator = new RandomDyadRankingDatasetGenerator(pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
				RealDyadRankingBasedRanker randomlyTrainedDyadRanker = new RealDyadRankingBasedRanker(pipelineFeatureMap, datasetFeatureMap, datasetGenerator);
				rankers.add(randomlyTrainedDyadRanker);
			}
		}

		// add baselines
		AlgorithmAndDatasetFeatureRegressionRanker algorithmAndDatasetRegressionRankerFull = new AlgorithmAndDatasetFeatureRegressionRanker(pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, new RandomForest(),
				new RandomRegressionDatasetGenerator(false, pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, -1));
		AveragePerformanceRanker averagePerformanceRanker = new AveragePerformanceRanker(pipelinePerformanceStorage);
		KnnRanker onennRanker = new KnnRanker(pipelinePerformanceStorage, datasetFeatureMap, new EuclideanDistance(), 1);
		KnnRanker twonnRanker = new KnnRanker(pipelinePerformanceStorage, datasetFeatureMap, new EuclideanDistance(), 2);
		AverageRankBasedRanker averageRankBasedRanker = new AverageRankBasedRanker(pipelinePerformanceStorage);
		RandomRanker randomRanker = new RandomRanker();
		OracleRanker oracleRanker = new OracleRanker(pipelinePerformanceStorage);
		rankers.addAll(Arrays.asList(algorithmAndDatasetRegressionRankerFull, averageRankBasedRanker, onennRanker, twonnRanker, randomRanker, averagePerformanceRanker));
		rankers.clear();
		rankers.add(algorithmAndDatasetRegressionRankerFull);

		List<Metric> metrics = Arrays.asList(new NormalizedDiscountedCumulativeGain(3), new NormalizedDiscountedCumulativeGain(5), new NormalizedDiscountedCumulativeGain(10), new KendallsTauBasedOnApache(),
				new KendallsTauBasedOnApacheAndRanks(), new PerformanceDifferenceOfAverageOnTopK(3), new PerformanceDifferenceOfBestOnTopK(3), new PerformanceDifferenceOfAverageOnTopK(5), new PerformanceDifferenceOfBestOnTopK(5));
		System.out.println(getCurrentTimestep() + " :: Running " + (10 * rankers.size()) + " experiments in total.");

		ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

		for (IdBasedRanker ranker : rankers) {
			Experiment experiment = new Experiment(pipelinePerformanceStorage, new SQLAdapter(dbHost, dbUser, dbPassword, databaseName, true), databaseName, tableName);
			ExperimentRunnable experimentRunnable = new ExperimentRunnable(experiment, experimentNumber, fold, trainindDatasetIds, testDatasetIds, ranker, metrics, numberOfRankingsToTest, numberOfPipelinesPerTestRanking);
			executorService.execute(experimentRunnable);
			experimentNumber++;
		}

		executorService.shutdown();
	}

	private static Pair<List<Integer>, List<Integer>> getTrainTestSplitForFold(int totalAmountOfFolds, int fold, List<Integer> datasets) {
		int numberOfTestDatasets = (int) Math.round(datasets.size() / (double) totalAmountOfFolds);

		List<Integer> testDatasets = new ArrayList<>();
		int offset = fold * numberOfTestDatasets;
		int upperBound = Math.min(numberOfTestDatasets, datasets.size() - offset);
		for (int i = 0; i < upperBound; i++) {
			testDatasets.add(datasets.get(offset + i));
		}
		List<Integer> trainingDatasets = datasets.stream().filter(d -> !testDatasets.contains(d)).collect(Collectors.toList());

		return new Pair<>(trainingDatasets, testDatasets);
	}

	// private static PLNetDyadRanker getDyadRankerForNumberOfPairwiseSamples(int numberOfPairwiseSamplesPerDataset, int datasetTestSplitId) throws IOException {
	// PLNetDyadRanker dyadRanker = new PLNetDyadRanker();
	// String filePath = pathToStoredRankingModels + "/ranker_" + numberOfPairwiseSamplesPerDataset + "_" + datasetTestSplitId + ".zip";
	// // System.out.println("Loading: " + filePath);
	// dyadRanker.loadModelFromFile(filePath);
	// return dyadRanker;
	// }

	private static String getCurrentTimestep() {
		SimpleDateFormat sdfDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");// dd/MM/yyyy
		Date now = new Date();
		String strDate = sdfDate.format(now);
		return strDate;
	}

}
