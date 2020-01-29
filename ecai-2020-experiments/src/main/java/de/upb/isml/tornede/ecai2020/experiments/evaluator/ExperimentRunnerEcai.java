package de.upb.isml.tornede.ecai2020.experiments.evaluator;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.ml.tsc.distances.EuclideanDistance;
import de.upb.isml.tornede.ecai2020.experiments.loss.KendallsTauBasedOnApacheAndRanks;
import de.upb.isml.tornede.ecai2020.experiments.loss.Metric;
import de.upb.isml.tornede.ecai2020.experiments.loss.NormalizedDiscountedCumulativeGain;
import de.upb.isml.tornede.ecai2020.experiments.loss.PerformanceDifferenceOfAverageOnTopK;
import de.upb.isml.tornede.ecai2020.experiments.loss.PerformanceDifferenceOfBestOnTopK;
import de.upb.isml.tornede.ecai2020.experiments.rankers.AlorsBasedRanker;
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
import de.upb.isml.tornede.ecai2020.experiments.rankers.regression.DyadRankingImitatingRegressionDatasetGenerator;
import de.upb.isml.tornede.ecai2020.experiments.rankers.regression.RegressionDatasetGenerator;
import de.upb.isml.tornede.ecai2020.experiments.rankers.regression.peralgorithm.PerAlgorithmDyadRankingImitatingDatasetGenerator;
import de.upb.isml.tornede.ecai2020.experiments.rankers.regression.peralgorithm.PerAlgorithmRegressionRanker;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import weka.classifiers.trees.RandomForest;

public class ExperimentRunnerEcai {

	public static void main(String[] args) throws Exception {

		ExperimentRunnerEcaiConfig config = ConfigFactory.create(ExperimentRunnerEcaiConfig.class);
		System.out.println(config);

		String databaseName = config.getDBDatabaseName();
		String tableName = config.getDBTableName();
		String dbUser = config.getDBUsername();
		String dbPassword = config.getDBPassword();
		String dbHost = config.getDBHost();

		List<String> approaches = config.getApproaches();
		System.out.println("Executing experiments for approaches: " + approaches);

		int numberOfCPUs = config.getAmountOfCPUsToUse();
		if (numberOfCPUs <= 0) {
			numberOfCPUs = Runtime.getRuntime().availableProcessors();
		}

		boolean foldsOnDatasets = config.foldsOnDatasets();
		boolean foldsOnAlgorithms = config.foldsOnAlgorithms();

		if (!foldsOnDatasets && !foldsOnAlgorithms) {
			System.out.println("Training with no folds at all makes no sense. Either folds on datasets, algorithms or both.");
			System.exit(0);
		}

		System.out.println("Folds on datasets: " + foldsOnDatasets);
		System.out.println("Folds on algorithms: " + foldsOnAlgorithms);

		List<Integer> datasetIds = config.getDatasetIds();

		for (int fold = 0; fold < 10; fold++) {
			// int fold = Integer.parseInt(args[0]); // TODO

			int experimentNumber = 0;
			Pair<List<Integer>, List<Integer>> datasetTrainTestSplit = getTrainTestSplitForFold(10, fold, datasetIds);
			System.out.println(getCurrentTimestep() + " :: Running experiments for fold " + fold);

			List<Integer> trainingDatasetIds = datasetTrainTestSplit.getX();
			List<Integer> testDatasetIds = datasetTrainTestSplit.getY();

			if (!foldsOnDatasets) {
				// if we do not want to fold on the datasets, we need to have the same training and test set with respect to the datasets
				trainingDatasetIds.addAll(testDatasetIds);
				testDatasetIds.addAll(trainingDatasetIds);
				System.out.println("Training and testing on identical datasets.");
			}
			System.out.println("Training on datasets: " + trainingDatasetIds);
			System.out.println("Testing on datasets: " + testDatasetIds);

			SQLAdapter sqlAdapter = new SQLAdapter(dbHost, dbUser, dbPassword, databaseName, true);

			PipelineFeatureRepresentationMap pipelineFeatureMap = new PipelineFeatureRepresentationMap(sqlAdapter, "algorithm_metafeatures");
			DatasetFeatureRepresentationMap datasetFeatureMap = new DatasetFeatureRepresentationMap(sqlAdapter, "dataset_metafeatures_new");
			PipelinePerformanceStorage pipelinePerformanceStorage = new PipelinePerformanceStorage(sqlAdapter, "algorithm_evaluations_with_timeouts");

			List<Integer> allPipelineIds = new ArrayList<>(pipelinePerformanceStorage.getPipelineIds());
			Collections.shuffle(allPipelineIds, new Random(1));
			Pair<List<Integer>, List<Integer>> algorithmTrainTestSplit = getTrainTestSplitForFold(10, fold, allPipelineIds);

			List<Integer> trainingAlgorithmIds = algorithmTrainTestSplit.getX();
			List<Integer> testAlgorithmIds = algorithmTrainTestSplit.getY();

			if (!foldsOnAlgorithms) {
				// if we do not want to fold on the algorithms, we need to have the same training and test set with respect to the algorithms
				trainingAlgorithmIds.addAll(testAlgorithmIds);
				testAlgorithmIds.addAll(trainingAlgorithmIds);
				System.out.println("Training and testing on identical algorithm sets.");
			}
			System.out.println("Training on algorithms: " + trainingAlgorithmIds);
			System.out.println("Testing on algorithms: " + testAlgorithmIds);

			List<IdBasedRanker> rankers = new ArrayList<>();

			int[] numberOfRankingsPerTrainingDataset = new int[] { 25, 50, 125 };
			int[] lengthOfRankingsForDyadRanker = new int[] { 2 };
			int numberOfRankingsToTest = 100;
			int numberOfPipelinesPerTestRanking = 10;

			for (int l : lengthOfRankingsForDyadRanker) {
				for (int n : numberOfRankingsPerTrainingDataset) {

					// add dyadic regression ranker trained on sparse training matrix
					if (approaches.contains("dyadic_feature_regression")) {
						RegressionDatasetGenerator datasetGenerator = new DyadRankingImitatingRegressionDatasetGenerator(false, pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
						AlgorithmAndDatasetFeatureRegressionRanker algorithmAndDatasetRegressionRanker = new AlgorithmAndDatasetFeatureRegressionRanker(pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, new RandomForest(),
								datasetGenerator);
						rankers.add(algorithmAndDatasetRegressionRanker);
					}

					// add per-algorithm-regression ranker trained on sparse matrix
					if (approaches.contains("per_algorithm_regression")) {
						PerAlgorithmRegressionRanker perAlgorithmRegressionRanker = new PerAlgorithmRegressionRanker(pipelinePerformanceStorage, datasetFeatureMap,
								new PerAlgorithmDyadRankingImitatingDatasetGenerator(datasetFeatureMap, pipelinePerformanceStorage, l, n), new RandomForest());
						rankers.add(perAlgorithmRegressionRanker);
					}

					// // add dyad ranking models
					if (approaches.contains("dyad_ranking")) {
						DyadRankingTrainingDatasetGenerator dyadDatasetGenerator = new RandomDyadRankingDatasetGenerator(pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
						RealDyadRankingBasedRanker randomlyTrainedDyadRanker = new RealDyadRankingBasedRanker(pipelineFeatureMap, datasetFeatureMap, dyadDatasetGenerator);
						rankers.add(randomlyTrainedDyadRanker);
					}

					// Alors Regression
					if (approaches.contains("alors_regression")) {
						RegressionDatasetGenerator datasetGeneratorAlorsRegression = new DyadRankingImitatingRegressionDatasetGenerator(false, pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
						rankers.add(new AlorsBasedRanker(datasetFeatureMap, pipelinePerformanceStorage, "REGRESSION", datasetGeneratorAlorsRegression));
					}

					// Alors NDCG
					if (approaches.contains("alors_ndcg")) {
						RegressionDatasetGenerator datasetGeneratorAlorsNDCG = new DyadRankingImitatingRegressionDatasetGenerator(false, pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
						rankers.add(new AlorsBasedRanker(datasetFeatureMap, pipelinePerformanceStorage, "NDCG", datasetGeneratorAlorsNDCG));
					}

					// average performance
					if (approaches.contains("average_performance")) {
						RegressionDatasetGenerator datasetGeneratorAveragePerformance = new DyadRankingImitatingRegressionDatasetGenerator(false, pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
						AveragePerformanceRanker averagePerformanceRanker = new AveragePerformanceRanker(pipelinePerformanceStorage, datasetGeneratorAveragePerformance, false);
						rankers.add(averagePerformanceRanker);
					}

					// average performance with bayes averaging
					if (approaches.contains("average_performance_bayesian_averaging")) {
						RegressionDatasetGenerator datasetGeneratorAveragePerformance = new DyadRankingImitatingRegressionDatasetGenerator(false, pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
						AveragePerformanceRanker averagePerformanceRanker = new AveragePerformanceRanker(pipelinePerformanceStorage, datasetGeneratorAveragePerformance, true);
						rankers.add(averagePerformanceRanker);
					}

					// 1-nn
					if (approaches.contains("1-nn")) {
						RegressionDatasetGenerator datasetGenerator1NNPerformance = new DyadRankingImitatingRegressionDatasetGenerator(false, pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
						KnnRanker onennRanker = new KnnRanker(pipelinePerformanceStorage, datasetFeatureMap, new EuclideanDistance(), 1, datasetGenerator1NNPerformance, false);
						rankers.add(onennRanker);
					}

					// 1-nn with bayesian averaging
					if (approaches.contains("1-nn_bayesian_averaging")) {
						RegressionDatasetGenerator datasetGenerator1NNPerformance = new DyadRankingImitatingRegressionDatasetGenerator(false, pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
						KnnRanker onennRanker = new KnnRanker(pipelinePerformanceStorage, datasetFeatureMap, new EuclideanDistance(), 1, datasetGenerator1NNPerformance, true);
						rankers.add(onennRanker);
					}

					// 2-nn
					if (approaches.contains("2-nn")) {
						RegressionDatasetGenerator datasetGenerator2NNPerformance = new DyadRankingImitatingRegressionDatasetGenerator(false, pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
						KnnRanker twonnRanker = new KnnRanker(pipelinePerformanceStorage, datasetFeatureMap, new EuclideanDistance(), 2, datasetGenerator2NNPerformance, false);
						rankers.add(twonnRanker);
					}

					// 2-nn with bayesian averaging
					if (approaches.contains("2-nn_bayesian_averaging")) {
						RegressionDatasetGenerator datasetGenerator2NNPerformance = new DyadRankingImitatingRegressionDatasetGenerator(false, pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
						KnnRanker twonnRanker = new KnnRanker(pipelinePerformanceStorage, datasetFeatureMap, new EuclideanDistance(), 2, datasetGenerator2NNPerformance, true);
						rankers.add(twonnRanker);
					}

					// average rank
					if (approaches.contains("average_rank")) {
						RegressionDatasetGenerator datasetGeneratorAverageRank = new DyadRankingImitatingRegressionDatasetGenerator(false, pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
						AverageRankBasedRanker averageRankBasedRanker = new AverageRankBasedRanker(pipelinePerformanceStorage, datasetGeneratorAverageRank, false);
						rankers.add(averageRankBasedRanker);
					}
					// average rank with bayesian averaging
					if (approaches.contains("average_rank_bayesian_averaging")) {
						RegressionDatasetGenerator datasetGeneratorAverageRank = new DyadRankingImitatingRegressionDatasetGenerator(false, pipelineFeatureMap, datasetFeatureMap, pipelinePerformanceStorage, l, n);
						AverageRankBasedRanker averageRankBasedRanker = new AverageRankBasedRanker(pipelinePerformanceStorage, datasetGeneratorAverageRank, true);
						rankers.add(averageRankBasedRanker);
					}
				}
			}

			// random rank
			if (approaches.contains("random")) {
				RandomRanker randomRanker = new RandomRanker();
				rankers.add(randomRanker);
			}

			// oracle
			OracleRanker oracleRanker = new OracleRanker(pipelinePerformanceStorage);
			if (approaches.contains("oracle")) {
				rankers.add(oracleRanker);
			}

			List<Metric> metrics = Arrays.asList(new NormalizedDiscountedCumulativeGain(3), new NormalizedDiscountedCumulativeGain(5), new NormalizedDiscountedCumulativeGain(10), new KendallsTauBasedOnApacheAndRanks(),
					new PerformanceDifferenceOfBestOnTopK(1), new PerformanceDifferenceOfAverageOnTopK(3), new PerformanceDifferenceOfBestOnTopK(3), new PerformanceDifferenceOfAverageOnTopK(5), new PerformanceDifferenceOfBestOnTopK(5));
			System.out.println(getCurrentTimestep() + " :: Running " + (10 * rankers.size()) + " experiments in total.");

			ExecutorService executorService = Executors.newFixedThreadPool(numberOfCPUs);

			for (IdBasedRanker ranker : rankers) {
				Experiment experiment = new Experiment(pipelinePerformanceStorage, new SQLAdapter(dbHost, dbUser, dbPassword, databaseName, true), databaseName, tableName);
				ExperimentRunnable experimentRunnable = new ExperimentRunnable(experiment, experimentNumber, fold, trainingDatasetIds, testDatasetIds, trainingAlgorithmIds, testAlgorithmIds, ranker, metrics, numberOfRankingsToTest,
						numberOfPipelinesPerTestRanking);
				executorService.execute(experimentRunnable);
				experimentNumber++;
			}

			executorService.shutdown();
			executorService.awaitTermination(30, TimeUnit.DAYS); // TODO
		}
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

	private static String getCurrentTimestep() {
		SimpleDateFormat sdfDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");// dd/MM/yyyy
		Date now = new Date();
		String strDate = sdfDate.format(now);
		return strDate;
	}

}
