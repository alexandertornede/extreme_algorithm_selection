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
import de.upb.isml.tornede.ecai2020.experiments.rankers.regression.RandomRegressionDatasetGenerator;
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

		PipelineFeatureRepresentationMap pipelineFeatureMap = new PipelineFeatureRepresentationMap(sqlAdapter, "algorithm_metafeatures");
		DatasetFeatureRepresentationMap datasetFeatureMap = new DatasetFeatureRepresentationMap(sqlAdapter, "dataset_metafeatures_new");
		PipelinePerformanceStorage pipelinePerformanceStorage = new PipelinePerformanceStorage(sqlAdapter, "algorithm_evaluations_with_timeouts");

		List<IdBasedRanker> rankers = new ArrayList<>();

		int[] numberOfRankingsPerTrainingDataset = new int[] { 25, 50, 125 };
		int[] lengthOfRankingsForDyadRanker = new int[] { 2, 3 };
		int numberOfRankingsToTest = 100;
		int numberOfPipelinesPerTestRanking = 10;

		for (int l : lengthOfRankingsForDyadRanker) {
			for (int n : numberOfRankingsPerTrainingDataset) {
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

		rankers.add(new AlorsBasedRanker(datasetFeatureMap, pipelinePerformanceStorage, "REGRESSION"));
		rankers.add(new AlorsBasedRanker(datasetFeatureMap, pipelinePerformanceStorage, "NDCG"));

		List<Metric> metrics = Arrays.asList(new NormalizedDiscountedCumulativeGain(3), new NormalizedDiscountedCumulativeGain(5), new NormalizedDiscountedCumulativeGain(10), new KendallsTauBasedOnApache(),
				new KendallsTauBasedOnApacheAndRanks(), new PerformanceDifferenceOfBestOnTopK(1), new PerformanceDifferenceOfAverageOnTopK(3), new PerformanceDifferenceOfBestOnTopK(3), new PerformanceDifferenceOfAverageOnTopK(5),
				new PerformanceDifferenceOfBestOnTopK(5));
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

	private static String getCurrentTimestep() {
		SimpleDateFormat sdfDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");// dd/MM/yyyy
		Date now = new Date();
		String strDate = sdfDate.format(now);
		return strDate;
	}

}
