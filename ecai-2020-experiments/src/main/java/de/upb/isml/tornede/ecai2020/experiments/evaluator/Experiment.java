package de.upb.isml.tornede.ecai2020.experiments.evaluator;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.kvstore.IKVStore;
import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.loss.Metric;
import de.upb.isml.tornede.ecai2020.experiments.rankers.IdBasedRanker;
import de.upb.isml.tornede.ecai2020.experiments.rankers.OracleRanker;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;

public class Experiment {

	private static final String METRIC_RESULT_COLUMN_NAME = "metric_result";
	private static final String METRIC_COLUMN_NAME = "metric";
	private static final String TEST_DATASET_ID_COLUMN_NAME = "test_dataset_id";
	private static final String APPROACH_COLUMN_NAME = "approach";
	private static final String DATASET_SPLIT_COLUMN_NAME = "dataset_split";
	private static final String RANKING_NUMBER_COLUMN_NAME = "ranking_to_test";

	private PipelinePerformanceStorage pipelinePerformanceStorage;

	private SQLAdapter sqlAdapter;
	private String databaseName;
	private String tableName;

	public Experiment(PipelinePerformanceStorage pipelinePerformanceStorage, SQLAdapter sqlAdapter, String databaseName, String tableName) throws SQLException {
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
		this.sqlAdapter = sqlAdapter;
		this.databaseName = databaseName;
		this.tableName = tableName;
		createResultsTableIfNecessary();
	}

	public void runExperiment(int datasetSplit, List<Integer> trainingDatasets, List<Integer> testDatasets, IdBasedRanker ranker, List<Metric> metrics, int numberOfRankingsToTest, int amountOfPipelinesToSelect) throws SQLException {
		List<Integer> pipelineIds = pipelinePerformanceStorage.getPipelineIds();

		OracleRanker oracleRanker = new OracleRanker(pipelinePerformanceStorage);

		ranker.initialize(datasetSplit);
		ranker.train(trainingDatasets);

		for (int datasetId : testDatasets) {
			// restrict pipelines we want to rank to those where we know the ground truth
			List<Integer> pipelinesToRank = pipelineIds.stream().filter(i -> pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(i, datasetId) > 0).collect(Collectors.toList());

			for (int i = 0; i < numberOfRankingsToTest; i++) {
				pipelinesToRank = getRandomPipelinesWithDistinctPerformance(datasetId, pipelinesToRank, amountOfPipelinesToSelect, new Random(datasetSplit * numberOfRankingsToTest + i));

				List<Pair<Integer, Double>> groundTruth = oracleRanker.getRankingOfPipelinesOnDataset(pipelinesToRank, datasetId);

				List<Pair<Integer, Double>> predictedRanking = ranker.getRankingOfPipelinesOnDataset(pipelinesToRank, datasetId);

				for (Metric metric : metrics) {
					double metricResult = metric.evaluate(groundTruth, predictedRanking);

					Map<String, Object> results = new HashMap<>();
					results.put(DATASET_SPLIT_COLUMN_NAME, datasetSplit);
					results.put(APPROACH_COLUMN_NAME, ranker.getName());
					results.put(TEST_DATASET_ID_COLUMN_NAME, datasetId);
					results.put(RANKING_NUMBER_COLUMN_NAME, i);
					results.put(METRIC_COLUMN_NAME, metric.getName());
					results.put(METRIC_RESULT_COLUMN_NAME, metricResult);
					sqlAdapter.insert(tableName, results);
				}
			}
			System.out.println("Finished test dataset " + datasetId);
		}
		sqlAdapter.close();
	}

	public List<Integer> getRandomPipelinesWithDistinctPerformance(int datasetId, List<Integer> pipelineIdsToRank, int amountOfPipelinesToSelect, Random random) {
		Set<Double> performancesSeen = new HashSet<>();
		List<Integer> selectedPipelinesToRank = new ArrayList<>(amountOfPipelinesToSelect);
		while (selectedPipelinesToRank.size() < amountOfPipelinesToSelect) {
			int randomPipelineId = pipelineIdsToRank.get(random.nextInt(pipelineIdsToRank.size()));
			if (!selectedPipelinesToRank.contains(randomPipelineId)) {
				double performanceOfRandomPipelineId = pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(randomPipelineId, datasetId);
				if (!performancesSeen.contains(performanceOfRandomPipelineId)) {
					selectedPipelinesToRank.add(randomPipelineId);
					performancesSeen.add(performanceOfRandomPipelineId);
				}
			}
		}
		return selectedPipelinesToRank;
	}

	private void createResultsTableIfNecessary() throws SQLException {
		List<IKVStore> resultSet = sqlAdapter.getResultsOfQuery("SHOW TABLES");
		boolean hasResultTable = resultSet.stream().anyMatch(kvStore -> kvStore.getAsString("Tables_in_" + databaseName).equals(tableName));

		if (!hasResultTable) {
			sqlAdapter.update("CREATE TABLE " + tableName + " (`" + DATASET_SPLIT_COLUMN_NAME + "` int NOT NULL, \r \n" + " `" + APPROACH_COLUMN_NAME + "` varchar(255),\r\n" + " `" + TEST_DATASET_ID_COLUMN_NAME + "` int NOT NULL, \r \n"
					+ " `" + RANKING_NUMBER_COLUMN_NAME + "` int NOT NULL, \r \n" + " `" + METRIC_RESULT_COLUMN_NAME + "` double NOT NULL, \r \n" + " `" + METRIC_COLUMN_NAME
					+ "` VARCHAR(255) NOT NULL\r \n) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin", new ArrayList<>());
		}
	}
}
