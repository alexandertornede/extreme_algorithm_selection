package de.upb.isml.tornede.ecai2020.experiments.evaluator;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
	private static final String TEST_PIPELINE_SET_COLUMN_NAME = "test_pipeline_set";
	private static final String TEST_DATASET_ID_COLUMN_NAME = "test_dataset_id";
	private static final String APPROACH_COLUMN_NAME = "approach";
	private static final String DATASET_SPLIT_COLUMN_NAME = "dataset_split";

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

	public void runExperiment(int datasetSplit, List<Integer> trainingDatasets, List<Integer> testDatasets, IdBasedRanker ranker, List<Metric> metrics, int numberOfTestPipelineSets) throws SQLException {
		List<Integer> pipelineIds = pipelinePerformanceStorage.getPipelineIds();

		OracleRanker oracleRanker = new OracleRanker(pipelinePerformanceStorage);

		ranker.train(trainingDatasets);

		for (int testPipelineSet = 0; testPipelineSet < numberOfTestPipelineSets; testPipelineSet++) {
			for (int datasetId : testDatasets) {
				// restrict pipelines we want to rank to those where we know the ground truth
				List<Integer> pipelinesToRank = pipelineIds.stream().filter(i -> pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(i, datasetId) > 0).collect(Collectors.toList());// randomlySelectPipelinesFromList(pipelineIds,
				// amountOfPipelinesToRank, random);

				List<Pair<Integer, Double>> groundTruth = oracleRanker.getRankingOfPipelinesOnDataset(pipelinesToRank, datasetId);

				List<Pair<Integer, Double>> predictedRanking = ranker.getRankingOfPipelinesOnDataset(pipelinesToRank, datasetId);

				for (Metric metric : metrics) {
					double metricResult = metric.evaluate(groundTruth, predictedRanking);

					Map<String, Object> results = new HashMap<>();
					results.put(DATASET_SPLIT_COLUMN_NAME, datasetSplit);
					results.put(APPROACH_COLUMN_NAME, ranker.getName());
					results.put(TEST_DATASET_ID_COLUMN_NAME, datasetId);
					results.put(TEST_PIPELINE_SET_COLUMN_NAME, testPipelineSet);
					results.put(METRIC_COLUMN_NAME, metric.getName());
					results.put(METRIC_RESULT_COLUMN_NAME, metricResult);
					sqlAdapter.insert(tableName, results);
				}
			}
		}

	}

	// private List<Integer> randomlySelectPipelinesFromList(List<Integer> pipelineIds, int amount, Random random) {
	// List<Integer> randomlySelectedPipelines = new ArrayList<>(amount);
	// while (randomlySelectedPipelines.size() < amount) {
	// int randomIndex = random.nextInt(pipelineIds.size());
	// int randomPipelineId = pipelineIds.get(randomIndex);
	// if (!randomlySelectedPipelines.contains(randomPipelineId)) {
	// randomlySelectedPipelines.add(randomPipelineId);
	// }
	// }
	// return randomlySelectedPipelines;
	// }

	private void createResultsTableIfNecessary() throws SQLException {
		List<IKVStore> resultSet = sqlAdapter.getResultsOfQuery("SHOW TABLES");
		boolean hasResultTable = resultSet.stream().anyMatch(kvStore -> kvStore.getAsString("Tables_in_" + databaseName).equals(tableName));

		if (!hasResultTable) {
			sqlAdapter.update("CREATE TABLE " + tableName + " (`" + DATASET_SPLIT_COLUMN_NAME + "` int NOT NULL, \r \n" + " `" + APPROACH_COLUMN_NAME + "` varchar(255),\r\n" + " `" + TEST_DATASET_ID_COLUMN_NAME + "` int NOT NULL, \r \n"
					+ " `" + TEST_PIPELINE_SET_COLUMN_NAME + "` int NOT NULL, \r \n" + " `" + METRIC_RESULT_COLUMN_NAME + "` double NOT NULL, \r \n" + " `" + METRIC_COLUMN_NAME
					+ "` VARCHAR(255) NOT NULL\r \n) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin", new ArrayList<>());
		}
	}
}
