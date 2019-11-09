package de.upb.isml.tornede.ecai2020.experiments.datasetgen;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.kvstore.IKVStore;
import ai.libs.jaicore.basic.sets.Pair;
import weka.core.Instances;

public class AlgorithmEvaluationRunner {

	public static void main(String[] args) throws Exception {

		int randomSeed = 1234;

		AlgorithmEvaluationRunnerConfig config = ConfigFactory.create(AlgorithmEvaluationRunnerConfig.class);

		long timeOutInSeconds = config.getTimeoutInSeconds();
		String databaseName = config.getDBDatabaseName();
		String tableName = config.getDBTableName();
		System.out.println(config);

		SQLAdapter sqlAdapter = new SQLAdapter(config.getDBHost(), config.getDBUsername(), config.getDBPassword(), databaseName, true);
		createTableIfNecessary(sqlAdapter, databaseName, tableName);

		ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

		AlgorithmGenerator generator = new AlgorithmGenerator(randomSeed);
		List<ComponentInstance> componentInstances = generator.generateRandomAlgorithmConfigurations(100);
		System.out.println("Generated component instances.");
		Map<Integer, Instances> datasetIdsToInstancesMap = OpenMLUtil.getOpenMLDatasetIdsToInstancesMap();
		System.out.println("Loaded datasets");

		for (int c = config.getStartAlgorithmId(); c < config.getEndAlgorithmId(); c++) {
			Pair<Integer, ComponentInstance> componentInstanceAndId = new Pair<>(c, componentInstances.get(c));
			for (Integer datasetId : datasetIdsToInstancesMap.keySet()) {
				AlgorithmEvaluationTask algorithmEvaluationTask = new AlgorithmEvaluationTask(datasetIdsToInstancesMap, new SQLAdapter(config.getDBHost(), config.getDBUsername(), config.getDBPassword(), databaseName, true), tableName,
						randomSeed, componentInstanceAndId, datasetId);
				AlgorithmEvaluationTaskWithTimeout taskWithTimeOut = new AlgorithmEvaluationTaskWithTimeout(algorithmEvaluationTask, TimeUnit.SECONDS, timeOutInSeconds);
				executorService.execute(taskWithTimeOut);
			}
		}
		sqlAdapter.close();
		executorService.shutdown();
	}

	private static final void createTableIfNecessary(SQLAdapter sqlAdapter, String databaseName, String tableName) throws SQLException {
		List<IKVStore> resultSet = sqlAdapter.getResultsOfQuery("SHOW TABLES");
		boolean hasTable = resultSet.stream().anyMatch(kvStore -> kvStore.getAsString("Tables_in_" + databaseName).equals(tableName));
		if (!hasTable) {
			System.out.println("Created table " + tableName);
			sqlAdapter.update("CREATE TABLE " + tableName + " (`" + AlgorithmEvaluationTask.SEED_NAME + "` int NOT NULL, " + " `" + AlgorithmEvaluationTask.DATASET_ID_NAME + "` int NOT NULL, " + " `"
					+ AlgorithmEvaluationTask.ALGORITHM_TEXT_NAME + "` TEXT NOT NULL, " + " `" + AlgorithmEvaluationTask.ALGORITHM_ID_NAME + "` int NOT NULL, " + " `" + AlgorithmEvaluationTask.ACCURACY_NAME + "` VARCHAR(255) NOT NULL, "
					+ " `" + AlgorithmEvaluationTask.STACKTRACE_NAME + "` TEXT DEFAULT NULL" + ") ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin", new ArrayList<>());
		}
	}

}
