package de.upb.isml.tornede.ecai2020.experiments.datasetgen.feature;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.kvstore.IKVStore;
import de.upb.isml.tornede.ecai2020.experiments.datasetgen.OpenMLUtil;
import weka.core.Instances;

public class DatasetMetafeatureGeneratorRunner {

	public static void main(String[] args) throws SQLException {

		DatasetMetafeatureGeneratorRunnerConfig config = ConfigFactory.create(DatasetMetafeatureGeneratorRunnerConfig.class);

		String databaseName = config.getDBDatabaseName();
		String tableName = config.getDBTableName();
		System.out.println(config);

		SQLAdapter sqlAdapter = new SQLAdapter(config.getDBHost(), config.getDBUsername(), config.getDBPassword(), databaseName, true);
		createTableIfNecessary(sqlAdapter, databaseName, tableName);

		ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

		Map<Integer, Instances> datasetIdsToInstancesMap = OpenMLUtil.getOpenMLDatasetIdsToInstancesMap();
		System.out.println("Loaded datasets");
		sqlAdapter.close();

		for (Entry<Integer, Instances> entry : datasetIdsToInstancesMap.entrySet()) {
			DatasetMetafeatureGenerationTask metafeatureGenerationTask = new DatasetMetafeatureGenerationTask(new SQLAdapter(config.getDBHost(), config.getDBUsername(), config.getDBPassword(), databaseName, true), tableName, entry.getKey(),
					entry.getValue());
			executorService.execute(metafeatureGenerationTask);
		}
		executorService.shutdown();
	}

	private static final void createTableIfNecessary(SQLAdapter sqlAdapter, String databaseName, String tableName) throws SQLException {
		List<IKVStore> resultSet = sqlAdapter.getResultsOfQuery("SHOW TABLES");
		boolean hasTable = resultSet.stream().anyMatch(kvStore -> kvStore.getAsString("Tables_in_" + databaseName).equals(tableName));
		if (!hasTable) {
			System.out.println("Created table " + tableName);
			sqlAdapter.update("CREATE TABLE " + tableName + " (`" + DatasetMetafeatureGenerationTask.DATASET_ID_NAME + "` int NOT NULL, " + " `" + DatasetMetafeatureGenerationTask.METAFEATURES_NAME + "` TEXT NOT NULL "
					+ ") ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin", new ArrayList<>());
		}
	}

}
