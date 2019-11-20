package de.upb.isml.tornede.ecai2020.experiments.datasetgen.algorithm;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.kvstore.IKVStore;

public class AlgorithmMetafeatureToDatabaseWriter {

	private static String databaseName = "conference_ecai2020";

	private static String ALGORITHM_ID_COLUMN_NAME = "id";
	private static String ALGORITHM_STRING_COLUMN_NAME = "string_representation";
	private static String ALGORITHM_METAFEATURE_NAME = "metafeatures";

	private static String ALGORITHM_DEFINITION_TABLE_NAME = "algorithm_metafeatures";

	public static void main(String[] args) throws SQLException {
		SQLAdapter sqlAdapter = new SQLAdapter("database_server", "user", "password", databaseName, true);
		createResultsTableIfNecessary(sqlAdapter);

		long randomSeed = 1234;
		AlgorithmGenerator generator = new AlgorithmGenerator(randomSeed);
		List<ComponentInstance> componentInstances = generator.generateRandomAlgorithmConfigurations(100);

		AlgorithmMetafeatureGenerator algorithmFeatureGenerator = new AlgorithmMetafeatureGenerator(generator.getBaseComponents());

		for (int i = 0; i < componentInstances.size(); i++) {
			ComponentInstance instance = componentInstances.get(i);
			String[] features = algorithmFeatureGenerator.generateMetafeatures(instance);
			String featureString = Arrays.stream(features).collect(Collectors.joining(" "));

			Map<String, Object> results = new HashMap<>();
			results.put(ALGORITHM_ID_COLUMN_NAME, i);
			results.put(ALGORITHM_STRING_COLUMN_NAME, instance.toString());
			results.put(ALGORITHM_METAFEATURE_NAME, featureString);
		}
		sqlAdapter.close();
	}

	private static void createResultsTableIfNecessary(SQLAdapter sqlAdapter) throws SQLException {
		List<IKVStore> resultSet = sqlAdapter.getResultsOfQuery("SHOW TABLES");
		boolean hasTable = resultSet.stream().anyMatch(kvStore -> kvStore.getAsString("Tables_in_" + databaseName).equals(ALGORITHM_DEFINITION_TABLE_NAME));

		if (!hasTable) {
			sqlAdapter.update("CREATE TABLE " + ALGORITHM_DEFINITION_TABLE_NAME + " (`" + ALGORITHM_ID_COLUMN_NAME + "` int NOT NULL, " + " `" + ALGORITHM_STRING_COLUMN_NAME + "` TEXT NOT NULL, " + " `" + ALGORITHM_METAFEATURE_NAME
					+ "` TEXT NOT NULL" + ") ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin", new ArrayList<>());
		}
	}

}
