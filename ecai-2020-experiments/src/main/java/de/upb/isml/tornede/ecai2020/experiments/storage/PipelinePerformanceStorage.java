package de.upb.isml.tornede.ecai2020.experiments.storage;

import java.sql.SQLException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.kvstore.IKVStore;
import ai.libs.jaicore.basic.sets.Pair;

public class PipelinePerformanceStorage {

	private static final String PIPELINE_ID_COLUMN_NAME = "pipeline_id";
	private static final String DATASET_ID_COLUMN_NAME = "dataset_id";
	private static final String ACCURACY_COLUMN_NAME = "loss";

	private Map<Pair<Integer, Integer>, Double> pipelineIdDatasetIdToPerformanceMap;

	public PipelinePerformanceStorage(SQLAdapter sqlAdapter, String tableName) {
		initialize(sqlAdapter, tableName);
	}

	private void initialize(SQLAdapter sqlAdapter, String tableName) {
		pipelineIdDatasetIdToPerformanceMap = new HashMap<>();

		String sqlQuery = "SELECT " + PIPELINE_ID_COLUMN_NAME + ", " + DATASET_ID_COLUMN_NAME + ", " + ACCURACY_COLUMN_NAME + " FROM " + tableName;

		try {
			List<IKVStore> resultSet = sqlAdapter.getResultsOfQuery(sqlQuery);
			for (IKVStore kvStore : resultSet) {
				int pipelineId = kvStore.getAsInt(PIPELINE_ID_COLUMN_NAME);
				int datasetId = kvStore.getAsInt(DATASET_ID_COLUMN_NAME);

				String zeroOneLossAsString = kvStore.getAsString(ACCURACY_COLUMN_NAME);
				double zeroOneLoss = 1;
				if (zeroOneLossAsString != null && !zeroOneLossAsString.equalsIgnoreCase("null")) {
					zeroOneLoss = Double.parseDouble(zeroOneLossAsString);
				}

				// do a 1- because the table stores zero-one loss, but we evaluate based on accuracy
				pipelineIdDatasetIdToPerformanceMap.put(new Pair<>(pipelineId, datasetId), (1 - zeroOneLoss));
			}

		} catch (SQLException e) {
			throw new RuntimeException("Could not initialize pipeline performance storage.", e);
		}
	}

	public double getPerformanceForPipelineWithIdOnDatasetWithId(int pipelineId, int datasetId) {
		Pair<Integer, Integer> key = new Pair<>(pipelineId, datasetId);
		if (!pipelineIdDatasetIdToPerformanceMap.containsKey(key)) {
			return 0;
		}
		return pipelineIdDatasetIdToPerformanceMap.get(key);
	}

	public List<Integer> getPipelineIds() {
		return pipelineIdDatasetIdToPerformanceMap.keySet().stream().map(e -> e.getX()).distinct().collect(Collectors.toList());
	}
}
