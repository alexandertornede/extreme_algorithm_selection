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
	private static final String ALGORITHM_ID_COLUMN_NAME = "algorithm_id";
	private static final String DATASET_ID_COLUMN_NAME = "dataset_id";
	private static final String LOSS_COLUMN_NAME = "loss";
	private static final String ACCURACY_COLUMN_NAME = "accuracy";

	private String alternativeColumnName = ALGORITHM_ID_COLUMN_NAME;
	private String metricColumnName = ACCURACY_COLUMN_NAME;

	private boolean evaluatingOnPipelines = false;

	private Map<Pair<Integer, Integer>, Double> pipelineIdDatasetIdToPerformanceMap;

	public PipelinePerformanceStorage(SQLAdapter sqlAdapter, String tableName) {
		if (tableName.startsWith("pipeline")) {
			evaluatingOnPipelines = true;
			alternativeColumnName = PIPELINE_ID_COLUMN_NAME;
			metricColumnName = LOSS_COLUMN_NAME;
		}
		initialize(sqlAdapter, tableName);
	}

	private void initialize(SQLAdapter sqlAdapter, String tableName) {
		pipelineIdDatasetIdToPerformanceMap = new HashMap<>();

		String sqlQuery = "SELECT " + alternativeColumnName + ", " + DATASET_ID_COLUMN_NAME + ", " + metricColumnName + " FROM " + tableName;

		try {
			List<IKVStore> resultSet = sqlAdapter.getResultsOfQuery(sqlQuery);
			for (IKVStore kvStore : resultSet) {
				int pipelineId = kvStore.getAsInt(alternativeColumnName);
				int datasetId = kvStore.getAsInt(DATASET_ID_COLUMN_NAME);

				if (evaluatingOnPipelines) {
					String zeroOneLossAsString = kvStore.getAsString(metricColumnName);
					double zeroOneLoss = 1;
					if (zeroOneLossAsString != null && !zeroOneLossAsString.equalsIgnoreCase("null")) {
						zeroOneLoss = Double.parseDouble(zeroOneLossAsString);
					}

					// do a 1- because the table stores zero-one loss, but we evaluate based on accuracy
					pipelineIdDatasetIdToPerformanceMap.put(new Pair<>(pipelineId, datasetId), (1 - zeroOneLoss));
				} else {
					String accuracyAsString = kvStore.getAsString(metricColumnName);
					double accuracy = 1;
					if (accuracyAsString != null && !accuracyAsString.equalsIgnoreCase("null")) {
						accuracy = Double.parseDouble(accuracyAsString);
					}

					pipelineIdDatasetIdToPerformanceMap.put(new Pair<>(pipelineId, datasetId), accuracy);
				}
			}

		} catch (SQLException e) {
			throw new RuntimeException("Could not initialize pipeline performance storage.", e);
		}
	}

	public double getPerformanceForPipelineWithIdOnDatasetWithId(int pipelineId, int datasetId) {
		Pair<Integer, Integer> key = new Pair<>(pipelineId, datasetId);
		if (!pipelineIdDatasetIdToPerformanceMap.containsKey(key) || pipelineIdDatasetIdToPerformanceMap.get(key) < 0) {
			return 0;
		}
		return pipelineIdDatasetIdToPerformanceMap.get(key);
	}

	public List<Integer> getPipelineIds() {
		return pipelineIdDatasetIdToPerformanceMap.keySet().stream().map(e -> e.getX()).distinct().collect(Collectors.toList());
	}
}
