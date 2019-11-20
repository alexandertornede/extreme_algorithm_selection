package de.upb.isml.tornede.ecai2020.experiments.storage;

import java.sql.SQLException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.kvstore.IKVStore;

public class DatasetFeatureRepresentationMap {

	private static final String FEATURE_REPRESENTATION_COLUMN_NAME = "metafeatures";
	private static final String DATASET_ID_COLUMN_NAME = "dataset_id";

	private Map<double[], Integer> featureToDatasetIdMap;
	private Map<Integer, double[]> datasetIdToFeaturesMap;

	public DatasetFeatureRepresentationMap(SQLAdapter sqlAdapter, String tableName) {
		initialize(sqlAdapter, tableName);
	}

	private void initialize(SQLAdapter sqlAdapter, String tableName) {
		featureToDatasetIdMap = new HashMap<>();
		datasetIdToFeaturesMap = new HashMap<>();

		String selectAllPipelinesQuery = "SELECT DISTINCT " + DATASET_ID_COLUMN_NAME + "," + FEATURE_REPRESENTATION_COLUMN_NAME + " FROM `" + tableName + "`";
		try {
			List<IKVStore> resultSet = sqlAdapter.getResultsOfQuery(selectAllPipelinesQuery);
			for (IKVStore kvStore : resultSet) {
				List<Double> featureRepresentationAsList = kvStore.getAsDoubleList(FEATURE_REPRESENTATION_COLUMN_NAME, " ");
				double[] featureRepresentation = featureRepresentationAsList.stream().mapToDouble(d -> d).toArray();

				int datasetId = kvStore.getAsInt(DATASET_ID_COLUMN_NAME);

				featureToDatasetIdMap.put(featureRepresentation, datasetId);
				datasetIdToFeaturesMap.put(datasetId, featureRepresentation);
			}

		} catch (SQLException e) {
			throw new RuntimeException("Could not initialize dataset feature representations.", e);
		}
	}

	public double[] getFeatureRepresentationForDataset(int datasetId) {
		return datasetIdToFeaturesMap.get(datasetId);
	}

	public int getDatasetIdForFeatureRepresentation(double[] featureRepresentation) {
		return featureToDatasetIdMap.get(featureRepresentation);
	}

	public List<Integer> getDatasetIds() {
		return datasetIdToFeaturesMap.keySet().stream().collect(Collectors.toList());
	}

	public int getNumberOfFeatures() {
		return datasetIdToFeaturesMap.values().stream().findAny().get().length;
	}
}
