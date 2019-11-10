package de.upb.isml.tornede.ecai2020.experiments.storage;

import java.sql.SQLException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.kvstore.IKVStore;

public class PipelineFeatureRepresentationMap {

	private static final String FEATURE_REPRESENTATION_COLUMN_NAME = "metafeatures";
	private static final String PIPELINE_ID_COLUMN_NAME = "id";

	private Map<double[], Integer> featureToPipelineIdMap;
	private Map<Integer, double[]> pipelineIdToFeaturesMap;

	public PipelineFeatureRepresentationMap(SQLAdapter sqlAdapter, String tableName) {
		initialize(sqlAdapter, tableName);
	}

	private void initialize(SQLAdapter sqlAdapter, String tableName) {
		featureToPipelineIdMap = new HashMap<>();
		pipelineIdToFeaturesMap = new HashMap<>();

		String selectAllPipelinesQuery = "SELECT DISTINCT " + PIPELINE_ID_COLUMN_NAME + "," + FEATURE_REPRESENTATION_COLUMN_NAME + " FROM `" + tableName + "`";
		try {
			List<IKVStore> resultSet = sqlAdapter.getResultsOfQuery(selectAllPipelinesQuery);
			for (IKVStore kvStore : resultSet) {
				List<Double> featureRepresentationAsList = kvStore.getAsDoubleList(FEATURE_REPRESENTATION_COLUMN_NAME, " ");
				double[] featureRepresentation = featureRepresentationAsList.stream().mapToDouble(d -> d).toArray();

				int pipelineId = kvStore.getAsInt(PIPELINE_ID_COLUMN_NAME);

				featureToPipelineIdMap.put(featureRepresentation, pipelineId);
				pipelineIdToFeaturesMap.put(pipelineId, featureRepresentation);
			}

		} catch (SQLException e) {
			throw new RuntimeException("Could not initialize pipeline feature representations.", e);
		}
	}

	public double[] getFeatureRepresentationForPipeline(int pipelineId) {
		return pipelineIdToFeaturesMap.get(pipelineId);
	}

	public int getPipelineIdForFeatureRepresentation(double[] featureRepresentation) {
		return featureToPipelineIdMap.get(featureRepresentation);
	}

}
