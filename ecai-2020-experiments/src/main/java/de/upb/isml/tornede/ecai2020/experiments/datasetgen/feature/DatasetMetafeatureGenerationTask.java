package de.upb.isml.tornede.ecai2020.experiments.datasetgen.feature;

import java.sql.SQLException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.ml.metafeatures.LandmarkerCharacterizer;
import weka.core.Instances;

public class DatasetMetafeatureGenerationTask implements Runnable {

	public static final String DATASET_ID_NAME = "dataset_id";
	public static final String METAFEATURES_NAME = "metafeatures";

	private SQLAdapter sqlAdapter;
	private String databaseTableName;

	private int datasetId;
	private Instances instances;

	public DatasetMetafeatureGenerationTask(SQLAdapter sqlAdapter, String databaseTableName, int datasetId, Instances instances) {
		this.sqlAdapter = sqlAdapter;
		this.databaseTableName = databaseTableName;
		this.datasetId = datasetId;
		this.instances = instances;
	}

	@Override
	public void run() {
		try {
			String metafeatures = computeMetafeaturesForDataset(instances);
			writeResultToDatabase(datasetId, metafeatures);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public String computeMetafeaturesForDataset(Instances dataset) throws Exception {
		Map<String, Double> metaFeatures = new LandmarkerCharacterizer().characterize(dataset);
		double[] datasetMetaFeatures = metaFeatures.entrySet().stream().mapToDouble(Map.Entry::getValue).toArray();
		String datasetMetaFeaturesAsString = Arrays.stream(datasetMetaFeatures).mapToObj(String::valueOf).collect(Collectors.joining(" "));

		return datasetMetaFeaturesAsString;
	}

	private void writeResultToDatabase(int datasetId, String datasetMetafeaturesAsString) throws SQLException {
		Map<String, Object> results = new HashMap<>();
		results.put(DATASET_ID_NAME, datasetId);
		results.put(METAFEATURES_NAME, datasetMetafeaturesAsString);
		sqlAdapter.insert(databaseTableName, results);
		System.out.println("Wrote metafeatures for dataset " + datasetId + ": " + datasetMetafeaturesAsString);
		sqlAdapter.close();
	}

}
