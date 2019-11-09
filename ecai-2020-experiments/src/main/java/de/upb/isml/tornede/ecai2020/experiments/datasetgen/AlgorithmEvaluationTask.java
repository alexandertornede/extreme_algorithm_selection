package de.upb.isml.tornede.ecai2020.experiments.datasetgen;

import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang.exception.ExceptionUtils;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.sets.Pair;
import weka.core.Instances;

public class AlgorithmEvaluationTask implements Runnable {

	public static final String SEED_NAME = "seed";
	public static final String DATASET_ID_NAME = "dataset_id";
	public static final String ALGORITHM_TEXT_NAME = "algorithm_string_representation";
	public static final String ALGORITHM_ID_NAME = "algorithm_id";
	public static final String ACCURACY_NAME = "accuracy";
	public static final String STACKTRACE_NAME = "stacktrace";

	private Map<Integer, Instances> datasetIdToInstancesMap;
	private SQLAdapter sqlAdapter;
	private long randomSeed;
	private Pair<Integer, ComponentInstance> componentInstanceAndId;
	private int datasetId;

	private String databaseTableName;

	public AlgorithmEvaluationTask(Map<Integer, Instances> datasetIdToInstancesMap, SQLAdapter sqlAdapter, String databaseTableName, long randomSeed, Pair<Integer, ComponentInstance> componentInstanceAndId, int datasetId) {
		this.datasetIdToInstancesMap = datasetIdToInstancesMap;
		this.sqlAdapter = sqlAdapter;
		this.databaseTableName = databaseTableName;
		this.randomSeed = randomSeed;
		this.componentInstanceAndId = componentInstanceAndId;
		this.datasetId = datasetId;
	}

	@Override
	public void run() {
		System.out.println("Running evaluation of " + componentInstanceAndId.getX() + " on " + datasetId);
		AlgorithmEvaluator evaluator = new AlgorithmEvaluator(randomSeed, datasetIdToInstancesMap);
		String stackTrace = null;
		double accuracy = -1;
		try {
			accuracy = evaluator.evaluateAlgorithm(componentInstanceAndId.getY(), datasetId);
		} catch (Exception ex) {
			stackTrace = ExceptionUtils.getStackTrace(ex);
		}

		try {
			writeResultToDatabase(randomSeed, datasetId, componentInstanceAndId, accuracy, stackTrace);
		} catch (SQLException e) {
			System.err.print("ERROR: Could not write evaluation result! \n" + ExceptionUtils.getStackTrace(e));
		}
	}

	private void writeResultToDatabase(long randomSeed, int datasetId, Pair<Integer, ComponentInstance> componentInstanceAndId, double result, String stackTrace) throws SQLException {

		Map<String, Object> results = new HashMap<>();
		results.put(SEED_NAME, String.valueOf(randomSeed));
		results.put(DATASET_ID_NAME, datasetId);
		results.put(ALGORITHM_TEXT_NAME, componentInstanceAndId.getY().toString());
		results.put(ALGORITHM_ID_NAME, componentInstanceAndId.getX());
		results.put(ACCURACY_NAME, result);
		if (stackTrace != null) {
			results.put(STACKTRACE_NAME, stackTrace);
		}

		sqlAdapter.insert(databaseTableName, results);
	}

	public Map<Integer, Instances> getDatasetIdToInstancesMap() {
		return datasetIdToInstancesMap;
	}

	public long getRandomSeed() {
		return randomSeed;
	}

	public Pair<Integer, ComponentInstance> getComponentInstanceAndId() {
		return componentInstanceAndId;
	}

	public int getDatasetId() {
		return datasetId;
	}

	public SQLAdapter getSqlAdapter() {
		return sqlAdapter;
	}

	public String getDatabaseTableName() {
		return databaseTableName;
	}

}
