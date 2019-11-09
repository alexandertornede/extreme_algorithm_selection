package de.upb.isml.tornede.ecai2020.experiments.datasetgen;

import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang.exception.ExceptionUtils;

import de.upb.isml.tornede.ecai2020.experiments.utils.TimeLimitedCodeBlock;

public class AlgorithmEvaluationTaskWithTimeout implements Runnable {

	private AlgorithmEvaluationTask algorithmEvaluationTask;
	private TimeUnit timeUnit;
	private long timeOut;

	public AlgorithmEvaluationTaskWithTimeout(AlgorithmEvaluationTask algorithmEvaluationTask, TimeUnit timeUnit, long timeOut) {
		this.algorithmEvaluationTask = algorithmEvaluationTask;
		this.timeUnit = timeUnit;
		this.timeOut = timeOut;
	}

	@Override
	public void run() {
		try {
			TimeLimitedCodeBlock.runWithTimeout(algorithmEvaluationTask, timeOut, timeUnit);
		} catch (Exception e) {
			try {
				writeExceptionToDatabase(e);
			} catch (SQLException ex) {
				System.err.print("ERROR: Could write evaluation result! \n" + ExceptionUtils.getStackTrace(ex));
			}
		} finally {
			algorithmEvaluationTask.getSqlAdapter().close();
		}
	}

	private void writeExceptionToDatabase(Exception exception) throws SQLException {
		Map<String, Object> results = new HashMap<>();
		results.put(AlgorithmEvaluationTask.SEED_NAME, String.valueOf(algorithmEvaluationTask.getRandomSeed()));
		results.put(AlgorithmEvaluationTask.DATASET_ID_NAME, algorithmEvaluationTask.getDatasetId());
		results.put(AlgorithmEvaluationTask.ALGORITHM_TEXT_NAME, algorithmEvaluationTask.getComponentInstanceAndId().getY().toString());
		results.put(AlgorithmEvaluationTask.ALGORITHM_ID_NAME, algorithmEvaluationTask.getComponentInstanceAndId().getX());
		results.put(AlgorithmEvaluationTask.ACCURACY_NAME, -1);
		results.put(AlgorithmEvaluationTask.STACKTRACE_NAME, ExceptionUtils.getStackTrace(exception));

		algorithmEvaluationTask.getSqlAdapter().insert(algorithmEvaluationTask.getDatabaseTableName(), results);
	}

}
