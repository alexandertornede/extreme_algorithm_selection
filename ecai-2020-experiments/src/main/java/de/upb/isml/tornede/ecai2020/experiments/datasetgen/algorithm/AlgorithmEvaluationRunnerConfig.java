package de.upb.isml.tornede.ecai2020.experiments.datasetgen.algorithm;

import org.aeonbits.owner.Config.Sources;

import ai.libs.jaicore.basic.IDatabaseConfig;

@Sources({ "file:./conf/algorithm_evaluation_runner.properties" })
public interface AlgorithmEvaluationRunnerConfig extends IDatabaseConfig {

	public static final String START_ALGORITHM_ID = "start_alg_id";
	public static final String END_ALGORITHM_ID = "end_alg_id";
	public static final String TIME_OUT_IN_SECONDS = "timeout_in_s";

	@Key(START_ALGORITHM_ID)
	public int getStartAlgorithmId();

	@Key(END_ALGORITHM_ID)
	public int getEndAlgorithmId();

	@Key(TIME_OUT_IN_SECONDS)
	public long getTimeoutInSeconds();
}
