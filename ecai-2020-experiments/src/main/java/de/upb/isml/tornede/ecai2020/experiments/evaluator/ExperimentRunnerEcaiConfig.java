package de.upb.isml.tornede.ecai2020.experiments.evaluator;

import java.util.List;

import org.aeonbits.owner.Config.Sources;

import ai.libs.jaicore.basic.IDatabaseConfig;

@Sources({ "file:./conf/experiment_runner.properties" })
public interface ExperimentRunnerEcaiConfig extends IDatabaseConfig {

	public static final String DATASET_IDS_NAME = "dataset_ids";

	@Key(DATASET_IDS_NAME)
	public List<Integer> getDatasetIds();

}
