package de.upb.isml.tornede.ecai2020.experiments.evaluator;

import java.util.List;

import org.aeonbits.owner.Config.Sources;

import ai.libs.jaicore.basic.IDatabaseConfig;

@Sources({ "file:./conf/experiment_runner.properties" })
public interface ExperimentRunnerEcaiConfig extends IDatabaseConfig {

	public static final String DATASET_IDS_NAME = "dataset_ids";
	public static final String AMOUNT_CPUS_NAME = "amount_cpus";
	public static final String FOLDS_ON_DATASETS_NAME = "folds_on_datasets";
	public static final String FOLDS_ON_ALGORITHMS_NAME = "folds_on_algorithms";
	public static final String NAME_APPROACHES = "approaches";

	@Key(DATASET_IDS_NAME)
	public List<Integer> getDatasetIds();

	@Key(AMOUNT_CPUS_NAME)
	public int getAmountOfCPUsToUse();

	@Key(FOLDS_ON_DATASETS_NAME)
	public boolean foldsOnDatasets();

	@Key(FOLDS_ON_ALGORITHMS_NAME)
	public boolean foldsOnAlgorithms();

	@Key(NAME_APPROACHES)
	public List<String> getApproaches();
}
