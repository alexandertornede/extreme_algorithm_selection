package de.upb.isml.tornede.ecai2020.experiments.evaluator;

import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import de.upb.isml.tornede.ecai2020.experiments.loss.Metric;
import de.upb.isml.tornede.ecai2020.experiments.rankers.IdBasedRanker;

public class ExperimentRunnable implements Runnable {

	private Experiment experiment;
	private int experimentNumber;
	private int datasetSplit;
	private List<Integer> trainingDatasets;
	private List<Integer> testDatasets;
	private List<Integer> trainingPipelines;
	private List<Integer> testPipelines;
	private IdBasedRanker ranker;
	private List<Metric> metrics;
	private int numberOfRankingsToTest;
	private int amountOfPipelinesToSelect;

	public ExperimentRunnable(Experiment experiment, int experimentNumber, int datasetSplit, List<Integer> trainingDatasets, List<Integer> testDatasets, List<Integer> trainingPipelines, List<Integer> testPipelines, IdBasedRanker ranker,
			List<Metric> metrics, int numberOfRankingsToTest, int amountOfPipelinesToSelect) {
		super();
		this.experiment = experiment;
		this.experimentNumber = experimentNumber;
		this.datasetSplit = datasetSplit;
		this.trainingDatasets = trainingDatasets;
		this.testDatasets = testDatasets;
		this.trainingPipelines = trainingPipelines;
		this.testPipelines = testPipelines;
		this.ranker = ranker;
		this.metrics = metrics;
		this.numberOfRankingsToTest = numberOfRankingsToTest;
		this.amountOfPipelinesToSelect = amountOfPipelinesToSelect;
	}

	@Override
	public void run() {
		try {
			System.out.println(getCurrentTimestep() + " :: Running experiment " + experimentNumber + ": " + ranker.getName());
			experiment.runExperiment(datasetSplit, trainingDatasets, testDatasets, trainingPipelines, testPipelines, ranker, metrics, numberOfRankingsToTest, amountOfPipelinesToSelect);
			System.out.println(getCurrentTimestep() + " :: Experiment " + experimentNumber + " done: " + ranker.getName());
		} catch (SQLException e) {
			e.printStackTrace();
		}
	}

	private static String getCurrentTimestep() {
		SimpleDateFormat sdfDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");// dd/MM/yyyy
		Date now = new Date();
		String strDate = sdfDate.format(now);
		return strDate;
	}

}
