package de.upb.isml.tornede.ecai2020.experiments.rankers;

import java.io.File;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;
import de.upb.isml.tornede.ecai2020.experiments.alors.Alors;
import de.upb.isml.tornede.ecai2020.experiments.alors.AlorsException;
import de.upb.isml.tornede.ecai2020.experiments.alors.latent_features.FeaturePredictorException;
import de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion.MatrixCompleterException;
import de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion.ModelBasedMatrixCompleter;
import de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion.cofirank.CofiConfig;
import de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion.cofirank.CofirankCPlusPlus;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;

public class AlorsBasedRanker implements IdBasedRanker {

	private static final String PATH_TO_COFIRANK_EXECUTABLE = "cofirank/cofirank-deploy";
	private static final String NAME_OF_COFIRANK_CONFIGURATION = "config.cfg";
	private static final String PATH_TO_COFIRANK_OUTPUTFOLDER = "cf_output";
	private static final String NAME_OF_COFIRANK_TRAINFILE = "train.lsvm";
	private static final String NAME_OF_COFIRANK_TESTFILE = "test.lsvm";

	private DatasetFeatureRepresentationMap datasetFeatureRepresentationMap;
	private PipelinePerformanceStorage pipelinePerformanceStorage;

	private List<Integer> datasetIdsSorted;
	private List<Integer> pipelineIdsSorted;

	private Alors alors;

	private String performanceMeasureToOptimize;
	private long randomSeed;

	public AlorsBasedRanker(DatasetFeatureRepresentationMap datasetFeatureRepresentationMap, PipelinePerformanceStorage pipelinePerformanceStorage, String performanceMeasureToOptimize) {
		this.datasetFeatureRepresentationMap = datasetFeatureRepresentationMap;
		this.pipelinePerformanceStorage = pipelinePerformanceStorage;
		this.performanceMeasureToOptimize = performanceMeasureToOptimize;
	}

	@Override
	public void initialize(long randomSeed) {
		this.randomSeed = randomSeed;
	}

	@Override
	public void train(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds) {
		String pathToCFOutputFolder = PATH_TO_COFIRANK_OUTPUTFOLDER + "_" + randomSeed + "_" + performanceMeasureToOptimize;
		File outputFolder = new File(pathToCFOutputFolder);
		if (!outputFolder.exists()) {
			outputFolder.mkdirs();
		}

		CofiConfig config = new CofiConfig(PATH_TO_COFIRANK_EXECUTABLE, pathToCFOutputFolder + "/" + NAME_OF_COFIRANK_CONFIGURATION, pathToCFOutputFolder, pathToCFOutputFolder + "/" + NAME_OF_COFIRANK_TRAINFILE,
				pathToCFOutputFolder + "/" + NAME_OF_COFIRANK_TESTFILE);
		config.setOptimizedMeasure(performanceMeasureToOptimize);
		ModelBasedMatrixCompleter matrixCompleter = new CofirankCPlusPlus(config);
		this.alors = new Alors(matrixCompleter);

		datasetIdsSorted = trainingDatasetIds.stream().sorted(Comparator.comparingInt(i -> i)).collect(Collectors.toList());

		pipelineIdsSorted = pipelinePerformanceStorage.getPipelineIds().stream().sorted(Comparator.comparingInt(i -> i)).collect(Collectors.toList());

		double[][] performanceMatrix = new double[datasetIdsSorted.size()][pipelineIdsSorted.size()];
		for (int i = 0; i < datasetIdsSorted.size(); i++) {
			int datasetId = datasetIdsSorted.get(i);
			for (int j = 0; j < pipelineIdsSorted.size(); j++) {
				int pipelineId = pipelineIdsSorted.get(j);
				double performance = pipelinePerformanceStorage.getPerformanceForPipelineWithIdOnDatasetWithId(pipelineId, datasetId);
				performanceMatrix[i][j] = performance; // *100
			}
		}

		double[][] datasetFeatureMatrix = new double[datasetIdsSorted.size()][datasetFeatureRepresentationMap.getNumberOfFeatures()];
		for (int i = 0; i < datasetIdsSorted.size(); i++) {
			int datasetId = datasetIdsSorted.get(i);
			double[] datasetFeatures = datasetFeatureRepresentationMap.getFeatureRepresentationForDataset(datasetId);
			for (int j = 0; j < datasetFeatureRepresentationMap.getNumberOfFeatures(); j++) {
				datasetFeatureMatrix[i][j] = datasetFeatures[j];
			}
		}

		try {
			alors.completeMatrixAndPrepareColdStart(performanceMatrix, datasetFeatureMatrix);
		} catch (MatrixCompleterException | FeaturePredictorException e) {
			throw new RuntimeException("Could not train Alors!", e);
		}
	}

	@Override
	public List<Pair<Integer, Double>> getRankingOfPipelinesOnDataset(List<Integer> pipelineIdsToRank, int datasetId) {
		double[] datasetFeatures = datasetFeatureRepresentationMap.getFeatureRepresentationForDataset(datasetId);
		try {
			double[] predictedPerformancesOfAllPipelines = alors.predictForFeatures(datasetFeatures);

			List<Pair<Integer, Double>> pipelinePerformancePairs = new ArrayList<>();
			for (int pipelineId : pipelineIdsToRank) {
				int indexOfPipelineId = pipelineIdsSorted.indexOf(pipelineId);
				double predictedPerformance = predictedPerformancesOfAllPipelines[indexOfPipelineId];
				pipelinePerformancePairs.add(new Pair<>(pipelineId, predictedPerformance));
			}

			return pipelinePerformancePairs.stream().sorted(Comparator.comparingDouble(p -> ((Pair<Integer, Double>) p).getY()).reversed()).collect(Collectors.toList());
		} catch (FeaturePredictorException | AlorsException ex) {
			throw new RuntimeException("Could not predict ranking using ALORS.", ex);
		}
	}

	@Override
	public String getName() {
		return "alors_cofirank_" + performanceMeasureToOptimize;
	}

}
