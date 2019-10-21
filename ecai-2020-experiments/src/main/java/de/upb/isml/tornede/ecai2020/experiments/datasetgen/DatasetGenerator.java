package de.upb.isml.tornede.ecai2020.experiments.datasetgen;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.math.linearalgebra.DenseDoubleVector;
import ai.libs.jaicore.math.linearalgebra.Vector;
import ai.libs.jaicore.ml.core.exception.TrainingException;
import ai.libs.jaicore.ml.dyadranking.Dyad;
import ai.libs.jaicore.ml.dyadranking.algorithm.featuretransform.FeatureTransformPLDyadRanker;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import ai.libs.jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import de.upb.isml.tornede.ecai2020.experiments.storage.DatasetFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelineFeatureRepresentationMap;
import de.upb.isml.tornede.ecai2020.experiments.storage.PipelinePerformanceStorage;
import de.upb.isml.tornede.ecai2020.experiments.utils.Util;

public class DatasetGenerator {

	private List<Integer> pipelineIds;
	private int[] numTrainingPairs;
	private DatasetFeatureRepresentationMap datasetFeatures;
	private PipelineFeatureRepresentationMap pipelineFeatures;
	private PipelinePerformanceStorage pipelinePerformances;

	public DatasetGenerator(int[] numTrainingPairs, DatasetFeatureRepresentationMap datasetFeatures, PipelineFeatureRepresentationMap pipelineFeatures, PipelinePerformanceStorage pipelinePerformances) {
		System.out.println("Initialized maps");
		this.pipelineIds = pipelinePerformances.getPipelineIds();
		this.numTrainingPairs = numTrainingPairs;
		this.datasetFeatures = datasetFeatures;
		this.pipelineFeatures = pipelineFeatures;
		this.pipelinePerformances = pipelinePerformances;
	}

	public int getRandomSeed(int splitnum, int numTrainingPairsIndex) {
		return splitnum * numTrainingPairs.length + numTrainingPairsIndex;
	}

	public DyadRankingDataset generateTrainingDataset(int splitnum, int numTrainingPairsIndex) throws IOException, URISyntaxException {
		// read train and test ids from json
		Pair<List<Integer>, List<Integer>> trainAndTestSplit = Util.getTrainingAndTestDatasetSplitsForSplitId(splitnum);
		List<Integer> trainInstances = trainAndTestSplit.getX();
		Random random = new Random(getRandomSeed(splitnum, numTrainingPairsIndex));

		List<IDyadRankingInstance> dyadRankingInstances = new ArrayList<>();

		// for every instance, sample numTrainingPairs as the train data
		trainInstances.forEach(datasetId -> {
			for (int i = 0; i < numTrainingPairs[numTrainingPairsIndex]; i++) {
				// sample first pipeline id and performance
				int pipeline1Id = pipelineIds.get(random.nextInt(pipelineIds.size()));
				double pipeline1Performance = pipelinePerformances.getPerformanceForPipelineWithIdOnDatasetWithId(pipeline1Id, datasetId);
				int pipeline2Id = pipeline1Id;
				double pipeline2Performance = pipeline1Performance;

				// sample 2nd pipeline that is different from the first
				while (pipeline1Performance == pipeline2Performance) {
					pipeline2Id = pipelineIds.get(random.nextInt(pipelineIds.size()));
					pipeline2Performance = pipelinePerformances.getPerformanceForPipelineWithIdOnDatasetWithId(pipeline2Id, datasetId);
				}

				// create dyads
				Vector instanceDatasetFeatures = new DenseDoubleVector(datasetFeatures.getFeatureRepresentationForDataset(datasetId));
				Vector instance1Features = new DenseDoubleVector(this.pipelineFeatures.getFeatureRepresentationForPipeline(pipeline1Id));
				Dyad dyad1 = new Dyad(instanceDatasetFeatures, instance1Features);
				Vector instance2Features = new DenseDoubleVector(this.pipelineFeatures.getFeatureRepresentationForPipeline(pipeline2Id));
				Dyad dyad2 = new Dyad(instanceDatasetFeatures, instance2Features);

				// add to ranking
				List<Dyad> dyads = new ArrayList<>();
				if (pipeline1Performance < pipeline2Performance) {
					dyads.add(dyad1);
					dyads.add(dyad2);
				} else {
					dyads.add(dyad2);
					dyads.add(dyad1);
				}

				DyadRankingInstance instance = new DyadRankingInstance(dyads);
				dyadRankingInstances.add(instance);
			}
		});

		return new DyadRankingDataset(dyadRankingInstances);
	}

	public static void main(String[] args) throws FileNotFoundException, IOException, TrainingException, URISyntaxException {
		SQLAdapter sqlAdapter = new SQLAdapter(args[0], args[1], args[2], args[3]);
		DatasetGenerator generator = new DatasetGenerator(new int[] { 10, 20, 30 }, new DatasetFeatureRepresentationMap(sqlAdapter, "dataset_metafeatures_mirror"),
				new PipelineFeatureRepresentationMap(sqlAdapter, "dyad_dataset_approach_5_performance_samples_full"), new PipelinePerformanceStorage(sqlAdapter, "pipeline_performance_5_classifiers"));
		System.out.println("Initialized generator");
		DyadRankingDataset dataset = generator.generateTrainingDataset(0, 0);

		System.out.println("Generated data");
		String fileName = "split_0_100_samples.data";
		try (FileOutputStream stream = new FileOutputStream(new File(fileName))) {
			dataset.serialize(stream);
		}
		System.out.println("wrote data");

		try (FileInputStream stream = new FileInputStream(new File(fileName))) {
			DyadRankingDataset data = new DyadRankingDataset();
			data.deserialize(stream);
			System.out.println("read data");

			// test if training is possible
			FeatureTransformPLDyadRanker ranker = new FeatureTransformPLDyadRanker();
			ranker.train(data);
		}
	}
}
