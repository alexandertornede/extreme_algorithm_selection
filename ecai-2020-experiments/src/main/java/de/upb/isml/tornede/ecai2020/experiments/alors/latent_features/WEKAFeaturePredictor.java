package de.upb.isml.tornede.ecai2020.experiments.alors.latent_features;

import java.util.ArrayList;
import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

/**
 * Predicts latent features using WEKA classifiers.
 *
 * @author helegraf
 *
 */
public class WEKAFeaturePredictor implements FeaturePredictor {

	// logging
	private Logger logger = LoggerFactory.getLogger(WEKAFeaturePredictor.class);

	// options
	private String classifierName = new RandomForest().getClass().getName();
	private String[] classifierOptions = new RandomForest().getOptions();

	// results
	private ArrayList<Classifier> regressors = new ArrayList<>();
	private ArrayList<Instances> datasets = new ArrayList<>();
	private boolean isPrepared = false;

	@Override
	public void train(double[][] featureMatrixX, double[][] featureMatrixU) throws FeaturePredictorException {
		// create a dataset for each classifier (each column in the predictable feature matrix U)
		ArrayList<Attribute> attInfo = new ArrayList<>();
		for (int i = 0; i < featureMatrixX[0].length; i++) {
			attInfo.add(new Attribute("att-" + i));
		}
		attInfo.add(new Attribute("target"));

		for (int i = 0; i < featureMatrixU[0].length; i++) {
			datasets.add(new Instances("data-" + i, attInfo, featureMatrixX.length));
			datasets.get(i).setClassIndex(attInfo.size() - 1);
		}

		for (int i = 0; i < featureMatrixX.length; i++) {
			for (int j = 0; j < featureMatrixU[0].length; j++) {
				datasets.get(j).add(new DenseInstance(1.0, addToArray(featureMatrixX[i], featureMatrixU[i][j])));
			}
		}

		// train classifiers
		logger.debug("Predicting with {}, options {}", classifierName, Arrays.toString(classifierOptions));
		for (int i = 0; i < datasets.size(); i++) {
			try {
				regressors.add(AbstractClassifier.forName(classifierName, classifierOptions));
				regressors.get(i).buildClassifier(datasets.get(i));
			} catch (Exception e1) {
				throw new FeaturePredictorException(e1);
			}
		}

		isPrepared = true;
	}

	@Override
	public double[] predict(double[] featureVectorX) throws FeaturePredictorException {
		if (!isPrepared) {
			throw new FeaturePredictorException("Predictor has not been prepared!");
		}

		double[] results = new double[regressors.size()];

		for (int i = 0; i < regressors.size(); i++) {
			DenseInstance testInstance = new DenseInstance(1.0, addToArray(featureVectorX, 0.0));
			testInstance.setDataset(datasets.get(i));
			try {
				results[i] = regressors.get(i).classifyInstance(testInstance);
			} catch (Exception e) {
				throw new FeaturePredictorException(e);
			}
		}

		return results;
	}

	private double[] addToArray(final double[] source, final double element) {
		final double[] destination = new double[source.length + 1];
		System.arraycopy(source, 0, destination, 0, source.length);
		destination[source.length] = element;
		return destination;
	}

	public boolean isPrepared() {
		return isPrepared;
	}

}
