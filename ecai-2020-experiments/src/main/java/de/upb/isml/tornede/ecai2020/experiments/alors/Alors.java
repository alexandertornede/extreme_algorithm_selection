package de.upb.isml.tornede.ecai2020.experiments.alors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.upb.isml.tornede.ecai2020.experiments.alors.latent_features.FeaturePredictor;
import de.upb.isml.tornede.ecai2020.experiments.alors.latent_features.FeaturePredictorException;
import de.upb.isml.tornede.ecai2020.experiments.alors.latent_features.WEKAFeaturePredictor;
import de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion.MatrixCompleterException;
import de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion.ModelBasedMatrixCompleter;

/**
 * A simple java implementation for ALORS [0], algorithm recommender system.
 *
 * <p>
 * [0] Mısır, Mustafa, and Michèle Sebag. "Alors: An algorithm recommender
 * system." Artificial Intelligence 244 (2017): 291-314.
 *
 * @author helegraf
 *
 */
public class Alors {

	// logging
	private Logger logger = LoggerFactory.getLogger(Alors.class);

	// parameters
	private ModelBasedMatrixCompleter matrixCompleter;
	private FeaturePredictor featurePredictor = new WEKAFeaturePredictor();

	// results
	private double[][] v;
	private boolean isPrepared = false;

	/**
	 * Initialize Alors using the given matrix completer, which has to be model
	 * based as it has to produce latent features.
	 *
	 * @param matrixCompleter the model-based matrix completer
	 */
	public Alors(ModelBasedMatrixCompleter matrixCompleter) {
		this.matrixCompleter = matrixCompleter;
	}

	/**
	 * Completes the given matrix as well as learning a mapping from instance to
	 * latent features.
	 *
	 * @param matrixM (rows = instances (e.g. users/ datasets/ ...), columns = items
	 *            (e.g. movies/ algorithms/ ...)
	 * @param matrixX matrix of instance features (rows = instances, columns =
	 *            features)
	 * @return an estimated of a completed matrix m
	 * @throws MatrixCompleterException if the matrix could not be completed
	 *             correctly or the latent features not learnt
	 * @throws FeaturePredictorException if the feature predictor for the latent
	 *             features could not be built
	 */
	public double[][] completeMatrixAndPrepareColdStart(double[][] matrixM, double[][] matrixX) throws MatrixCompleterException, FeaturePredictorException {
		// do matrix completion for M
		logger.debug("Completing matrix with matrix completer {}", matrixCompleter.getClass());
		double[][] mHead = matrixCompleter.complete(matrixM);
		v = matrixCompleter.getV();

		// train model for feature vector
		logger.debug("Training feature predictor {}", featurePredictor.getClass());
		featurePredictor.train(matrixX, matrixCompleter.getU());

		isPrepared = true;
		return mHead;
	}

	/**
	 * Returns a prediction for the given instance.
	 *
	 * @param featureVectorX the instance features
	 * @return a prediction of item values, e.g. algorithm performances or ranks for
	 *         the case of algorithm selection
	 * @throws FeaturePredictorException if the latent features for the instance
	 *             could not be predicted
	 * @throws AlorsException if Alors has not been prepared for prediction
	 */
	public double[] predictForFeatures(double[] featureVectorX) throws FeaturePredictorException, AlorsException {
		if (!isPrepared) {
			throw new AlorsException("Alors has not been prepared for predictions.");
		}

		// feed into prediction model for rf; then multiply latent feature vector with
		// algorithm feature vector matrix
		double[] latentFeatures = featurePredictor.predict(featureVectorX);

		double[] result = new double[v.length];
		for (int i = 0; i < v.length; i++) {
			double entry = 0;
			for (int j = 0; j < v[i].length; j++) {
				entry += latentFeatures[j] * v[i][j];
			}
			result[i] = entry;
		}

		return result;
	}

	public ModelBasedMatrixCompleter getMatrixCompleter() {
		return matrixCompleter;
	}

	public void setMatrixCompleter(ModelBasedMatrixCompleter matrixCompleter) {
		this.matrixCompleter = matrixCompleter;
	}

	public FeaturePredictor getFeaturePredictor() {
		return featurePredictor;
	}

	public void setFeaturePredictor(FeaturePredictor featurePredictor) {
		this.featurePredictor = featurePredictor;
	}

	public boolean isPrepared() {
		return isPrepared;
	}
}
