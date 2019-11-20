package de.upb.isml.tornede.ecai2020.experiments.alors.latent_features;

/**
 * Represents a very general multi-target regressor, used in the context of
 * ALORS to predict latent features from instance features.
 *
 * @author helegraf
 *
 */
public interface FeaturePredictor {

	/**
	 * Train the feature predictor on the given instance and latent features.
	 *
	 * @param featureMatrixX the instance features
	 * @param featureMatrixU the latent features to predict
	 * @throws FeaturePredictorException if the training is not successful
	 */
	public void train(double[][] featureMatrixX, double[][] featureMatrixU) throws FeaturePredictorException;

	/**
	 * Make a prediction of latent features for given instance features.
	 *
	 * @param featureVectorX the instance features
	 * @return a prediction of latent features
	 * @throws FeaturePredictorException if the prediction was not possible
	 */
	public double[] predict(double[] featureVectorX) throws FeaturePredictorException;
}
