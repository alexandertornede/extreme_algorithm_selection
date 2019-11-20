package de.upb.isml.tornede.ecai2020.experiments.alors.latent_features;

/**
 * Indicates that a feature predictor has a problem learning or prediction
 * latent features.
 *
 * @author helegraf
 *
 */
public class FeaturePredictorException extends Exception {

	private static final long serialVersionUID = 4087026993950208774L;

	public FeaturePredictorException(Throwable cause) {
		super(cause);
	}

	public FeaturePredictorException(String message) {
		super(message);
	}

}
