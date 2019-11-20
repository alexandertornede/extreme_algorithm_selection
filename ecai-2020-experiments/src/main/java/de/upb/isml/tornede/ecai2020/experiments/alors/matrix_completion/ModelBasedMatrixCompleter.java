package de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion;

/**
 * A class that completes a matrix model-based; by factorizing it into two
 * matrices U and V that together estimate a given matrix. The matrices U and V
 * thus represent latent features.
 *
 * @author helegraf
 *
 */
public interface ModelBasedMatrixCompleter extends MatrixCompleter {

	/**
	 * Get the latent features for the instances, e.g. users in the movie
	 * recommendation setting or datasets in the algorithm recommendation setting.
	 *
	 * @return the feature matrix of instance features
	 * @throws MatrixCompleterException if the instance features cannot be retrieved
	 */
	public double[][] getU() throws MatrixCompleterException;

	/**
	 * Get the latent features for the item, e.g. movies in the movie recommendation
	 * setting or algorithms in the algorithm recommendation setting.
	 *
	 * @return the feature matrix of item features
	 * @throws MatrixCompleterException if the item features cannot be retrieved
	 */
	public double[][] getV() throws MatrixCompleterException;
}
