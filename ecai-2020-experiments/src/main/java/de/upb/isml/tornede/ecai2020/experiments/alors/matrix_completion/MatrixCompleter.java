package de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion;

/**
 * A class giving the functionality of completing a given matrix, either memory-
 * or modelbased.
 *
 * @author helegraf
 *
 */
public interface MatrixCompleter {

	/**
	 * Estimate missing values for the given matrix; does not necessarily return the
	 * same values for entries that were already present in the given incomplete
	 * matrix.
	 *
	 * @param matrix the matrix to be completed
	 * @return an estimate of the matrix completion
	 * @throws MatrixCompleterException if the matrix cannot be completed
	 */
	public double[][] complete(double[][] matrix) throws MatrixCompleterException;
}
