package de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion;

/**
 * Indicated that a given matrix could not be completed.
 *
 * @author helegraf
 *
 */
public class MatrixCompleterException extends Exception {

	private static final long serialVersionUID = -683055500590824039L;

	public MatrixCompleterException(String msg) {
		super(msg);
	}

	public MatrixCompleterException(Throwable cause) {
		super(cause);
	}

	public MatrixCompleterException(String msg, Throwable cause) {
		super(msg, cause);
	}

}
