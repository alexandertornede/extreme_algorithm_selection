package de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion.cofirank;

import de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion.MatrixCompleterException;

/**
 * Indicates that a matrix could not be completed because cofirank could not
 * properly be executed.
 *
 * @author helegraf
 *
 */
public class CofiException extends MatrixCompleterException {

	private static final long serialVersionUID = 6268379980300404390L;

	public CofiException(String msg) {
		super(msg);
	}

	public CofiException(Throwable cause) {
		super(cause);
	}

	public CofiException(String msg, Throwable cause) {
		super(msg, cause);
	}
}