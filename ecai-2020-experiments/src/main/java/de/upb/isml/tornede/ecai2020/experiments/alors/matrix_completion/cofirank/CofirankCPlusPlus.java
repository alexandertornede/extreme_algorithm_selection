package de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion.cofirank;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion.MatrixCompleterException;
import de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion.ModelBasedMatrixCompleter;

/**
 * A wrapper for the c++ implementation [0] of cofirank [1].
 *
 * <p>
 * [0] https://github.com/helegraf/cofirank; forked from
 * https://github.com/markusweimer/cofirank
 *
 * <p>
 * [1] Weimer, Markus, et al. "Cofi rank-maximum margin matrix factorization for
 * collaborative ranking." Advances in neural information processing systems.
 * 2008.
 *
 * @author helegraf
 *
 */
public class CofirankCPlusPlus implements ModelBasedMatrixCompleter {

	// logging
	private Logger logger = LoggerFactory.getLogger(CofirankCPlusPlus.class);

	// configuration
	private CofiConfig config;

	/**
	 * Creates a new Cofirank wrapper using the given configuration. Any execution
	 * of {@link #complete(double[][])} will execute cofirank with this
	 * configuration.
	 *
	 * @param config the configuration to be used
	 */
	public CofirankCPlusPlus(CofiConfig config) {
		this.config = config;
	}

	@Override
	public double[][] complete(double[][] matrix) throws CofiException {
		try {
			// write the matrix into COFI format (assume not huge non-sparse matrix)
			writeLSVMMatrix(config.getTestFilePath(), new double[0][0]);
			writeLSVMMatrix(config.getTrainFilePath(), matrix);

			// create configuration
			String configPath = config.createConfig();

			// Command to create an external process
			String command = config.getExecutablePath() + " " + configPath;

			// Running the above command
			logger.info("Running Cofirank");
			Runtime run = Runtime.getRuntime();
			Process proc = run.exec(command);

			// execute
			proc.waitFor();
			readCofiOutput(proc);

			// read the result
			return parseNonSparseLSVM("F.lsvm");

		} catch (IOException | InterruptedException e1) {
			throw new CofiException("Cofi-Run incomplete", e1);
		}
	}

	private void readCofiOutput(Process proc) throws CofiException {
		try {
			// Read any errors from the attempted command
			BufferedReader stdError = new BufferedReader(new InputStreamReader(proc.getErrorStream()));
			String firstLine = stdError.readLine();
			String line = stdError.readLine();

			if (line != null || !firstLine.startsWith("All output including logs will go to")) {
				StringBuffer buff = new StringBuffer();
				buff.append("Exception while running COFI!");
				buff.append(System.lineSeparator());
				buff.append(firstLine);
				buff.append(line);

				while ((line = stdError.readLine()) != null) {
					buff.append(line);
				}

				throw new CofiException(buff.toString());
			}
		} catch (IOException e) {
			throw new CofiException(e);
		}
	}

	private void writeLSVMMatrix(String location, double[][] matrix) throws IOException {
		logger.debug("Writing matrix to {}", location);

		try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File(location)))) {
			for (int i = 0; i < matrix.length; i++) {
				for (int j = 0; j < matrix[i].length; j++) {
					if (!Double.isNaN(matrix[i][j])) {
						writer.write(String.format(Locale.US, "%d:%f ", j + 1, matrix[i][j]));
					}
				}

				writer.write(System.lineSeparator());
			}
		}
	}

	@Override
	public double[][] getU() throws MatrixCompleterException {
		try {
			return parseNonSparseLSVM("U.lsvm");
		} catch (IOException e) {
			throw new MatrixCompleterException("Could not parse u-matrix", e);
		}
	}

	@Override
	public double[][] getV() throws MatrixCompleterException {
		try {
			return parseNonSparseLSVM("M.lsvm");
		} catch (IOException e) {
			throw new MatrixCompleterException("Could not parse v-matrix", e);
		}
	}

	private double[][] parseNonSparseLSVM(String locationRelativeToCOFIOutFolder) throws IOException {
		logger.debug("Parsing matrix {}", locationRelativeToCOFIOutFolder);

		try (BufferedReader reader = new BufferedReader(new FileReader(Paths.get(config.getOutFolderPath(), locationRelativeToCOFIOutFolder).toString()))) {
			String line = reader.readLine();

			List<double[]> u = new ArrayList<>();
			while (line != null) {
				// parse lsvm file
				line = line.trim();
				if (line != "") {
					String[] parts = line.split(" ");
					double[] entries = new double[parts.length];

					for (int i = 0; i < parts.length; i++) {
						entries[i] = Double.parseDouble(parts[i].split(":")[1]);
					}

					u.add(entries);

				}

				line = reader.readLine();
			}

			return u.toArray(new double[0][0]);
		}
	}

	public CofiConfig getConfig() {
		return config;
	}

	public void setConfig(CofiConfig config) {
		this.config = config;
	}

}
