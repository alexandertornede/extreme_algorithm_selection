package de.upb.isml.tornede.ecai2020.experiments.alors.matrix_completion.cofirank;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Represents a configuration for {@link CofirankCPlusPlus}. Holds parameters
 * neccessary for executing cofirank and can create configuration files to be
 * read by the c++ executable of cofirank.
 *
 * @author helegraf
 *
 */
public class CofiConfig {
	// logging
	private Logger logger = LoggerFactory.getLogger(CofiConfig.class);

	// Where COFI
	private String executablePath;
	private String configurationPath;
	private String outFolderPath;

	private String trainFilePath;
	private String testFilePath;

	// COFI options (not complete)
	private int dimW = 10;

	private int userPhaseLambda = 5;
	private int moviePhaseLambda = 5;

	private double minProgress = 0.1;
	private int minIterations = 3;
	private int maxIterations = 30;

	private String optimizedMeasure = "REGRESSION";
	private int ndcgKLoss = 10;
	private double ndcgLossExponent = -0.25;

	private int numberOfRows;
	private int numberOfColumns;

	/**
	 * Create a new cofirank configuration with the given parameters that don't have
	 * default values. Other parameters can be adapted with corresponding setters.
	 * The class currently does not set all possible options of cofirank.
	 *
	 * @param executablePath the complete path to the executable of cofirank
	 *            (including the executable)
	 * @param configurationPath the complete path to the location of the
	 *            configuration file (including the file name ending
	 *            in .cfg, configuration will be created to here)
	 * @param outFolderPath the folder where the results will be written to,
	 *            should exist before executing cofirank
	 * @param trainFilePath the location of the training data file
	 * @param testFilePath the location of the testing data file (testing data
	 *            file must always exist but can be empty)
	 */
	public CofiConfig(String executablePath, String configurationPath, String outFolderPath, String trainFilePath, String testFilePath, int numberOfRows, int numberOfColumns) {
		this.executablePath = executablePath;
		this.configurationPath = configurationPath;
		this.outFolderPath = outFolderPath;
		this.trainFilePath = trainFilePath;
		this.testFilePath = testFilePath;
		this.numberOfRows = numberOfRows;
		this.numberOfColumns = numberOfColumns;
	}

	/**
	 * Creates a configuration file in the configured location with the configured
	 * options, to be read by the cofirank c++ executable.
	 *
	 * @return the path to the config file, to be given as an argument to cofirank
	 * @throws IOException if the config file cannot be written to the configured
	 *             location
	 */
	public String createConfig() throws IOException {
		File newConfig = new File(configurationPath);
		logger.debug("Writing configuration to {}", newConfig.getAbsolutePath());

		try (BufferedWriter writer = new BufferedWriter(new FileWriter(newConfig))) {

			// File location & output writing
			writer.write(String.format(Locale.US, "string cofibmrm.format SVMLIGHT%n"));
			writer.write(String.format(Locale.US, "string cofibmrm.DtrainFile %s%n", trainFilePath));
			writer.write(String.format(Locale.US, "string cofibmrm.DtestFile %s%n", testFilePath));
			writer.write(String.format(Locale.US, "string cofibmrm.evaluation WEAK%n"));
			writer.write(String.format(Locale.US, "string cofi.outfolder %s/%n", outFolderPath));

			// Training file
			writer.write(String.format(Locale.US, "int cofi.trainfile.size1 %d%n", numberOfRows));
			writer.write(String.format(Locale.US, "int cofi.trainfile.size2 %d%n", numberOfColumns));

			// Misc options
			writer.write(String.format(Locale.US, "int cofi.useOffset 0%n"));
			writer.write(String.format(Locale.US, "int cofi.useGraphkernel 0%n"));
			writer.write(String.format(Locale.US, "int cofi.dimW %d%n", dimW));

			// Model storage
			writer.write(String.format(Locale.US, "int cofi.storeU 1%n"));
			writer.write(String.format(Locale.US, "int cofi.storeM 1%n"));
			writer.write(String.format(Locale.US, "int cofi.storeF 1%n"));
			writer.write(String.format(Locale.US, "int cofi.storeModel 1%n"));

			// Lambdas
			writer.write(String.format(Locale.US, "double cofi.userphase.lambda %d%n", userPhaseLambda));
			writer.write(String.format(Locale.US, "double cofi.moviephase.lambda %d%n", moviePhaseLambda));

			// Adaptive regularization
			writer.write(String.format(Locale.US, "int cofi.useAdaptiveRegularization 0%n"));
			writer.write(String.format(Locale.US, "int cofi.adaptiveRegularization.uExponent 0%n"));
			writer.write(String.format(Locale.US, "double cofi.adaptiveRegularization.wExponent 0.5%n"));

			// Iteration Control
			writer.write(String.format(Locale.US, "double cofi.minProgress %f%n", minProgress));
			writer.write(String.format(Locale.US, "int cofi.minIterations %d%n", minIterations));
			writer.write(String.format(Locale.US, "int cofi.maxIterations %d%n", maxIterations));

			// Loss options
			writer.write(String.format(Locale.US, "string cofi.loss %s%n", optimizedMeasure));
			writer.write(String.format(Locale.US, "int loss.ndcg.trainK %d%n", ndcgKLoss));
			writer.write(String.format(Locale.US, "double loss.ndcg.c_exponent %f%n", ndcgLossExponent));

			// Evaluation options
			writer.write(String.format(Locale.US, "int cofi.eval.evaluateOnTestSet 0%n"));
			writer.write(String.format(Locale.US, "int cofi.eval.evaluateOnTrainSet 1%n"));
			writer.write(String.format(Locale.US, "int cofi.eval.binary 0%n"));
			writer.write(String.format(Locale.US, "int cofi.eval.ndcg 1%n"));
			writer.write(String.format(Locale.US, "int cofi.eval.ndcg.k 1%n"));
			writer.write(String.format(Locale.US, "int cofi.eval.norm 0%n"));
			writer.write(String.format(Locale.US, "int cofi.eval.rmse 1%n"));

			// BMRM options
			writer.write(String.format(Locale.US, "double bmrm.gammaTol 0.01%n"));
			writer.write(String.format(Locale.US, "double bmrm.epsilonTol -1.0%n"));
			writer.write(String.format(Locale.US, "int bmrm.maxIter 4000%n"));
		}

		return newConfig.getPath();
	}

	public String getExecutablePath() {
		return executablePath;
	}

	public void setExecutablePath(String executablePath) {
		this.executablePath = executablePath;
	}

	public String getConfigurationPath() {
		return configurationPath;
	}

	public void setConfigurationPath(String configurationPath) {
		this.configurationPath = configurationPath;
	}

	public String getOutFolderPath() {
		return outFolderPath;
	}

	public void setOutFolderPath(String outFolderPath) {
		this.outFolderPath = outFolderPath;
	}

	public String getTrainFilePath() {
		return trainFilePath;
	}

	public void setTrainFilePath(String trainFilePath) {
		this.trainFilePath = trainFilePath;
	}

	public String getTestFilePath() {
		return testFilePath;
	}

	public void setTestFilePath(String testFilePath) {
		this.testFilePath = testFilePath;
	}

	public int getDimW() {
		return dimW;
	}

	public void setDimW(int dimW) {
		this.dimW = dimW;
	}

	public int getUserPhaseLambda() {
		return userPhaseLambda;
	}

	public void setUserPhaseLambda(int userPhaseLambda) {
		this.userPhaseLambda = userPhaseLambda;
	}

	public int getMoviePhaseLambda() {
		return moviePhaseLambda;
	}

	public void setMoviePhaseLambda(int moviePhaseLambda) {
		this.moviePhaseLambda = moviePhaseLambda;
	}

	public double getMinProgress() {
		return minProgress;
	}

	public void setMinProgress(double minProgress) {
		this.minProgress = minProgress;
	}

	public int getMinIterations() {
		return minIterations;
	}

	public void setMinIterations(int minIterations) {
		this.minIterations = minIterations;
	}

	public int getMaxIterations() {
		return maxIterations;
	}

	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}

	public String getOptimizedMeasure() {
		return optimizedMeasure;
	}

	public void setOptimizedMeasure(String optimizedMeasure) {
		this.optimizedMeasure = optimizedMeasure;
	}

	public int getNdcgKLoss() {
		return ndcgKLoss;
	}

	public void setNdcgKLoss(int ndcgKLoss) {
		this.ndcgKLoss = ndcgKLoss;
	}

	public double getNdcgLossExponent() {
		return ndcgLossExponent;
	}

	public void setNdcgLossExponent(double ndcgLossExponent) {
		this.ndcgLossExponent = ndcgLossExponent;
	}
}
