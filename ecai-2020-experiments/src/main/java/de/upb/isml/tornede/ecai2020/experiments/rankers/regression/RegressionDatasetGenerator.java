package de.upb.isml.tornede.ecai2020.experiments.rankers.regression;

import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instances;

public interface RegressionDatasetGenerator {

	public Instances generateTrainingDataset(List<Integer> trainingDatasetIds, List<Integer> trainingPipelineIds);

	public void initialize(long randomSeed);

	public String getName();

	public ArrayList<Attribute> getAttributeInfo();

}
