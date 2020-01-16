package de.upb.isml.tornede.ecai2020.experiments.rankers.regression.peralgorithm;

import java.util.ArrayList;
import java.util.List;

import ai.libs.jaicore.basic.sets.Pair;
import weka.core.Attribute;
import weka.core.Instances;

public interface PerAlgorithmRegressionDatasetGenerator {

	public List<Pair<Integer, Instances>> generateTrainingDataset(List<Integer> trainingDatasetIds);

	public void initialize(long randomSeed);

	public String getName();

	public ArrayList<Attribute> getAttributeInfo();
}
