package de.upb.isml.tornede.ecai2020.experiments.datasetgen;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.xml.DataSetDescription;
import org.openml.apiconnector.xml.Study;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class OpenMLUtil {

	public static void downloadOpenMLDatasets(String pathToStorageFolder) {
		OpenmlConnector con = new OpenmlConnector();
		Study study;
		try {
			study = con.studyGet(99);
			Integer[] datasetIDs = study.getDataset();
			for (int datasetId : datasetIDs) {
				DataSetDescription datasetDescription = con.dataGet(datasetId);
				File datasetFile = con.datasetGet(datasetDescription);
				File newDatasetFile = new File(pathToStorageFolder + "/" + datasetId + ".arff");
				datasetFile.renameTo(newDatasetFile);
				datasetFile.delete();
			}
		} catch (Exception e) {
			throw new RuntimeException("Could not get OpenML study", e);
		}
	}

	public static Map<Integer, Instances> getOpenMLDatasetIdsToInstancesMap() {
		Map<Integer, Instances> datasetIdToInstancesMap = new HashMap<>();
		OpenmlConnector con = new OpenmlConnector();
		Study study;
		try {
			study = con.studyGet(99);
			Integer[] datasetIDs = study.getDataset();
			for (int datasetId : datasetIDs) {
				if (datasetId != 40927) {
					System.out.println("Obtaining dataset " + datasetId);
					DataSetDescription datasetDescription = con.dataGet(datasetId);
					String targetAttribute = datasetDescription.getDefault_target_attribute();
					File datasetFile = con.datasetGet(datasetDescription);
					DataSource dataSource = new DataSource(datasetFile.getAbsolutePath());
					Instances instances = dataSource.getDataSet();
					instances.setClass(instances.attribute(targetAttribute));
					datasetIdToInstancesMap.put(datasetId, instances);
				}
			}
			return datasetIdToInstancesMap;
		} catch (Exception e) {
			throw new RuntimeException("Could not get OpenML study", e);
		}
	}

	public static List<Integer> getDatasetIds() {
		OpenmlConnector con = new OpenmlConnector();
		Study study;
		try {
			study = con.studyGet(99);
			Integer[] datasetIDs = study.getDataset();
			return Arrays.asList(datasetIDs);
		} catch (Exception e) {
			throw new RuntimeException("Could not get OpenML study", e);
		}
	}
}
