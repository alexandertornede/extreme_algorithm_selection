package de.upb.isml.tornede.ecai2020.experiments.utils;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.json.JSONArray;
import org.json.JSONObject;

import ai.libs.jaicore.basic.sets.Pair;

public class Util {

	private Util() {
		// hiding the public constructor
	}

	public static Pair<List<Integer>, List<Integer>> getTrainingAndTestDatasetSplitsForSplitId(int datasetSplitId) throws IOException, URISyntaxException {

		List<String> fileContent = Files.readAllLines(Paths.get("splits/" + datasetSplitId + ".json"));

		StringBuilder stringBuilder = new StringBuilder();
		for (String line : fileContent) {
			stringBuilder.append(line);
		}

		JSONObject jsonObject = new JSONObject(stringBuilder.toString());

		JSONArray trainDatasetArray = jsonObject.getJSONArray("trainDatasets");

		List<Integer> trainingDatasetIds = new ArrayList<>(trainDatasetArray.length());
		for (int i = 0; i < trainDatasetArray.length(); i++) {
			trainingDatasetIds.add(trainDatasetArray.getInt(i));
		}

		JSONArray testDatasetArray = jsonObject.getJSONArray("testDatasets");

		List<Integer> testDatasetIds = new ArrayList<>(testDatasetArray.length());
		for (int i = 0; i < testDatasetArray.length(); i++) {
			testDatasetIds.add(testDatasetArray.getInt(i));
		}

		return new Pair<>(trainingDatasetIds, testDatasetIds);

	}

}
