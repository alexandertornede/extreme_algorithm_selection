package de.upb.isml.tornede.ecai2020.experiments.datasetgen.algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ai.libs.hasco.model.CategoricalParameterDomain;
import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.Parameter;

public class AlgorithmMetafetureGenerator {

	private List<Component> components;

	private Map<String, Integer> featureToIndexMap;
	private int amountOfFeatures;

	public AlgorithmMetafetureGenerator(Collection<Component> components) {
		this.components = new ArrayList<>(components);
		createFeatureToIndexMap();
	}

	public String[] generateMetafeatures(ComponentInstance instance) {
		Component component = instance.getComponent();
		String[] metafeatures = new String[amountOfFeatures];
		Arrays.fill(metafeatures, "0");
		Map<String, String> totalValuesExceptForRequiredInterfaces = new HashMap<>();

		metafeatures[featureToIndexMap.get(component.getName())] = "1";

		totalValuesExceptForRequiredInterfaces.put(component.getName(), "1");

		for (Parameter parameter : component.getParameters()) {
			String parameterValue = instance.getParameterValue(parameter);
			if (parameter.isCategorical()) {
				int parameterIndex = featureToIndexMap.get(getCombinedParameterIdentifier(parameter, parameterValue));
				metafeatures[parameterIndex] = "1";

				totalValuesExceptForRequiredInterfaces.put(getCombinedParameterIdentifier(parameter, parameterValue), "1");
			} else {
				int parameterIndex = featureToIndexMap.get(parameter.getName());
				metafeatures[parameterIndex] = parameterValue;

				totalValuesExceptForRequiredInterfaces.put(parameter.getName(), parameterValue);
			}
		}
		for (Entry<String, ComponentInstance> entry : instance.getSatisfactionOfRequiredInterfaces().entrySet()) {
			String[] featureRepresentationOfSatisfiedInterface = generateMetafeatures(entry.getValue());
			for (int i = 0; i < featureRepresentationOfSatisfiedInterface.length; i++) {
				String featureValue = featureRepresentationOfSatisfiedInterface[i];
				if (!featureValue.equals("0")) {
					if (!metafeatures[i].equals("0")) {
						throw new RuntimeException("Trying to overwrite feature value " + metafeatures[i] + " with " + featureValue);
					}
					metafeatures[i] = featureValue;
				}
			}
		}
		System.out.println(totalValuesExceptForRequiredInterfaces);
		return metafeatures;
	}

	private void createFeatureToIndexMap() {
		featureToIndexMap = new HashMap<>();
		for (Component component : components) {
			featureToIndexMap.put(component.getName(), amountOfFeatures);
			amountOfFeatures++;
			for (Parameter parameter : component.getParameters()) {
				if (parameter.isCategorical()) {
					CategoricalParameterDomain domain = (CategoricalParameterDomain) parameter.getDefaultDomain();
					for (String value : domain.getValues()) {
						featureToIndexMap.put(getCombinedParameterIdentifier(parameter, value), amountOfFeatures);
						amountOfFeatures++;
					}
				} else {
					featureToIndexMap.put(parameter.getName(), amountOfFeatures);
					amountOfFeatures++;
				}
			}
		}
	}

	private String getCombinedParameterIdentifier(Parameter parameter, String parameterValue) {
		return parameter.getName() + "=" + parameterValue;
	}
}
