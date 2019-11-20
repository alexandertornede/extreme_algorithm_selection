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
import weka.core.Attribute;

public class AlgorithmMetafeatureGenerator {

	private List<Component> components;

	private Map<String, Integer> featureToIndexMap;
	private int amountOfFeatures;

	public AlgorithmMetafeatureGenerator(Collection<Component> components) {
		this.components = new ArrayList<>(components);
		createFeatureToIndexMap();
	}

	public String[] generateMetafeatures(ComponentInstance instance) {
		Component component = instance.getComponent();
		String[] metafeatures = new String[amountOfFeatures];
		Arrays.fill(metafeatures, "0");

		metafeatures[featureToIndexMap.get(component.getName())] = "1";

		for (Parameter parameter : component.getParameters()) {
			String parameterValue = instance.getParameterValue(parameter);
			if (parameter.isCategorical()) {
				int parameterIndex = featureToIndexMap.get(getCategoricalParameterIdentifier(component, parameter, parameterValue));
				metafeatures[parameterIndex] = "1";
			} else {
				int parameterIndex = featureToIndexMap.get(getNumericalParameterIdentifier(component, parameter));
				metafeatures[parameterIndex] = parameterValue;

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
		return metafeatures;
	}

	private String getNumericalParameterIdentifier(Component component, Parameter parameter) {
		return component.getName() + "." + parameter.getName();
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
						featureToIndexMap.put(getCategoricalParameterIdentifier(component, parameter, value), amountOfFeatures);
						amountOfFeatures++;
					}
				} else {
					featureToIndexMap.put(getNumericalParameterIdentifier(component, parameter), amountOfFeatures);
					amountOfFeatures++;
				}
			}
		}
	}

	public ArrayList<Attribute> getWekaAttributeList() {
		ArrayList<Attribute> attributes = new ArrayList<>();
		for (Component component : components) {
			attributes.add(new Attribute(component.getName(), Arrays.asList("0", "1")));
			for (Parameter parameter : component.getParameters()) {
				if (parameter.isCategorical()) {
					CategoricalParameterDomain domain = (CategoricalParameterDomain) parameter.getDefaultDomain();
					for (String value : domain.getValues()) {
						attributes.add(new Attribute(getCategoricalParameterIdentifier(component, parameter, value), Arrays.asList("0", "1")));
					}
				} else {
					attributes.add(new Attribute(getNumericalParameterIdentifier(component, parameter)));
				}
			}
		}
		return attributes;
	}

	private String getCategoricalParameterIdentifier(Component component, Parameter parameter, String parameterValue) {
		return component.getName() + "." + parameter.getName() + "=" + parameterValue;
	}
}
