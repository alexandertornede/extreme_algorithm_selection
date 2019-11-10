package de.upb.isml.tornede.ecai2020.experiments.datasetgen.algorithm;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.stream.Collectors;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.hasco.serialization.ComponentLoader;
import ai.libs.mlplan.multiclass.wekamlplan.weka.WekaPipelineFactory;

public class AlgorithmGenerator {

	private Collection<Component> baseComponents;

	private Random random;

	public AlgorithmGenerator(long randomSeed) {
		this.random = new Random(randomSeed);
		ComponentLoader loader;
		try {
			loader = new ComponentLoader(new File("components/weka-singlelabel-base.json"), true);
			baseComponents = loader.getComponents();
		} catch (IOException e) {
			throw new RuntimeException("Could not load components file.", e);
		}
	}

	public List<ComponentInstance> generateRandomAlgorithmConfigurations(int maximumAmountPerAlgorithm) {
		List<ComponentInstance> generatedAlgorithms = new ArrayList<>();
		// make sure we filter out support vector kernels as we do not want to instantiate these
		List<Component> componentsToInstantiate = baseComponents.stream().filter(c -> !c.getName().startsWith("weka.classifiers.functions.supportVector")).collect(Collectors.toList());
		for (Component component : componentsToInstantiate) {
			int upperBoundOnNumberOfConfigurationsToSample = maximumAmountPerAlgorithm;
			for (int i = 0; i < upperBoundOnNumberOfConfigurationsToSample; i++) {
				ComponentInstance instance = getRandomComponentInstance(component);
				generatedAlgorithms.add(instance);
			}
		}
		return generatedAlgorithms.stream().distinct().collect(Collectors.toList());
	}

	public ComponentInstance getRandomComponentInstance(Component component) {
		ComponentInstance finalComponentInstance = null;
		while (!canInstantiateComponentInstance(finalComponentInstance)) {
			finalComponentInstance = getRandomPossiblyIncompatibleComponentInstance(component);
		}
		return finalComponentInstance;
	}

	private ComponentInstance getRandomPossiblyIncompatibleComponentInstance(Component component) {
		if (component.getRequiredInterfaces().isEmpty()) {
			return fixIntegerParameters(ComponentUtil.randomParameterizationOfComponent(component, random));
		}
		Map<String, String> requiredInterfaces = new HashMap<>(component.getRequiredInterfaces());
		Map<String, ComponentInstance> fulfilledInterfaces = new HashMap<>();
		for (Entry<String, String> entry : requiredInterfaces.entrySet()) {
			Collection<Component> componentsProvidingInterfaces = ComponentUtil.getComponentsProvidingInterface(baseComponents, entry.getValue());
			Component randomComponentProvidingInterface = componentsProvidingInterfaces.stream().collect(Collectors.toList()).get(random.nextInt(componentsProvidingInterfaces.size()));
			ComponentInstance randomComponentInstanceProvidingInterface = getRandomPossiblyIncompatibleComponentInstance(randomComponentProvidingInterface);
			fulfilledInterfaces.put(entry.getKey(), randomComponentInstanceProvidingInterface);
		}
		ComponentInstance instance = ComponentUtil.randomParameterizationOfComponent(component, random);
		Map<String, ComponentInstance> satisfiedInterfacesOfComponentInstance = instance.getSatisfactionOfRequiredInterfaces();
		for (Entry<String, ComponentInstance> entry : fulfilledInterfaces.entrySet()) {
			satisfiedInterfacesOfComponentInstance.put(entry.getKey(), entry.getValue());
		}
		return fixIntegerParameters(instance);
	}

	private boolean canInstantiateComponentInstance(ComponentInstance componentInstance) {
		if (componentInstance == null) {
			return false;
		}
		try {
			WekaPipelineFactory factory = new WekaPipelineFactory();
			factory.getComponentInstantiation(componentInstance);
			return true;
		} catch (Exception ex) {
			return false;
		}
	}

	public ComponentInstance fixIntegerParameters(ComponentInstance instance) {
		Map<String, String> parameterNameToValueMap = instance.getParameterValues();
		for (Entry<String, String> entry : parameterNameToValueMap.entrySet()) {
			String trimmedEntryValue = entry.getValue().trim();
			if (instance.getComponent().getParameterWithName(entry.getKey()).isNumeric()) {
				if (trimmedEntryValue.endsWith(".0")) {
					parameterNameToValueMap.put(entry.getKey(), trimmedEntryValue.substring(0, trimmedEntryValue.length() - 2));
				}
			}
		}
		return instance;
	}

	public Collection<Component> getBaseComponents() {
		return baseComponents;
	}

}
